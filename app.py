"""docstring for packages."""
import time
import os
import logging
import signal
from datetime import datetime
from multiprocessing import Pool, Process, Queue
from multiprocessing import cpu_count
from functools import partial
from queue import Empty as EmptyQueueException
import tornado.ioloop
import tornado.web
from prometheus_client import Gauge, generate_latest, REGISTRY
from prometheus_api_client import PrometheusConnect, Metric
from configuration import Configuration
import model
import model_fourier
import model_lstm
import schedule

state = "running"

# Set up logging
_LOGGER = logging.getLogger(__name__)

# Configured Prometheus metric names list
METRICS_LIST = Configuration.metrics_list

MODEL_LIST = {
    "prophet": model,
    "fourier": model_fourier,
    "lstm": model_lstm
}

# Set of unique metric series
UNIQUE_SERIES_SET = set()

# List of ModelPredictor Objects shared between processes
PREDICTOR_MODEL_LIST = list()

# A gauge set for the predicted values
GAUGE_DICT = dict()

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)


class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue):
        """Check if new predicted values are available in the queue before the get request."""
        try:
            model_list = data_queue.get_nowait()
            self.settings["model_list"] = model_list
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # Update metric value on every request and publish the metric
        for predictor_model in self.settings["model_list"]:
            # Get the current metric value so that it can be compared with the
            # predicted values
            try:
                current_metric_value = Metric(
                    pc.get_current_metric_value(
                        metric_name=predictor_model.metric.metric_name,
                        label_config=predictor_model.metric.label_config,
                    )[0]
                )
            except IndexError:
                continue

            metric_name = predictor_model.metric.metric_name
            prediction = predictor_model.predict_value(datetime.now())

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (
                current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]
            ) and (
                current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]
            ):
                anomaly = 0

            # Create a new time series that has value_type=anomaly
            # This value is 1 if an anomaly is found 0 if not
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app(data_queue):
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    return tornado.web.Application(
        [
            (r"/metrics", MainHandler, dict(data_queue=data_queue)),
            (r"/", MainHandler, dict(data_queue=data_queue)),
        ]
    )

def update_model_list():
    global UNIQUE_SERIES_SET
    global PREDICTOR_MODEL_LIST
    global GAUGE_DICT

    _LOGGER.info("Updating predictor model list from Prometheus metrics")

    for metric_name in METRICS_LIST:
        metric_list = pc.get_current_metric_value(metric_name=metric_name)

        for unique_metric in metric_list:
            metric = Metric(unique_metric)
            unique_key = metric.metric_name + "-" + "-".join(metric.label_config.values())

            # Initialize a predictor for new metrics
            if unique_key not in UNIQUE_SERIES_SET:
                _LOGGER.info(
                    "New predictor model from Prometheus metrics: %s",
                    unique_key
                )

                UNIQUE_SERIES_SET.add(unique_key)

                predictor = MODEL_LIST[Configuration.model].MetricPredictor(
                    unique_metric,
                    rolling_data_window_size=Configuration.rolling_training_window_size,
                )

                PREDICTOR_MODEL_LIST.append(predictor)

                if metric.metric_name not in GAUGE_DICT:
                    label_list = list(metric.label_config.keys())
                    label_list.append("value_type")

                    GAUGE_DICT[metric.metric_name] = Gauge(
                        metric.metric_name + "_" + predictor.model_name,
                        predictor.model_description,
                        label_list,
                    )

def train_individual_model(predictor_model, initial_run):
    metric_to_predict = predictor_model.metric
    pc = PrometheusConnect(
        url=Configuration.prometheus_url,
        headers=Configuration.prom_connect_headers,
        disable_ssl=True,
    )

    data_start_time = datetime.now() - Configuration.metric_chunk_size
    if initial_run:
        data_start_time = (
            datetime.now() - Configuration.rolling_training_window_size
        )

    # Download new metric data from prometheus
    new_metric_data = pc.get_metric_range_data(
        metric_name=metric_to_predict.metric_name,
        label_config=metric_to_predict.label_config,
        start_time=data_start_time,
        end_time=datetime.now(),
    )[0]

    # Train the new model
    start_time = datetime.now()
    predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes)

    _LOGGER.info(
        "Total Training time taken = %s, for metric: %s %s",
        str(datetime.now() - start_time),
        metric_to_predict.metric_name,
        metric_to_predict.label_config,
    )
    return predictor_model

def train_model(initial_run=False, data_queue=None):
    """Train the machine learning model."""
    global PREDICTOR_MODEL_LIST
    parallelism = min(Configuration.parallelism, cpu_count())
    _LOGGER.info(f"Training models using ProcessPool of size:{parallelism}")
    training_partial = partial(train_individual_model, initial_run=initial_run)
    with Pool(parallelism) as p:
        result = p.map(training_partial, PREDICTOR_MODEL_LIST)
    PREDICTOR_MODEL_LIST = result
    data_queue.put(PREDICTOR_MODEL_LIST)

def scheduled_job(data_queue=None):
    update_model_list()
    train_model(initial_run=False, data_queue=data_queue)

def terminateProcess(signalNumber, frame):
    _LOGGER.info("Received signal - terminating the process")
    global state
    state = "terminating"

if __name__ == "__main__":
    # Trap TERM and INT for process termination.
    signal.signal(signal.SIGTERM, terminateProcess)
    signal.signal(signal.SIGINT, terminateProcess)

    # Queue to share data between the tornado server and the model training
    predicted_model_queue = Queue()

    # Update model list
    update_model_list()

    # Initial run to generate metrics, before they are exposed
    train_model(initial_run=True, data_queue=predicted_model_queue)

    # Set up the tornado web app
    app = make_app(predicted_model_queue)
    app.listen(8080)
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    # Start up the server to expose the metrics.
    server_process.start()

    # Schedule model list updates and model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        scheduled_job, data_queue=predicted_model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    # Run scheduled jobs
    while state == "running":
        schedule.run_pending()
        time.sleep(1)

    # Terminating ...

    # Drain the queue
    while not predicted_model_queue.empty():
        predicted_model_queue.get_nowait()

    # Kill and join the server process
    if server_process.is_alive():
        server_process.kill()
        server_process.join(timeout=3)
