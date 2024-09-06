import logging

# Set up logging for evaluation
eval_log_filename = "evals.log"
eval_logger = logging.getLogger("eval_logger")
eval_logger.setLevel(logging.INFO)
eval_handler = logging.FileHandler(eval_log_filename)
eval_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
eval_logger.addHandler(eval_handler)


def log_evaluation_result(component_name, metric_name, value):
    """Log evaluation results to evals.log."""
    eval_logger.info(
        f"Component: {component_name}, Metric: {metric_name}, Value: {value}"
    )
