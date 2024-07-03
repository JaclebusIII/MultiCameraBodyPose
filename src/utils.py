import logging
import logging.config
from os import path

module_logging_name = "MultiCameraBodyPose"
module_logger = logging.getLogger(module_logging_name)


def get_logger(logger_name: str) -> logging.Logger:
  return logging.getLogger(f"{module_logging_name}.{logger_name}")


def set_module_log_level(log_level: str) -> None:
  global module_logger
  module_logger.setLevel(log_level)
