import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import config

FORMATTER = logging.Formatter(fmt="%(asctime)s — %(name)s — %(levelname)s — %(message)s", datefmt='%Y-%m-%d %H:%M:%S')


def __get_console_handler():
    """
    configure to sends logger messages to console
    :return: logging handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    """
    create and returns a logger
    :param logger_name: logger name to use
    :return: the logger configured
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(__get_console_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


# log reference to all
log = get_logger("SMP")
