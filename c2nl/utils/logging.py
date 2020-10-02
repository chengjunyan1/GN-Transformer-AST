# -*- coding: utf-8 -*-
# src: https://github.com/wasiahmad/NeuralCodeSum/blob/master/c2nl/utils/logging.py
from __future__ import absolute_import

import logging

logger = logging.getLogger()


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger
