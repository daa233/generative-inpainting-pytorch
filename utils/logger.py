import os
import sys
import datetime
import logging


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return str(datetime.datetime.now()).replace('-', '') \
        .replace(' ', '').replace(':', '').replace('.', '')


def get_logger(checkpoint_path):
    log_filename = date_uid()
    logger = logging.getLogger(log_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    stream_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(os.path.join(checkpoint_path, log_filename + '.log'))
    stream_hdlr.setFormatter(formatter)
    file_hdlr.setFormatter(formatter)
    logger.addHandler(stream_hdlr)
    logger.addHandler(file_hdlr)
    logger.setLevel(logging.INFO)
    return logger
