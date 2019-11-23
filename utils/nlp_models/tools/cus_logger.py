# -*- coding:utf-8 -*-
import logging
import sys

mswindows = (sys.platform == "win32")


def get_logger(name):
    logger = logging.getLogger(name)
    if mswindows:
        ch = logging.FileHandler(r"D:\code\github\zxx\data\log\run.log", encoding='utf-8')
    else:
        ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s] [%(module)s.%(funcName)s(%(filename)s:%(lineno)d)] %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
