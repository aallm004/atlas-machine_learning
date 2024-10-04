#!/usr/bin/env python3
"""documentation"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of data set

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    moving average calculatoin uses bias correction

    Returns: list containing the moving averages of data
    """
    weighted_average = 0
    moving_average = []
    for x, i in enumerate(data, 1):
        weighted_average = beta * weighted_average + (1 - beta) * x
        moving_average.append(weighted_average / (1 - beta**i))
    return moving_average
