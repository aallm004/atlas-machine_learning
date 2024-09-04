#!/usr/bin/env python3
"""Defines summation_i_squared"""


def summation_i_squared(n):
    """calculates sum of squares of numbers"""
    if not isinstance(n, int) or n < 1:
        return None

    result = (n * (n + 1) * (2 * n + 1)) // 6

    return result
