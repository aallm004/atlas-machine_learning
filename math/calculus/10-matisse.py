#!/usr/bin/env python3
"""function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    # Check if the input is a valid list and not empty
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # If polynomial is constnt, its derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate the derivative
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    # If all coeff became zero, return [0]
    if len(derivative) == 0:
        return [0]

    return derivative
