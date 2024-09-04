#!/usr/bin/env python3
"""function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""

    # Check if poly is a valid list
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if C is a valid int
    if not isinstance(C, int):
        return None

    # Calculate integral
    result = [C]
    for i, coeff in enumerate(poly):

        # Check if coeff is valid
        if not isinstance(coeff, (int, float)):
            return None

        # Calculate new coefficient
        new_coeff = coeff / (i + 1)

        # Convert to int if possible
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)

        result.append(new_coeff)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
