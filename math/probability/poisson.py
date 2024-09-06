#!/usr/bin/env python3
"""poisson distrubution"""


class Poisson:
    """Poisson class"""

    def __init__(self, data=None, lambtha=1.0):
        """Poisson constructor"""
        self.lambtha = float(lambtha)

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (sum(data)) / len(data)
