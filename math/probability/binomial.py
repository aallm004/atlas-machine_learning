#!/usr/bin/env python3
"""binominominomial"""


class Binomial:
    """Class that represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""

        if data is None:
            self.data = data
            self.n = int(n)
            self.p = float(p)
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.p = sum(data) / len(data)

            self.n = round(max(data) / self.p)

            self.p = sum(data) / (self.n * len(data))

