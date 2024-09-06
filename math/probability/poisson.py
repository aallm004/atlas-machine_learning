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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)

        if k < 0:
            return 0
        else:
            factorial = 1
            for i in range(1, k + 1):
                factorial *= i
            pmf = ((self.lambtha ** k) * (2.7182818285 ** -self.lambtha)
                   ) / factorial

        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        k = int(k)

        if k < 0:
            return 0
        else:
            cdf_v = 0
            for i in range(k + 1):
                factorial = 1
                for n in range(1, i+1):
                    factorial *= n

                cdf_v += (2.7182818285 ** -self.lambtha) * ((self.lambtha ** i
                                                             ) / factorial)

            return cdf_v
