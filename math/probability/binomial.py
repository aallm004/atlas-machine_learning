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

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data))

            self.p = 1 - (variance / mean)

            self.n = round(mean / self.p)

            self.p = mean / self.n

    def factorial(self, x):
        """finds the factorial"""
        if x == 0 or x == 1:
            return 1
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result

    def comb(self, n, k):
        """finds the binomial coefficient"""
        return self.factorial(n) // (self.factorial(k) * self.factorial(n - k))

    def pmf(self, k):
        """calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        binomial_coefficient = self.comb(self.n, k)
        return binomial_coefficient * (self.p ** k) * (
            (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """calculates the value of the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        
            return cdf
