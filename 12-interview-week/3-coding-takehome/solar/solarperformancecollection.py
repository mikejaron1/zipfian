"""
SolarPerformanceColleciton is a dataset to hold system production data.
It is designed to be efficient, both in time and storage space.
"""

import sqlite3


class SolarPerformanceCollection(object):
    """
    """

    def __init__(self):
        """
        """
        self.systems = {}

    def add(self, system):
        """ ADD a system to the collection

        usage:
            system = SolarPerformance('Sleepy')
            spc = SolarPerformanceCollection()
            spc.add(system)
        """
        self.systems[system.name()] = system

    def count(self):
        """
        """
        return len(self.systems)

    def max(self):
        """
        """
        return max(sp.lifetimeperformance()
                   for name, sp in self.systems.iteritems())

    def min(self):
        """
        """
        return min(sp.lifetimeperformance()
                   for name, sp in self.systems.iteritems())

    def percentile(self, pct):
        """ Calculate the nth percentile of the data collection

        usage:
        tenthpercentileperformance = spc.percentile(10)
        """
        num = len(self.systems) * pct / 100
        values = [sp.lifetimeperformance()
                  for name, sp in self.systems.iteritems()]
        values.sort(reverse=True)
        return values[:num]

    def top(self, k):
        """ Return an array of dictionaries of performance
        (lifetimeperformance) and names (systemname) for the top k systems.
        The results should be ordered in descending order.

        usage:
        systemperformance = spc.top(10)

        where systemperformance = [
                {'systemname': 'Sleepy', 'lifetimeperformance': 1.10},
                {'systemname': 'Doc', 'lifetimeperformance': 1.08,
                ...
            ]
        """
        systemperformance = [{'systemname': name,
                             'lifetimeperformance': sp.lifetimeperformance()}
                             for name, sp in self.systems.iteritems()]
        systemperformance.sort(key=lambda x: x['lifetimeperformance'],
                               reverse=True)
        return systemperformance[:k]
