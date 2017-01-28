"""
SolarPerformance Test Set
"""
import unittest
from solar import SolarPerformance, NameException
import sqlite3


EXPECTEDRESULTS = {
    'Sneezy': 0.919177337397975,
    'Sleepy': 1.18652810676406,
    'Mickey Mouse': 1.05477555768535,
    'Dopey': 0.966604278620813,
    'Doc': 1.10329187448733,
    'Grumpy': 0.702467910341306,
    'Bashful': 1.00987703264902,
    'Happy': 1.11151164379285
}


class TestSolarPerformance(unittest.TestCase):
    """
    SolarPerformance Test Set
    """
    def setUp(self):
        self.expectedkeys = EXPECTEDRESULTS.keys()
        self.name = self.expectedkeys[0]

    def tearDown(self):
        pass

    def test_init(self):
        """
        Class should be initialized with a systemname, and should load data
        from the database
        """
        _ = SolarPerformance(self.name)

    def test_names(self):
        """
        Class should have a class method to list all of the system names
        """

        expectedkeys = self.expectedkeys
        expectedkeys.sort()

        actualkeys = SolarPerformance.names()
        actualkeys.sort()

        self.assertEqual(actualkeys, expectedkeys)

    def test_name(self):
        """
        Instance should have a public method 'name' which returns the
        system name
        """

        instance = SolarPerformance(self.name)
        self.assertEqual(instance.name(), self.name)

    def test_lifetimeperformance(self):
        """
        Instance should have a public method 'lifetimeperformance' which
        returns the system's lifetime performance, calculated as:

        lifetimeperformance = sum(actualkwh) / sum(expectedkwh)
        """

        instance = SolarPerformance(self.name)
        lifetimeperformance = instance.lifetimeperformance()
        self.assertTrue(isinstance(lifetimeperformance, float))
        self.assertTrue(lifetimeperformance > 0.0)
        self.assertTrue(lifetimeperformance > 0.0)
        self.assertAlmostEqual(lifetimeperformance, EXPECTEDRESULTS[self.name])

    def test_nonexistantname(self):
        """
        Class should return a useful error message if a non-existant
        name is provided to initialize
        """

        self.assertRaises(NameException, SolarPerformance, 'Silly Name')

    def test_sqlinjection(self):
        """
        Class should not be fooled by tricky sqlinjection
        """

        self.assertRaises(NameException,
                          SolarPerformance,
                          "'; drop table systems; --")
        con = sqlite3.connect(SolarPerformance.dbname)
        cur = con.cursor()
        cur.execute('''SELECT name
                       FROM sqlite_master
                       WHERE TYPE='table' AND NAME='systems';''')
        self.assertTrue(len(cur.fetchall()) == 1, "systems table dropped")
        con.close()

if __name__ == "__main__":
    unittest.main()
