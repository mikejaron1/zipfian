"""
"""
import sqlite3


class NameException(Exception):
    pass


class SolarPerformance(object):
    """
    """

    dbname = 'test.db'

    def __init__(self, systemname):
        """
        """

        conn = sqlite3.connect(SolarPerformance.dbname)
        cur = conn.cursor()
        if ';' in systemname:
            raise NameException('invalid name: "%s"' % systemname)
        cur.execute("SELECT * FROM systems WHERE name='%s' LIMIT 1;"
                    % systemname)
        if not cur.fetchone():
            raise NameException('name "%s" does not exist in database'
                                % systemname)
        self.systemname = systemname

    @classmethod
    def names(self):
        """
        """

        conn = sqlite3.connect(SolarPerformance.dbname)
        cur = conn.cursor()
        cur.execute("SELECT * FROM systems;")
        names = [name for i, name in cur.fetchall()]
        conn.close()
        return names

    def name(self):
        """
        """

        return self.systemname

    def lifetimeperformance(self):
        """
        """

        con = sqlite3.connect(SolarPerformance.dbname)
        cur = con.cursor()
        cur.execute("""SELECT SUM(actualkwh) / SUM(expectedkwh)
                       FROM data
                       JOIN systems
                       ON
                           data.systemid=systems.systemid AND
                           systems.name='%s';"""
                    % self.systemname)
        value = cur.fetchone()[0]
        cur.close()
        return value
