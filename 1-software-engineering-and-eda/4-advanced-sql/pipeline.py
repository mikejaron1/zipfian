import psycopg2
from datetime import datetime

conn = psycopg2.connect(dbname='socialmedia', user='giovanna', host='/tmp')
c = conn.cursor()

date = datetime.today().strftime("%Y%m%d")

c.execute(
    '''CREATE TABLE tmp_friends_%s AS 
    SELECT userid1 AS userid, COUNT(*) AS num_friends
    FROM
    ((SELECT userid1, userid2 FROM friends)
    UNION
    (SELECT userid2, userid1 FROM friends) a
    GROUP BY userid1;
    ''' % date
)

c.execute(
    '''CREATE TABLE tmp_logins_%s AS
    SELECT a.userid, a.cnt AS logins_7d_mobile, b.cnt AS logins_7d_web
    FROM
    (SELECT userid, COUNT(*) AS cnt
    FROM logins
    WHERE logins.tmstmp > current_date - 7 AND type='mobile'
    GROUP BY userid) a
    JOIN
    (SELECT userid, COUNT(*) AS cnt
    FROM logins
    WHERE logins.tmstmp > current_date - 7 AND type='web'
    GROUP BY userid) b
    ON a.userid=b.userid;
    ''' % date
)

c.execute(
    '''CREATE TABLE users_%s AS
    SELECT a.userid, a.reg_date, a.last_login, f.num_friends,
        COALESCE(l.logins_7d_mobile + l.logins_7d_web, 0) AS logins_7d,
        COALESCE(l.logins_7d_mobile, 0) AS logins_7d_mobile,
        COALESCE(l.logins_7d_web, 0) AS logins_7d_web,
        CASE WHEN optout.userid IS NULL THEN 0 ELSE 1 END
    FROM
    (SELECT r.userid, r.tmstmp::date AS reg_date, MAX(l.tmstmp::date) AS last_login
    FROM registrations r
    LEFT OUTER JOIN logins l
    ON r.userid=l.userid
    GROUP BY r.userid, r.tmstmp) a
    LEFT OUTER JOIN tmp_friends_%s f
    ON f.userid=a.userid
    LEFT OUTER JOIN tmp_logins_%s l
    ON l.userid=a.userid
    LEFT OUTER JOIN optout
    ON a.userid=optout.userid;
    ''' % (date, date, date)
)

conn.commit()
conn.close()
