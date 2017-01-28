
1. Number of users who have registered each day.

    ```sql
    SELECT tmstmp::date AS date, COUNT(*) AS cnt
    FROM registrations
    GROUP BY tmstmp::date
    ORDER BY tmstmp::date;
    ```

1. Number of users who have registered each day of the week.

    ```sql
    SELECT EXTRACT(dow FROM tmstmp), COUNT(*) AS cnt
    FROM registrations
    GROUP BY EXTRACT(dow FROM tmstmp)
    ORDER BY COUNT(*);
    ```

    Day 6 (Saturday) has the most registrations.

1. Users who haven't logged in in the last 7 days and have not opted out of receiving email. The data is from August 14, 2014, so take the last 7 days as of that date.

    Two ways. This is a double except. First take away the users who have not logged in in the last 7 days. Then take away the users who have opted out of email.

    ```sql
    SELECT r.userid
    FROM registrations AS r
    LEFT OUTER JOIN logins AS l
    ON
        r.userid=l.userid AND
        l.tmstmp>timestamp '2014-08-07'
    LEFT OUTER JOIN optout AS o
    ON r.userid=o.userid
    WHERE
        l.userid IS NULL AND
        o.userid IS NULL
    ORDER BY userid;
    ```

    ```sql
    SELECT userid
    FROM registrations
    EXCEPT
    SELECT userid
    FROM logins
    WHERE tmstmp > timestamp '2014-08-07'
    EXCEPT
    SELECT userid
    FROM optout
    ORDER BY userid;
    ```

1. Number of users who registered on the same day.

    ```sql
    SELECT a.userid, COUNT(1)
    FROM registrations a JOIN registrations b
    ON
        date_part('year', a.tmstmp)=date_part('year', b.tmstmp) AND
        date_part('doy', a.tmstmp)=date_part('doy', b.tmstmp)
    GROUP BY a.userid;
    ```

1. Users who have logged in on mobile more times than web and are in test group A.

    ```sql
    SELECT u.userid
    FROM (SELECT mobile.userid
          FROM (SELECT userid, COUNT(*) AS cnt
                FROM logins
                WHERE type='mobile'
                GROUP BY userid) mobile
          JOIN (SELECT userid, COUNT(*) AS cnt
                FROM logins
                WHERE type='web'
                GROUP BY userid) web
          ON mobile.userid=web.userid AND mobile.cnt > web.cnt) u
    JOIN test_group t
    ON u.userid=t.userid AND t.grp='A';
    ```

1. Most communicated with user.

    ```sql
    WITH num_messages AS (
        SELECT a.usr, a.other, a.cnt + b.cnt AS cnt
        FROM (SELECT sender AS usr, recipient AS other, COUNT(*) AS cnt
              FROM messages
              GROUP BY sender, recipient) a
        JOIN (SELECT recipient AS usr, sender AS other, COUNT(*) AS cnt
              FROM messages
              GROUP BY sender, recipient) b
        ON a.usr=b.usr AND a.other=b.other)

    SELECT num_messages.usr, num_messages.other, cnt
    FROM num_messages
    JOIN (SELECT usr, MAX(cnt) AS max_cnt
          FROM num_messages
          GROUP BY usr) a
    ON num_messages.usr=a.usr AND num_messages.cnt=a.max_cnt
    ORDER BY usr;
    ```

1. Most content in messages to user.

    ```sql
    WITH message_lengths AS (
        SELECT a.usr, a.other, a.cnt + b.cnt AS cnt
        FROM (SELECT sender AS usr, recipient AS other, SUM(char_length(message)) AS cnt
              FROM messages
              GROUP BY sender, recipient) a
        JOIN (SELECT recipient AS usr, sender AS other, SUM(char_length(message)) AS cnt
              FROM messages
              GROUP BY sender, recipient) b
        ON a.usr=b.usr AND a.other=b.other)

    SELECT message_lengths.usr, message_lengths.other, cnt
    FROM message_lengths
    JOIN (SELECT usr, max(cnt) AS max_cnt
          FROM message_lengths
          GROUP BY usr) a
    ON message_lengths.usr=a.usr AND message_lengths.cnt=a.max_cnt
    ORDER BY usr;
    ```

1. Percent of time the above two are different.

    ```sql
    SELECT COUNT(*)
    FROM (SELECT num_messages.usr, num_messages.other, cnt
          FROM num_messages
          JOIN (SELECT usr, max(cnt) AS max_cnt
                FROM num_messages
                GROUP BY usr) a
          ON num_messages.usr=a.usr AND num_messages.cnt=a.max_cnt
          ORDER BY usr) a
    JOIN (SELECT message_lengths.usr, message_lengths.other, cnt
          FROM message_lengths
          JOIN (SELECT usr, max(cnt) AS max_cnt
                FROM message_lengths
                GROUP BY usr) a
          ON message_lengths.usr=a.usr AND message_lengths.cnt=a.max_cnt
          ORDER BY usr) b
    ON a.usr=b.usr AND a.other!=b.other;
    ```

    591 / 1000 = 59%

1. Number of friends and messages received.

    ```sql
    CREATE TABLE friends_and_messages AS
    
    WITH cleaned_friends AS (
        (SELECT userid1, userid2 FROM friends)
        UNION
        (SELECT userid2, userid1 FROM friends))

    SELECT f.userid, f.friends, m.messages
    FROM (SELECT userid1 AS userid, COUNT(*) AS friends
          FROM cleaned_friends
          GROUP BY userid1) f
    JOIN (SELECT recipient AS userid, COUNT(*) AS messages
          FROM messages
          GROUP BY recipient) m
    ON f.userid=m.userid;
    ```

1. Break users into 10 cohorts based on their number of friends and get the average number of messages for each group.

    First get the max number of friends:
    ```sql
    SELECT MAX(friends) FROM friends_and_messages;
    ```
    The max is 38 friends, so let's break it up into these 10 groups:
    0-3, 4-7, 8-11, 12-15, 16-19, 20-23, 24-27, 28-31, 32-35, 36-39

    The cohort is the number of friends divided by 4, rounded down.

    ```sql
    SELECT friends/((SELECT MAX(friends) FROM friends_and_messages) / 10 + 1) AS cohort, AVG(messages)
    FROM friends_and_messages
    GROUP BY 1
    ORDER BY 1;
    ```
