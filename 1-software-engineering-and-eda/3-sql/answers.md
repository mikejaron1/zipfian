Basic Exploration
=================

1. What are the tables in our database? Run `\d` to find out.

    ```sql
    readychef=# \d
               List of relations
     Schema |   Name    | Type  |  Owner
    --------+-----------+-------+----------
     public | events    | table | postgres
     public | meals     | table | postgres
     public | referrals | table | postgres
     public | users     | table | postgres
     public | visits    | table | postgres
    (5 rows)
    ```

2. What columns does each table have? Run `\d tablename` to find out.

    ```sql
    readychef=# \d users
                Table "public.users"
       Column    |       Type        | Modifiers
    -------------+-------------------+-----------
     userid      | integer           |
     dt          | date              |
     campaign_id | character varying |

    readychef=# \d meals
              Table "public.meals"
     Column  |       Type        | Modifiers
    ---------+-------------------+-----------
     meal_id | integer           |
     type    | character varying |
     dt      | date              |
     price   | integer           |

    readychef=# \d referrals
         Table "public.referrals"
       Column    |  Type   | Modifiers
    -------------+---------+-----------
     referred    | integer |
     referred_by | integer |

    readychef=# \d visits
        Table "public.visits"
     Column |  Type   | Modifiers
    --------+---------+-----------
     dt     | date    |
     userid | integer |

    readychef=# \d events
              Table "public.events"
     Column  |       Type        | Modifiers
    ---------+-------------------+-----------
     dt      | date              |
     userid  | integer           |
     meal_id | integer           |
     event   | character varying |
    ```

Select Statements
===============================
1. Run a SELECT statement on each table. Limit the number of rows to 10.

    ```sql
    SELECT * FROM users LIMIT 10;
    SELECT * FROM meals LIMIT 10;
    SELECT * FROM referrals LIMIT 10;
    SELECT * FROM visits LIMIT 10;
    SELECT * FROM events LIMIT 10;
    ```

2. Write a `SELECT` statement that would get just the userids.

    ```sql
    SELECT userid FROM users;
    ```

3. Maybe you're just interested in what the campaign ids are. Use 'SELECT DISTINCT' to figure out all the possible values of that column.

    ```sql
    SELECT DISTINCT campaign_id FROM users;
    ```

Where Clauses/Filtering:
===============================
1. Using the WHERE clause, write a new SELECT statement that returns all rows where Campaign_ID is equal to FB.

    ```sql
    SELECT * FROM users WHERE campaign_id='FB';
    ```
    
2. We don't need the campaign id in the result since they are all the same, so only include the other two columns.

    ```sql
    SELECT userid, dt FROM users WHERE campaign_id='FB';
    ```

Aggregation Functions
===============================
1. Write a query to get the count of just the users who came from Facebook.

    *Note that I can interchangable use `COUNT(*)`, `COUNT(1)`, `COUNT(campaign_id)`. The last of these three would get a different result if I was using `COUNT (DISTINCT ...)`.*

    ```sql
    SELECT COUNT(1) FROM users WHERE campaign_id='FB';
    ```

2. Now, count the number of users coming from each service.

    ```sql
    SELECT campaign_id, COUNT(*)
    FROM users
    GROUP BY campaign_id;
    ```

3. Use `COUNT (DISTINCT columnname)` to get the number of unique dates that appear in the `users` table.

    ```sql
    SELECT COUNT(DISTINCT dt) FROM users;
    ```

4. There's also `MAX` and `MIN` functions, which do what you might expect. Write a query to get the first and last registration date from the `users` table.

    ```sql
    SELECT MIN(dt), MAX(dt) FROM users;
    ```

5. Calculate the mean price for a meal.

    ```sql
    SELECT AVG(price) FROM meals;
    ```
    
6. Now get the average price, the min price and the max price for each meal type.

    ```sql
    SELECT type, AVG(price), MIN(price), MAX(price)
    FROM meals
    GROUP BY type;
    ```

7.  Alias all the above columns.

    ```sql
    SELECT
        type,
        AVG(price) AS avg_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM meals
    GROUP BY type;
    ```

8. Add a `WHERE` clause to the above query to consider only meals in the first quarter of 2013 (month<=3 and year=2013).

    ```sql
    SELECT
        type,
        AVG(price) AS avg_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM meals
    WHERE
        date_part('year', dt)=2013 AND
        date_part('month', dt)<=3
    GROUP BY type;
    ```

9. Modify the above query so that we get the aggregate values for each month and type.

    ```sql
    SELECT
        type,
        date_part('month', dt) AS month,
        AVG(price) AS avg_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM meals
    WHERE
        date_part('year', dt)=2013 AND
        date_part('month', dt)<=3
    GROUP BY type, month;
    ```

10. From the `events` table, write a query that gets the total number of buys, likes and shares for each meal id.

    ```sql
    SELECT
        meal_id,
        SUM(CASE WHEN event='bought' THEN 1 ELSE 0 END) AS bought,
        SUM(CASE WHEN event='like' THEN 1 ELSE 0 END) AS liked,
        SUM(CASE WHEN event='share' THEN 1 ELSE 0 END) AS shared
    FROM events
    GROUP BY meal_id;
    ```

Sorting
===============================
1. Let's start with a query which gets the average price for each type.

    ```sql
    SELECT type, AVG(price) AS avg_price
    FROM meals
    GROUP BY type;
    ```

2. To make it easier to read, sort the results by the `type` column.

    ```sql
    SELECT type, AVG(price) AS avg_price
    FROM meals
    GROUP BY type
    ORDER BY type;
    ```

3. Now return the same table again, except this time order by the price in descending order.

    ```sql
    SELECT type, AVG(price) AS avg_price
    FROM meals
    GROUP BY type
    ORDER BY avg_price DESC;
    ```

Joins
===============================
1. Write a query to get one table that joins the `events` table with the `users` table (on `userid`)

    ```sql
    SELECT
        users.userid,
        users.campaign_id,
        events.meal_id,
        events.event
    FROM users
    JOIN events
    ON users.userid=events.userid;
    ```

2. Also include information about the meal, like the `type` and the `price`. Only include the `bought` events.

    ```sql
    SELECT
        users.userid,
        users.campaign_id,
        events.meal_id,
        meals.type,
        meals.price
    FROM users
    JOIN events
    ON
        users.userid=events.userid AND
        events.event='bought'
    JOIN meals
    ON meals.meal_id=events.meal_id;
    ```

3. Write a query to get how many of each meal were bought.

    ```sql
    SELECT type, COUNT(1)
    FROM meals
    JOIN events
    ON
        meals.meal_id=events.meal_id AND
        events.event='bought'
    GROUP BY type;
    ```

Subqueries
===============================

1. Write a query to get meals that are above the average meal price.

    ```sql
    SELECT *
    FROM meals
    WHERE price > (SELECT AVG(price) FROM MEALS);
    ```

2. Write a query to get the meals that are above the average meal price *for that type*.

    ```sql
    SELECT meals.*
    FROM meals
    JOIN (
        SELECT type, AVG(price) AS price
        FROM meals
        GROUP BY type) average
    ON
        meals.type=average.type AND
        meals.price>average.price;
    ```

3. Modify the above query to give a count of the number of meals per type that are above the average price.

    ```sql
    SELECT meals.type, COUNT(1)
    FROM meals
    JOIN (
        SELECT type, AVG(price) AS price
        FROM meals
        GROUP BY type) average
    ON
        meals.type=average.type AND
        meals.price>average.price
    GROUP BY meals.type;
    ```

4. Calculate the percentage of users which come from each service.

    ```sql
    SELECT
        campaign_id,
        CAST(COUNT(1) AS REAL) / (SELECT COUNT(1) FROM users) AS percent
    FROM users
    GROUP BY campaign_id;
    ```

Extra Credit
==============
1. Answer the question, _"What user from each campaign bought the most items?"_

    ```sql
    WITH user_campaign_counts AS (
        SELECT campaign_id, users.userid, COUNT(1) AS cnt
        FROM users
        JOIN events
        ON users.userid=events.userid
        GROUP BY campaign_id, users.userid)

    SELECT u.campaign_id, u.userid, u.cnt
    FROM user_campaign_counts u
    JOIN (SELECT campaign_id, MAX(cnt) AS cnt
          FROM user_campaign_counts
          GROUP BY campaign_id) m
    ON u.campaign_id=m.campaign_id AND u.cnt=m.cnt
    ORDER BY u.campaign_id, u.userid;
    ```

2. For each day, get the total number of users who have registered as of that day. You should get a table that has a `dt` and a `cnt` column.

    ```sql
    SELECT a.dt, COUNT(1)
    FROM (SELECT DISTINCT dt FROM users) a
    JOIN users b
    ON b.dt <= a.dt
    GROUP BY a.dt
    ORDER BY a.dt;
    ```

3. What day of the week gets meals with the most buys?

    ```sql
    SELECT date_part('dow', meals.dt) AS dow, COUNT(1) AS cnt
    FROM meals
    JOIN events
    ON
        meals.meal_id=events.meal_id AND
        event='bought'
    GROUP BY 1
    ORDER BY cnt DESC
    LIMIT 1;
    ```

4. Which month had the highest percent of users who visited the site purchase a meal?

    ```sql
    SELECT v.month, v.year, CAST(v.cnt AS REAL) / e.cnt AS avg
    FROM (
        SELECT
            date_part('month', dt) AS month,
            date_part('year', dt) AS year,
            COUNT(1) AS cnt
        FROM visits
        GROUP BY 1, 2) v
    JOIN (
        SELECT
            date_part('month', dt) AS month,
            date_part('year', dt) AS year,
            COUNT(1) AS cnt
        FROM events
        WHERE event='bought'
        GROUP BY 1, 2) e
    ON v.month=e.month AND v.year=e.year
    ORDER BY avg DESC
    LIMIT 1;
    ```

5. Find all the meals that are above the average price of the previous 7 days.

    ```sql
    SELECT a.meal_id
    FROM meals a
    JOIN meals b
    ON b.dt <= a.dt AND b.dt > a.dt - 7
    GROUP BY a.meal_id, a.price
    HAVING a.price > AVG(b.price);
    ```

6. What percent of users have shared more meals than they have liked?

    ```sql
    SELECT CAST(COUNT(1) AS REAL) / (SELECT COUNT(1) FROM users)
    FROM (
        SELECT userid
        FROM events
        GROUP BY userid
        HAVING
            SUM(CASE WHEN event='share' THEN 1 ELSE 0 END) >
            SUM(CASE WHEN event='like' THEN 1 ELSE 0 END)) t;
    ```

7. For every day, count the number of users who have visited the site and done no action.

    ```sql
    SELECT visits.dt, COUNT(1)
    FROM visits
    LEFT OUTER JOIN events
    ON visits.userid=events.userid AND visits.dt=events.dt
    WHERE events.userid IS NULL
    GROUP BY visits.dt;
    ```

8. Find all the days with a greater than average number of meals.

    ```sql
    SELECT dt
    FROM meals
    GROUP BY dt
    HAVING
        COUNT(1) >
        (SELECT COUNT(1) FROM meals) / (SELECT COUNT(DISTINCT dt) FROM meals);
    ```

9. Find all the users who bought a meal before liking or sharing a meal.

    This query will take a long time to run. I recommend instead create mini tables to run the query on like this:

    ```sql
    CREATE TABLE miniusers AS SELECT * FROM users WHERE userid<=10;
    CREATE TABLE minievents AS SELECT * FROM events WHERE userid<=10;
    ```

    ```sql
    SELECT u.userid
    FROM users u
    JOIN events a
    ON
        u.userid=a.userid AND
        a.event='like' OR a.event='share'
    JOIN events b
    ON
        u.userid=b.userid AND
        b.event='bought'
    GROUP BY u.userid
    HAVING MIN(b.dt) < MIN(a.dt);
    ```
