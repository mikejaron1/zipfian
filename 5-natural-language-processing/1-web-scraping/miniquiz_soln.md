Problem Statement: It's common at Hitch to want to know various business metrics about recent trips. Given the above subset of Hitch's schema, write executable SQL queries to answer the following questions:

1. Between 12/1/2013 10:00:00 PST & 12/8/2013 17:00:00 PST, how many completed trips (Hint: look at the trips.status column) were requested on iphones in City #5? on android phones?

    ```sql
    SELECT request_device, COUNT(1)
    FROM trips
    WHERE
        status='completed' AND
        request_device IN ('android', 'iphone') AND
        trips.request_at > '12/01/2013 10:00:00 PST' AND
        trips.request_at < '12/8/2013 17:00:00 PST' AND
        trips.city_id = 5
    GROUP BY request_device;
    ```


2. In City #8, how many unique, currently unbanned clients completed a trip in October 2013? Of these, how many trips did each client take?

    ```sql
    SELECT COUNT(DISTINCT users.usersid)
    FROM trips
    JOIN users
    ON
        trips.client_id=users.usersid AND
        trips.status='completed' AND
        NOT users.banned AND
        date_part('month', trips.request_at)=10 AND
        date_part('year', trips.request_at)=2013 AND
        trips.city_id = 8;
    ```

    ```sql
    SELECT users.usersid, COUNT(1)
    FROM trips
    JOIN users
    ON
        trips.client_id=users.usersid AND
        trips.status='completed' AND
        NOT users.banned AND
        date_part('month', trips.request_at)=10 AND
        date_part('year', trips.request_at)=2013 AND
        trips.city_id = 8
    GROUP BY users.usersid;
    ```


3. In City #8, how many unique, currently unbanned clients completed a trip between 9/10/2013 and 9/20/2013 with drivers who started between 9/1/2013 and 9/10/2013 and are currently banned?

    ```sql
    SELECT COUNT(DISTINCT clients.usersid)
    FROM trips
    JOIN users clients
    ON
        trips.client_id=clients.usersid AND
        trips.status='completed' AND
        clients.banned=False AND
        trips.request_at >= '2013-09-13' AND
        trips.request_at <= '2013-09-20' AND
        trips.city_id = 8
    JOIN users drivers
    ON
        trips.driver_id=drivers.usersid AND
        drivers.creationtime >= '2013-09-01' AND
        drivers.creationtime <= '2013-09-10' AND
        drivers.banned;
    ```


4. Extra Credit: Add to your statement in 2) to exclude Hitch admins. Hitch admins have an email address from @hitch.com (example: ‘jsmith@hitch.com’).

    ```sql
    SELECT users.usersid, COUNT(1)
    FROM trips
    JOIN users
    ON
        trips.client_id=users.usersid AND
        trips.status='completed' AND
        NOT users.banned AND
        date_part('month', trips.request_at)=10 AND
        date_part('year', trips.request_at)=2013 AND
        users.email LIKE '%@hitch.com'
    GROUP BY users.usersid;
    ```
