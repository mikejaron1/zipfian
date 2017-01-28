### Creating Tables

1. Adding the serde:

    ```
    hive> ADD JAR ${SAMPLE}/libs/jsonserde.jar;
    converting to local s3://elasticmapreduce/samples/hive-ads/libs/jsonserde.jar
    Added /mnt/var/lib/hive/downloaded_resources/jsonserde.jar to class path
    Added resource: /mnt/var/lib/hive/downloaded_resources/jsonserde.jar
    ```

2. Create the table:

    ```
    hive> CREATE EXTERNAL TABLE impressions (
    >         requestBeginTime STRING,
    >         ad_id STRING,
    >         impression_id STRING,
    >         referrer STRING,
    >         user_agent STRING,
    >         user_cookie STRING,
    >         ip STRING
    >     )
    >     PARTITIONED BY (dt STRING)
    >     ROW FORMAT
    >     SERDE 'com.amazon.elasticmapreduce.JsonSerde'
    >     WITH SERDEPROPERTIES (
    >         'paths'='requestBeginTime,
    >         adId,
    >         impressionId,
    >         referrer,
    >         userAgent,
    >         userCookie,
    >         ip')
    >     LOCATION '${SAMPLE}/tables/impressions';
    OK
    Time taken: 0.725 seconds
    ```

3. Describe the table:

    ```
    hive> describe impressions;
    OK
    requestbegintime        string                  from deserializer
    ad_id                   string                  from deserializer
    impression_id           string                  from deserializer
    referrer                string                  from deserializer
    user_agent              string                  from deserializer
    user_cookie             string                  from deserializer
    ip                      string                  from deserializer
    dt                      string

    # Partition Information
    # col_name              data_type               comment

    dt                      string
    Time taken: 0.107 seconds, Fetched: 13 row(s)
    ```

4. Selecting:

    ```
    hive> SELECT * FROM impressions LIMIT 10;
    OK
    Time taken: 0.177 seconds
    ```

5. Add a partition:

    ```
    hive> ALTER TABLE impressions ADD PARTITION (dt='2009-04-13-08-05');
    OK
    Time taken: 0.701 seconds
    ```

    Retry selecting:

    ```
    hive> SELECT * FROM impressions LIMIT 10;
    OK
    1239610346000   m9nwdo67Nx6q2kI25qt5On7peICfUM  omkxkaRpNhGPDucAiBErSh1cs0MThC  cartoonnetwork.com  Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; FunWebProducts; GTB6; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET   wcVWWTascoPbGt6bdqDbuWTPPHgOPs  69.191.224.234  2009-04-13-08-05
    ...
    ```

6. Add all partitions:

    ```
    hive> MSCK REPAIR TABLE impressions;
    OK
    Partitions not in metastore:    impressions:dt=2009-04-12-13-00 impressions:dt=2009-04-12-13-05 impressions:dt=2009-04-12-13-10 impressions:dt=2009-04-12-13-15 impressions:dt=2009-04-12-13-20
    ...
    ```

7. Count the number of distinct times.

    ```sql
    SELECT COUNT(DISTINCT dt) FROM impressions;
    ```

    Result: 241

8. Create clicks table.

    ```
    hive>     CREATE EXTERNAL TABLE clicks (
        >         impression_id STRING
        >     )
        >     PARTITIONED BY (dt STRING)
        >     ROW FORMAT SERDE 'com.amazon.elasticmapreduce.JsonSerde'
        >     WITH SERDEPROPERTIES ('paths'='impressionId')
        >     LOCATION '${SAMPLE}/tables/clicks';
    OK
    Time taken: 0.314 seconds
    hive> MSCK REPAIR TABLE clicks;
    OK
    ...
    ```

9. Count the number of distinct times in clicks table.

    ```sql
    SELECT COUNT(DISTINCT dt) FROM clicks;
    ```

    Result: 241


### Combining the Clicks and Impressions Tables

1. Create `joined_impressions` table:

    ```sql
    CREATE EXTERNAL TABLE joined_impressions (
        requestBeginTime STRING,
        ad_id STRING,
        impression_id STRING,
        referrer STRING, 
        user_agent STRING,
        user_cookie STRING,
        ip STRING,
        clicked BOOLEAN
    )
    PARTITIONED BY (day STRING, hour STRING)
    STORED AS SEQUENCEFILE
    LOCATION '${OUTPUT}/joined_impressions';
    ```

2. Create temporary impressions table:

    ```sql
    CREATE TABLE tmp_impressions (
        requestBeginTime STRING,
        ad_id STRING,
        impression_id STRING,
        referrer STRING, 
        user_agent STRING,
        user_cookie STRING,
        ip STRING
    )
    PARTITIONED BY (day STRING, hour STRING)
    STORED AS SEQUENCEFILE;
    ```

3. Insert into `tmp_impressions` table:

    ```sql
    INSERT OVERWRITE TABLE tmp_impressions
        SELECT
            from_unixtime(CAST((CAST(i.requestBeginTime AS BIGINT) / 1000)
                               AS INT)) AS requestBeginTime, 
            i.ad_id,
            i.impression_id,
            i.referrer,
            i.user_agent,
            i.user_cookie,
            i.ip
        FROM impressions i
        WHERE 
            i.dt >= '${DAY}-${HOUR}-00' AND
            i.dt < '${NEXT_DAY}-${NEXT_HOUR}-00';
    ```
4. Create temporary clicks table:

    ```sql
    CREATE TABLE tmp_clicks (
        impression_id STRING
    )
    STORED AS SEQUENCEFILE;
    ```

5. Insert into `tmp_clicks` table:

    ```sql
    INSERT OVERWRITE TABLE tmp_clicks
        SELECT
            c.impression_id
        FROM clicks c
        WHERE
            c.dt >= '${DAY}-${HOUR}-00' AND
            c.dt < '${NEXT_DAY}-${NEXT_HOUR}-20';
    ```


ssh hadoop@ec2-54-191-148-211.us-west-2.compute.amazonaws.com -i ~/blah.pem

6. Fill in the `joined_impressions` table:

    ```sql
    INSERT OVERWRITE TABLE joined_impressions
        PARTITION (day='${DAY}', hour='${HOUR}')
        SELECT
            i.requestBeginTime,
            i.ad_id,
            i.impression_id,
            i.referrer,
            i.user_agent,
            i.user_cookie,
            i.ip,
            IF(c.impression_id IS NULL, false, true) AS clicked
        FROM tmp_impressions i
        LEFT OUTER JOIN tmp_clicks c
        ON i.impression_id=c.impression_id;
    ```

### URL
1. get feature from referral

    ```sql
    SELECT
        concat('page:', lower(referrer)) AS feature,
        ad_id,
        clicked
    FROM joined_impressions
    LIMIT 10;
    ```

### Combining the Features
1. Create `tmp_features` table

    ```sql
    CREATE TABLE tmp_features AS
    SELECT
        concat('ua:', TRIM(LOWER(temp.feature))) AS feature,
        temp.ad_id,
        temp.clicked
        FROM (
            MAP
                joined_impressions.user_agent,
                joined_impressions.ad_id,
                joined_impressions.clicked
            USING '${SAMPLE}/libs/split_user_agent.py'
                AS feature STRING, ad_id STRING, clicked BOOLEAN
            FROM joined_impressions
        ) temp
    UNION ALL
    SELECT 
        concat('ip:', regexp_extract(ip, '^([0-9]{1,3}\.[0-9]{1,3}).*', 1))
            AS feature,
        ad_id,
        clicked
    FROM 
        joined_impressions
    UNION ALL
    SELECT
        concat('page:', lower(referrer)) AS feature,
        ad_id,
        clicked
    FROM joined_impressions;
    ```

2. Create `feature_index` table.

    ```sql
    CREATE EXTERNAL TABLE feature_index (
        feature STRING,
        ad_id STRING,
        clicked_percent DOUBLE
    )
    PARTITIONED BY (day STRING, hour STRING)
    STORED AS SEQUENCEFILE
    LOCATION '${OUTPUT}/feature_index';
    ```

3. Populate feature_index table

    ```sql
    INSERT OVERWRITE TABLE feature_index
        PARTITION (day='${DAY}', hour='${HOUR}')
        SELECT
            f.feature,
            f.ad_id,
            CAST(SUM(IF(f.clicked='true', 1, 0)) AS DOUBLE) / COUNT(*)
                AS clicked_percent
        FROM tmp_features AS f
        GROUP BY f.feature, f.ad_id;
    ```


### Applying the Heuristic
1. Try the features 'us:safari' and 'ua:chrome'.

    ```sql
    SELECT
        ad_id,
        -1 * SUM(LOG(IF(0.0001 > clicked_percent, 0.0001, clicked_percent)))
            AS value
    FROM
        feature_index
    WHERE
        day='${DAY}' AND
        hour='${HOUR}' AND
        (feature='ua:safari' OR feature='ua:chrome')
    GROUP BY ad_id
    ORDER BY value ASC
    LIMIT 100;
    ```
