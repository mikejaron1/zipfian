 -- Add JSON Serde 
  ADD JAR ${SAMPLE}/libs/jsonserde.jar ;

-- Create the external impressions table
   CREATE EXTERNAL TABLE impressions (
    requestBeginTime string, ad_id string, impression_id string, referrer string, 
    user_agent string, user_cookie string, ip string
  )
  PARTITIONED BY (dt string)
  ROW FORMAT 
    serde 'com.amazon.elasticmapreduce.JsonSerde'
    with serdeproperties ( 'paths'='requestBeginTime, adId, impressionId, referrer, userAgent, userCookie, ip' )
  LOCATION '${SAMPLE}/tables/impressions' ;

-- Recover partitions
  MSCK REPAIR TABLE impressions;

-- Create clicks table
CREATE EXTERNAL TABLE clicks (
    impression_id string
  )
  PARTITIONED BY (dt string)
  ROW FORMAT 
    SERDE 'com.amazon.elasticmapreduce.JsonSerde'
    WITH SERDEPROPERTIES ( 'paths'='impressionId' )
  LOCATION '${SAMPLE}/tables/clicks' ;

  CREATE EXTERNAL TABLE joined_impressions (
    requestBeginTime string, ad_id string, impression_id string, referrer string, 
      user_agent string, user_cookie string, ip string, clicked Boolean
    )
    PARTITIONED BY (day string, hour string)
    STORED AS SEQUENCEFILE
    LOCATION '${OUTPUT}/joined_impressions'
  ;
-- Recover partitions
  MSCK REPAIR TABLE clicks;

-- Create temporary impressions table
  CREATE TABLE tmp_impressions (
  requestBeginTime string, ad_id string, impression_id string, referrer string, 
    user_agent string, user_cookie string, ip string
    )
    STORED AS SEQUENCEFILE
  ;

-- Insert data into our temporary impressions table for our period of interest
  INSERT OVERWRITE TABLE tmp_impressions 
  SELECT 
    from_unixtime(cast((cast(i.requestBeginTime as bigint) / 1000) as int)) requestBeginTime, 
    i.ad_id, i.impression_id, i.referrer, i.user_agent, i.user_cookie, i.ip
  FROM 
    impressions i
  WHERE 
    i.dt >= '${DAY}-${HOUR}-00' and i.dt < '${NEXT_DAY}-${NEXT_HOUR}-00'
;

-- Create temporary clicks table
  INSERT OVERWRITE TABLE tmp_clicks
  SELECT 
    c.dt
  FROM 
    clicks c
  WHERE 
    c.dt >= '${DAY}-${HOUR}-00' AND c.dt < '${NEXT_DAY}-${NEXT_HOUR}-20'
;

-- Join our impressions table with our click table
 INSERT OVERWRITE TABLE joined_impressions PARTITION (day='${DAY}', hour='${HOUR}')
  SELECT 
    i.requestBeginTime, i.ad_id, i.impression_id, i.referrer, i.user_agent, i.user_cookie, 
    i.ip, (c.impression_id is not null) clicked
  FROM 
    tmp_impressions i LEFT OUTER JOIN tmp_clicks c ON i.impression_id = c.impression_id
  ;

-- Create the feature index table
 CREATE TABLE feature_index (feature string, ad_id string, clicked_percent double) STORED AS SEQUENCEFILE;

-- Munge the user agents with a UDF and then combine all the tables
   INSERT OVERWRITE TABLE feature_index
    SELECT
      temp.feature,
      temp.ad_id,
      sum(if(temp.clicked, 1, 0)) / cast(count(1) as DOUBLE) as clicked_percent
    FROM (
      SELECT concat('ua:', trim(lower(ua.feature))) as feature, ua.ad_id, ua.clicked
      FROM (
        MAP joined_impressions.user_agent, joined_impressions.ad_id, joined_impressions.clicked
        USING '${SAMPLE}/libs/split_user_agent.py' as (feature STRING, ad_id STRING, clicked BOOLEAN)
      FROM joined_impressions
    ) ua
    
    UNION ALL
    
    SELECT concat('ip:', regexp_extract(ip, '^([0-9]{1,3}\.[0-9]{1,3}).*', 1)) as feature, ad_id, clicked
    FROM joined_impressions
    
    UNION ALL
    
    SELECT concat('page:', lower(referrer)) as feature, ad_id, clicked
    FROM joined_impressions
  ) temp
  GROUP BY temp.feature, temp.ad_id;

-- Grand Finale

  SELECT 
    ad_id, -sum(log(if(0.0001 > clicked_percent, 0.0001, clicked_percent))) AS value
  FROM 
    feature_index
  WHERE 
    feature = 'ua:safari' OR feature = 'ua:chrome'
  GROUP BY 
    ad_id
  ORDER BY 
    value ASC
  LIMIT 100
  ;