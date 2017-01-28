1. Write a query to build a table we can export to build a model for predicting churn.

    ```sql
    CREATE TABLE churn AS (
        SELECT a.id AS advertiser_id, name, city, state, business_type,
               CASE WHEN advertiser_id IS NULL THEN false ELSE true END AS churn
        FROM advertisers a
        LEFT OUTER JOIN campaigns c
        ON
            a.id=c.advertiser_id AND
            start_data + duration * interval '1 day' >= now() - interval '14 days'
    );
    ```


2. Write a query to calculate these metrics: accuracy, precision, recall (sensitivity), specificity.

    ```sql
    WITH confusion_matrix AS (
        SELECT
            SUM(CASE WHEN t.churn AND p.churn THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN NOT t.churn AND p.churn THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN t.churn AND NOT p.churn THEN 1 ELSE 0 END) AS fn,
            SUM(CASE WHEN NOT t.churn AND NOT p.churn THEN 1 ELSE 0 END) AS tn
        FROM churn t
        JOIN predicted_churn p
        ON t.advertiser_id=p.advertiser_id
    )
    SELECT
        1.0 * (tp + tn) / (tp + fp + fn + tn) AS accuracy,
        1.0 * tp / (tp + fp) AS precision,
        1.0 * tp / (tp + fn) AS recall,
        1.0 * tn / (fp + tn) AS specificity
    FROM confusion_matrix;
    ```
