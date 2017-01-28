/* Define Cohort by Month Signed Up */

CREATE VIEW month_cohort
  AS
    SELECT
      id, DATE_TRUNC('month', MIN("Created_Date")) as cohort
    FROM "Users"
GROUP BY id;

/* Define Retention by Month */

CREATE VIEW months_visited
  AS
    SELECT
      "user_id" AS id, date_trunc('month', "date") AS month_active
    FROM "visit"
    GROUP BY 1,2
    ORDER BY 1;

/* Combine the Results */

CREATE VIEW month_cohort_retention
  AS
    SELECT 
      cohort, month_active, 
      rank() OVER (PARTITION BY cohort ORDER BY month_active ASC) AS month_rank,
      COUNT(DISTINCT(m.id)) AS uniques,
      COUNT(DISTINCT(m.id)) / (first_value(COUNT(DISTINCT(m.id))) OVER (PARTITION BY cohort))::REAL AS fraction_retained
    FROM month_cohort c
    JOIN months_visited m
    ON c.id = m.id
    GROUP BY 1,2 
    ORDER BY 1,2;
