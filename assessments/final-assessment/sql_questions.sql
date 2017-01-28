SELECT DISTINCT
    id, name
FROM students
JOIN applications
ON
    students.id=applications.stud_id AND
    applications.date<'2014-06-01';

--  id |  name
-- ----+---------
--   2 | jonny
--  13 | ajay
--  15 | michael
--   4 | erin
--  14 | adam
--  11 | bruno
--  12 | jana
--   7 | ethan
--   1 | wini
--  10 | stephen
-- (10 rows)



SELECT
    c.id, c.name, COUNT(a.stud_id)
FROM companies c
LEFT OUTER JOIN applications a
ON c.id = a.comp_id
GROUP BY c.id, c.name;

--  id |     name     | count
-- ----+--------------+-------
--   2 | lumiata      |    10
--   1 | tesla        |     3
--   7 | looker       |     5
--   5 | facebook     |     4
--   8 | tagged       |     0
--   9 | keen io      |     3
--  10 | khan academy |     6
--   6 | heroku       |     2
--   4 | radius       |     3
--   3 | stripe       |     4
-- (10 rows)
