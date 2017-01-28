1. Your company is testing out a new version of the site. They would like to see if the new version results in more friend requests sent. They decided to A/B test it by randomly giving 5% of hits the new site. The results are in these SQL tables. The experiment was run from 2014-08-21 to 2014-08-28 so you should only include friend request data over that time frame.

    ```
    landing_page_test
        userid
        group

    logins
        userid
        date

    friend_requests
        userid
        recipient
        date
    ```

    The `landing_page_test` has a group for every `userid`, either `new_page` or `old_page`.

    Write a SQL query (or queries) to get the data to fill in the following table.

    |    group | number of logins | number of friend requests |
    | -------- | ---------------- | ------------------------- |
    | new page |                  |                           |
    | old page |                  |                           |

    ```sql
    SELECT "group", login_cnt, requests_cnt
    FROM (
        SELECT "group", COUNT(1) AS login_cnt
        FROM landing_page_test
        JOIN logins
        ON landing_page_test.userid=logins.userid
        GROUP BY "group") l
    JOIN (
        SELECT "group", COUNT(1) AS requests_cnt
        FROM landing_page_test
        JOIN friend_requests
        ON friend_requests.userid=logins.userid
        GROUP BY "group") r
    ON l."group"=r."group";
    ```


2. Now that you've collected the data, let's say these are the results you pulled from the SQL tables:

    |    group | number of logins | number of friend requests |
    | -------- | ---------------- | ------------------------- |
    | new page |            51982 |                       680 |
    | old page |          1039410 |                     12801 |


    Are you confident that the new landing page is better? Show your work with both a frequentist and a Bayesian approach.

    If not, how would you recommend your team to proceed?


    **Frequentist**
    ```python
    from scipy.stats import chi2_contingency

    chi2, p, dof, expected = \
        chi2_contingency([[51982 - 680, 680],
                          [1039410 - 12801, 12801]])
    print "p value is:", p
    ```
    You get a p-value of 0.128, which is not statistically significant.


    **Bayesian**
    ```python
    from numpy.random import beta as beta_dist

    num_samples = 10000
    new_samples = beta_dist(1 + 51982, 1 + 51982 - 680, num_samples)
    old_samples = beta_dist(1 + 1039410, 1 + 1039410 - 12801, num_samples)

    print "percent of time new is better than old:", \
        np.mean(new_samples > old_samples)
    ```
    You get a value of around 0.55, which is not large enough to be confident.
