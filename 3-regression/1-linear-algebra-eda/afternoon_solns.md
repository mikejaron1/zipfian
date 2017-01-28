##Part 1: Exploratory Data Analysis (EDA)
 
In this scenario, you are a data scientist at [Bay Area Bike Share](http://www.bayareabikeshare.com/). Your task
is to provide insights on bike user activity and behavior to the products team. 

Import libraries

```python
import pandas as pd
import statsmodels.api as sms 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
%matplotlib inline
```

1. Load the file `data/201402_trip_data.csv` into a dataframe. Provide the argument `parse_dates=['start_date', 'end_date']`
   with `pandas.read_csv()` to read the columns in as datetime objects. 
   
   Make 4 extra columns from the `start_date` column (We will use these in later questions):
   - `month` would contain only the month component
   - `dayofweek` would indicate what day of the week the date is
   - `date` would contain only the date component 
   - `hour` would only contain the hour component
   - [Hint to deal with datetime objects in pandas](http://stackoverflow.com/questions/25129144/pandas-return-hour-from-datetime-column-directly)

   ```python
   def read_trips_and_preprocess(filename):
      # Read in file
      trips = pd.read_csv(filename, parse_dates=['start_date'])
      start_time = trips['start_date']
      
      # Get hour, date, dayofweek
      hr = start_time.apply(lambda x: x.hour) + 1
      date = start_time.apply(lambda x: x.date())    
      dayofweek = start_time.apply(lambda x: x.dayofweek + 1) #Monday=1, Sunday=7
      month = start_time.apply(lambda x: x.month)
      
      trips['hr'] = hr
      trips['date'] = date
      trips['dayofweek'] = dayofweek
      trips['month'] = month
      trips['count'] = 1
      
      return trips
      
   trips = read_trips_and_preprocess('input/201402_trip_data.csv')
   ```

2. Group the bike rides by `month` and count the number of users per month. Plot the number of users for each month. 
   What do you observe? Provide a likely explanation to your observation. Real life data can often be messy/incomplete
   and cursory EDA is often able to reveal that.
   
   ```python
   trips.groupby('month').count()['count'].plot(marker='o')
   ```
   
   ![image](images/count_mth.png)
   
3. Plot the daily user count from September to December. Mark the `mean` and `mean +/- 1.5 * Standard Deviation` as 
   horizontal lines on the plot. This would help you identify the outliers in your data. Describe your observations. 
   
   ```python
   def mth_plot(n):
    mth = trips[trips['month'] == n]
    mth_cnt = mth.groupby('date').count()['count']
    mth_cnt.plot(marker='o', markersize=5, alpha=.5, rot=90)

   fig = plt.figure(figsize=(12, 4))
   mth_plot(9)
   mth_plot(10)
   mth_plot(11)
   mth_plot(12)
   plt.ylabel('Number of Users', fontsize=14)
   plt.xlabel('Month', fontsize=14)
   count_on_each_day = trips.groupby('date').count()['count']
   mean_activity = count_on_each_day.mean()
   upper_std = mean_activity + 1.5 * np.std(count_on_each_day)
   lower_std = mean_activity - 1.5 * np.std(count_on_each_day)
   plt.axhline(upper_std, linestyle='--', c='black', alpha=.4)
   plt.axhline(lower_std, linestyle='--', c='black', alpha=.4)
   plt.axhline(mean_activity, c='black', alpha=.4)
   plt.show()
   ```
   
   ![image](images/timeseries.png)

4. Plot the distribution of the daily user counts for all months as a histogram. Fit a 
   [KDE](http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html) to the histogram.
   What is the distribution and explain why the distribution might be shaped as such. 
    
   ```python
   # Plotting the KDE of weekday and weekend combined
   x = np.linspace(0, 1200, 1000)
   kde_pdf = gaussian_kde(count_on_each_day)
   y = kde_pdf(x)
   plt.plot(x, y, c='r', lw=2)
   count_on_each_day = trips.groupby('date').count()['count']
   plt.hist(count_on_each_day, normed=1, bins=15, edgecolor='none', alpha=.4)
   plt.ylabel('Probability Density', fontsize=14)
   plt.xlabel('Number of Users', fontsize=14)
   plt.show()
   ```
    
   ![image](images/kde.png)
  
   Replot the distribution of daily user counts after binning them into weekday or weekend rides. Refit  
   KDEs onto the weekday and weekend histograms.

   ```python
   cnt = trips.groupby(['date', 'dayofweek']).count()['count'].reset_index()
   weekend = cnt[cnt['dayofweek'] > 5]['count'].values
   weekday = cnt[cnt['dayofweek'] <= 5]['count'].values
   plt.hist(weekend, bins=15, alpha=.1, edgecolor='none', color='g', normed=1)
   plt.hist(weekday, bins=15, alpha=.1, edgecolor='none', color='b', normed=1)
   
   # Plotting the KDE of weekday and weekend
   kde_pdf = gaussian_kde(weekday)
   x = np.linspace(min(weekday), max(weekday), 1000)
   y = kde_pdf(x)
   plt.plot(x, y, color='b', lw=2, label='weekday')
   
   kde_pdf = gaussian_kde(weekend)
   x = np.linspace(min(weekend), max(weekend), 1000)
   y = kde_pdf(x)
   plt.plot(x, y, color='g', lw=2, label='weekend')
   plt.ylabel('Probability Density', fontsize=14)
   plt.xlabel('Number of Users', fontsize=14)
   
   plt.legend(frameon=False)
   plt.show()
   ```
   
   ![image](images/weekdayweekend.png)

5. Now we are going to explore hourly trends of user activity. Group the bike rides by `date` and `hour` and count 
   the number of rides in the given hour on the given date. Make a 
   [boxplot](http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/) of the hours in the day **(x)** against
   the number of users **(y)** in that given hour. 
   
   ```python
   def plot_basic_trends(df):
    start_time = df['start_date']
    hr = start_time.apply(lambda x: x.hour) + 1
    date = start_time.apply(lambda x: x.date())    
    dayofweek = start_time.apply(lambda x: x.dayofweek + 1) #Monday=1, Sunday=7
    df['hr'] = hr
    df['date'] = date
    df['dayofweek'] = dayofweek
    df['count'] = 1
        
    hr_cnt = df.groupby(['date', 'hr']).count()['count'].reset_index()
    gpby = hr_cnt.groupby('hr')
    lst = [gpby.get_group(hr)['count'] for hr in gpby.groups]
    plt.boxplot(lst)
    plt.ylim(0, 200)
    plt.xlabel('Hour of the Day', fontsize=14)
    plt.ylabel('User Freq.', fontsize=14)
    
   plot_basic_trends(trips)
   ```
   
   ![image](images/basic.png)
   
6. Someone from the analytics team made a line plot (_right_) that he claims is showing the same information as your
   boxplot (_left_). What information can you gain from the boxplot that is missing in the line plot?

   ```
   The variation of the user frequency at each hour in the day over the different days.
   ```

7. Replot the boxplot in `6.` after binning your data into weekday and weekend. Describe the differences you observe
   between hour user activity between weekday and weekend? 
    
8. There are two types of bike users (specified by column `Subscription Type`: `Subscriber` and `Customer`. Given this
   information and the weekend and weekday categorization, plot and inspect the user activity trends. Suppose the 
   product team wants to run a promotional campaign, what are you suggestions in terms of who the promotion should 
   apply to and when it should apply for the campaign to be effective?
   
   ```python
   def plot_trends2(df, customer_type):
    df = df[df['subscription_type'] == customer_type] #Customer

    wkday_df = df[df['dayofweek'] <= 5]
    wkend_df = df[df['dayofweek'] > 5]
        
    wkday_date_hr_cnt = wkday_df.groupby(['date', 'hr']).count()['count'].reset_index()
    wkend_date_hr_cnt = wkend_df.groupby(['date', 'hr']).count()['count'].reset_index()
    
    wkday_gpby = wkday_date_hr_cnt.groupby('hr')
    wkend_gpby = wkend_date_hr_cnt.groupby('hr')
    
    wkday_lst = [wkday_gpby.get_group(hr)['count'] for hr in wkday_gpby.groups]
    wkend_lst = [wkend_gpby.get_group(hr)['count'] for hr in wkend_gpby.groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.boxplot(wkday_lst)
    ax2.boxplot(wkend_lst)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel('Hour of the Day', fontsize=14)
    ax1.set_ylabel('User Freq.', fontsize=14)
    ax1.set_title('Weekday')
    ax2.set_ylim(0, 200)
    ax2.set_xlabel('Hour of the Day', fontsize=14)
    ax2.set_ylabel('User Freq.', fontsize=14)
    ax2.set_title('Weekend')
    plt.suptitle(customer_type, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
   plot_trends2(trips, 'Subscriber')
   plot_trends2(trips, 'Customer')
   ```
   
   - I would suggest the promotional campaign to be held during week days during `8am - 10am` and `5pm - 7pm`, 
     and on weekends during `12pm - 5pm`. The former period has more subscribers and the second has more customers.
   
   ![image](images/nonsub.png)
   ![image](images/customer.png)
   
9. **Extra Credit:** You are also interested in identifying stations with low usage. Load the csv file 
   `data/201402_station_data.csv` into a dataframe. The `docksize` column specifies how many bikes the station can hold. 
   The `lat` and `long` columns specify the latitude and longitude of the station. 
   
   - Merge the station data with the trip data
   - Compute usage by counting the total users starting at a particular station divided by the dockcount
   - Normalize usage to range from `0`to `1`
   - Using plotly, plot the latitude and longitude of the stations as scatter points, the usage will be indicated 
     by the transperancy and the size of the points
     
   - Merge the `trips` and `stations` data
   
   ```python
   def merge_station_trips(station_fname, trips_fname):
    stations = pd.read_csv(station_fname)
    trips = pd.read_csv(trips_fname, parse_dates=['start_date'])
    trips['count'] = 1
    # Merge the trips data with station by station name
    start_merge = pd.merge(trips, stations, left_on='start_station', right_on='name')
    # Adjust usage by the dockcount
    start_merge['count'] = (start_merge['count'] * 1.) / start_merge.dockcount
    # Compute the usage of each station
    start_merge = start_merge.groupby(['start_station', 'lat', 'long']).sum()[['count']].reset_index()
    # Normalize usage of each stations to range from 0-1
    normalize_cnt = start_merge['count'] / (start_merge['count'].max())
    
    return start_merge, normalize_cnt

   start_merge, normalize_cnt = merge_station_trips('data/201402_station_data.csv', 'data/201402_trip_data.csv')
   ```
   
   - Plot the usage using plotly
   
   ```python
   import plotly.plotly as py
   from plotly.graph_objs import *

   txt = ['%s: %.1f' % (station, cnt * 100) for station, cnt in zip(start_merge['start_station'], normalize_cnt)]
   data = Data([Scatter(x=start_merge['long'],
                        y=start_merge['lat'],
                        mode='markers',
                        marker=Marker(opacity=normalize_cnt, size=normalize_cnt * 100),
                        text=txt)])
   py.iplot(data)
   ```
  
   ![scatter](images/plotly.png)


##Part 2: Intro to Linear Regression

Linear regression is an approach to modeling the relationship between a continuous dependent (**y**) variable and 
one or more continuous independent (**x**) variables. Here you will be introduced to fitting the model and interpreting
the results before we dive more into the details of linear regression tomorrow.

1. We will be using the `prestige` data in `statsmodels`. `statsmodels` is the de facto library for performing regression
   tasks in Python. Load the data with the follow code. **Remember to add a column of `1` to the x matrix for the 
   model to fit an intercept**.

   ```python 
   import statsmodels.api as sm
   prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
   y = prestige['prestige']
   x = prestige[['income', 'education']].astype(float)
   ```

2. Explore the data by making a [scatter_matrix](http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html#visualization-scatter-matrix)
   and a [boxplot](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html)
   to show the range of each of your variables.
   
3. The beta coefficients of a linear regression model can be calculated by solving the normal equation.
   Using numpy, write a function that solves the **normal equation** (below).
   As input your function should take a matrix of features (**x**) and
   a vector of target (**y**). You should return a vector of beta coefficients 
   that represent the line of best fit which minimizes the residual. 
   Calculate  R<sup>2</sup>. 
   
   <div align="center">
      <img height="30" src="images/normal_equation.png">
   </div>

3. Verify your results using statsmodels. Use the code below as a reference.
   ```python
   import statsmodels.api as sms
   model = sms.OLS(y, x).fit()
   summary = model.summary()
   ```

4. Interpret your result summary, focusing on the beta coefficents, p-values, F-statistic, and the R<sup>2</sup>. 
