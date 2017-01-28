```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm
```

1. Load the data

    ```python
    data = pd.read_json('data/logins.json', typ='series')
    data.head()
    ```

    ```
    0   2012-03-01 00:05:55
    1   2012-03-01 00:06:23
    2   2012-03-01 00:06:52
    3   2012-03-01 00:11:23
    4   2012-03-01 00:12:47
    dtype: datetime64[ns]
    ```

2. Resample hourly

    ```python
    ts = pd.TimeSeries(1, data).resample(rule='H', how='count')
    ts.head()
    ```

    2012-03-01 00:00:00    31
    2012-03-01 01:00:00    18
    2012-03-01 02:00:00    37
    2012-03-01 03:00:00    23
    2012-03-01 04:00:00    14
    Freq: H, dtype: int64




# 3. Plot the data

ts.plot()




![png](hitch_timeseries_files/hitch_timeseries_3_1.png)



    # 4. Add a best fit line to the graph
    
    X = ts.index.values.astype(int).reshape(len(ts), 1)
    y = ts.values
    lr = LinearRegression().fit(X, y)
    print "R^2:", lr.score(X, y)
    
    ax = ts.plot(lw=0.5)
    line = pd.Series(lr.predict(X), index=ts.index)
    line.plot(style='r', ax=ax)

    R^2: 0.026291605947





    <matplotlib.axes._subplots.AxesSubplot at 0x10a732790>




![png](hitch_timeseries_files/hitch_timeseries_4_2.png)



    # 5. Exponentially-weighted moving average
    
    ax = ts.plot(lw=0.5)
    pd.ewma(ts, halflife=12).plot(style='r', ax=ax)




    <matplotlib.axes._subplots.AxesSubplot at 0x10ae0d090>




![png](hitch_timeseries_files/hitch_timeseries_5_1.png)



    # 6. Fill in weekends
    
    ax = ts.plot(lw=0.5)
    moving_average = pd.ewma(ts, halflife=12)
    moving_average.plot(style='r', ax=ax)
    
    weekends = ts.index.map(lambda x: x.weekday() in [5, 6])
    ax.fill_between(ts.index, moving_average, where=weekends, color='r')




    <matplotlib.collections.PolyCollection at 0x10b472410>




![png](hitch_timeseries_files/hitch_timeseries_6_1.png)



    # 7. Create a dataframe that has hour and weekday features as well as the count feature
    
    df = pd.DataFrame(ts).rename(columns={0: 'count'})
    df['hour'] = df.index.map(lambda timestamp: timestamp.hour)
    df['weekday'] = df.index.map(lambda timestamp: timestamp.weekday())
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>hour</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-03-01 00:00:00</th>
      <td> 31</td>
      <td> 0</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>2012-03-01 01:00:00</th>
      <td> 18</td>
      <td> 1</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>2012-03-01 02:00:00</th>
      <td> 37</td>
      <td> 2</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>2012-03-01 03:00:00</th>
      <td> 23</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>2012-03-01 04:00:00</th>
      <td> 14</td>
      <td> 4</td>
      <td> 3</td>
    </tr>
  </tbody>
</table>
</div>




    # 8. Use OneHotEncoder to encode the discrete hour and weekday values
    
    encoder = OneHotEncoder()
    X = encoder.fit_transform(df[['hour', 'weekday']])


    # 9. You want to predict the `count` column, so make this the `y` variable
    
    y = df['count'].values


    # 10. Split the dataset into two halves based on time. The first half
    # will be the training set and the second half the test set.
    
    split_index = int(X.shape[0] * 0.8)
    X_train = X[:split_index, :]
    y_train = y[:split_index]
    X_test = X[split_index:, :]
    y_test = y[split_index:]


    # 11. Run a grid_search on the resulting matrices using Support Vector Regression
    
    params = {
        'C': [1, 10, 100],
        'kernel': ['rbf', 'sigmoid'],
        'gamma': [0.0, 0.1],
    }
    
    optimizer = GridSearchCV(SVR(), param_grid=params)
    optimizer.fit(X_train, y_train)
    print "best parameters:", optimizer.best_params_
    model = optimizer.best_estimator_

    best parameters: {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}



    # 12. Check the performance of our *out of sample* predictions on the test
    # dataset using mean squared error.
    # Compare this the performance of the model on the training set.
    
    print "    in sample MSE:", mean_squared_error(y_train, model.predict(X_train))
    print "out of sample MSE:", mean_squared_error(y_test, model.predict(X_test))

        in sample MSE: 28.7286101515
    out of sample MSE: 122.85847184



    # 13. As a sanity check, determine how much better you're doing than
    # the mean estimator
    
    dummy = DummyRegressor().fit(X_train, y_train)
    
    print "    dummy in sample MSE:", mean_squared_error(y_train, dummy.predict(X_train))
    print "dummy out of sample MSE:", mean_squared_error(y_test, dummy.predict(X_test))

        dummy in sample MSE: 142.367595485
    dummy out of sample MSE: 320.466734148



    # 14. Make a graph of the true data and your prediction with the best model.
    
    plt.figure(figsize=(18, 8))
    plt.plot(ts.index, ts.values, lw=0.5, label='true')
    plt.plot(ts.index.values[:split_index], model.predict(X_train), lw=0.5, label='in sample prediction')
    plt.plot(ts.index.values[split_index:], model.predict(X_test), lw=0.5, label='out of sample prediction')
    plt.legend()




    <matplotlib.legend.Legend at 0x10ac843d0>




![png](hitch_timeseries_files/hitch_timeseries_14_1.png)



    # 15. Use statsmodels autocorrelation function to get a visual representation
    # of what the appropriate lag should be
    
    nlags = 14 * 24  # 14 days over 24 hours
    acf_data = sm.tsa.acf(df['count'], nlags=nlags)
    acf_series = pd.Series(data=acf_data, index=np.arange(nlags + 1) / 24.)
    acf_series.plot()
    
    ordered_lags = np.argsort(acf_data)
    print "top lags:"
    for lag in ordered_lags[-1:-9:-1]:
        print "%6d (%.2f days): %.2f" % (lag, lag / 24., acf_data[lag])

    top lags:
         0 (0.00 days): 1.00
         1 (0.04 days): 0.85
       168 (7.00 days): 0.74
         2 (0.08 days): 0.73
       167 (6.96 days): 0.70
       169 (7.04 days): 0.70
       166 (6.92 days): 0.61
       170 (7.08 days): 0.60



![png](hitch_timeseries_files/hitch_timeseries_15_1.png)



    # 16. Use statsmodels ARMA (autoregressive moving model) to build a
    # better model of the data.
    
    train_series = df['count'][:split_index]
    test_series = df['count'][split_index:]
    
    model = sm.tsa.AR(endog=train_series).fit(maxlag=nlags)
    
    start_time = train_series.index[model.k_ar].isoformat()
    end_time = test_series.index[-1].isoformat()
    prediction = model.predict(start=start_time, end=end_time, dynamic=True)
    
    print "    in sample MSE:", mean_squared_error(train_series[model.k_ar:], prediction[:-len(test_series)])
    print "out of sample MSE:", mean_squared_error(test_series, prediction[-len(test_series):])

        in sample MSE: 27.0814085951
    out of sample MSE: 94.2746936466



    # 17. Plot the predictions from ARMA on top of the real values!
    
    plt.figure(figsize=(18, 8))
    plt.plot(ts.index, ts.values, lw=0.5, label='true')
    plt.plot(train_series.index.values[model.k_ar:], prediction[:-len(test_series)], lw=0.5, label='in sample prediction')
    plt.plot(test_series.index.values, prediction[-len(test_series):], lw=0.5, label='out of sample prediction')
    plt.legend()




    <matplotlib.legend.Legend at 0x10b7b3510>




![png](hitch_timeseries_files/hitch_timeseries_17_1.png)

