## Miniquiz: Pandas Practice

**Include the code in** ```miniquiz.md```.

Load ```data/salary_data.csv``` into a pandas dataframe and answer the following
questions.

(**Estimated time: 30 minutes**)

1. Rename the columns as ```['name', 'job_title', 'department', 'salary']```

   ```python
   df = pd.read_csv('data/salary.csv')
   df.rename(columns={'Name': 'name',
                      'Position Title': 'job_title',
                      'Department':'department',
                      'Employee Annual Salary':'salary',
                       'Join Date': 'join_date'},
                       inplace=True)
   ```

2. Check the data types in each column using ```df.info()```. Which column
   has the wrong type? Use ```apply``` on the column to replace it with
   the right type.

   ```python
   df.info()
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32160 entries, 0 to 32159
    Data columns (total 4 columns):
    name          32160 non-null object
    job_title     32160 non-null object
    department    32160 non-null object
    salary        32160 non-null float64
    dtypes: float64(1), object(3)

   # Salary is supposed to be int or float but there is a '$' at the start

   # Parsing into float
   df['salary'] = df['salary'].apply(lambda x: float(x.lstrip('$')))
   ```

3. List the top 5 job title with the highest salary on average?

   ```python
   df.groupby('job_title')['salary'].mean().order(ascending=False)
   ```

4. Find the number of people who has the word ``POLICE`` in his / her
   ``job_title``. Read about ```.str.contain()``` [here](http://stackoverflow.com/questions/11350770/pandas-dataframe-select-by-partial-string).

   ```python
   df[df['job_title'].str.contains('POLICE')].count()
   ```

5. For the people who has the word ``POLICE`` in his / her ``job_title``,
   what percentage of them are ```POLICE OFFICER```? This is achievable
   in one line of code.

   ```python
   df[df['job_title'].str.contains('POLICE')]['job_title'].value_counts(normalize=1)
   ```

6. How many people joined between 2000-07-13 and 2000-08-13? Set the
   ```join_date``` column as the index. Sort by the dataframe by the index.
   Select rows of a particular time window
   by ```df.ix['2000-07-13' : '2000-08-13'].

   ```python
   df.ix['2000-07-13 00:00:00' : '2000-08-13 00:00:00']['name'].count()
   ```

