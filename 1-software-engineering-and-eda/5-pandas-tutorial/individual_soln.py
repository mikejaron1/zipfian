### To just do some basic exploratory analysis, compute the most common cause of hospitalization (affliction) Compare the execution time of using value_counts() vs. counting with groupby()

import pandas as pd

## 1) FIND WAS THE MOST COMMON ILLESS... 
### TO FIND THAT YOU HAVE TO GROUPBY THE ILLNESS, AND SUM THE DISCHARGES
## HERE IS HOW I GOT THAT
df = pd.read_csv("../data/hospital-costs.csv")
cc = df[["APR DRG Description", "Discharges"]]
cc = cc.groupby(cc["APR DRG Description"]).sum()
cc = cc.sort("Discharges",ascending=False)
cc.head(5)


## Count That CASH


import pandas as pd
df = pd.read_csv("../data/hospital-costs.csv")


# 1:  Create a new column that is the Discharges x Mean Charge. Then do the same for the "Mean Cost".
df['Total Charge'] = df['Discharges'] * df['Mean Charge']
df['Total Cost'] = df['Discharges'] * df['Mean Cost']


# 2:  With theses two new Total Charges and Total Costs columns, calculate the Markup for each row.
df['Markup'] = df['Mean Charge'] / df['Mean Cost']


# 3: Tell me which procedure has the highest markup, and which one has the lowest markup
lowest = df.sort("Markup")
highest = df.sort("Markup", ascending=False)
lowest.head(1)
# print highest.head(1)


## Find that MONEY

# In[7]:

df = pd.read_csv("../data/hospital-costs.csv")
df['Total Charge'] = df['Discharges'] * df['Mean Charge']
df['Total Cost'] = df['Discharges'] * df['Mean Cost']
df['Markup'] = df['Mean Charge'] / df['Mean Cost']

#1:
net = df[['Facility Name', 'Total Charge', 'Total Cost']]
#2: 
net = net.groupby(net['Facility Name']).sum()
#3:
net['Net Income'] = net['Total Charge'] - net["Total Cost"]
lowest_net = net.sort("Net Income").head(1)
highest_net = net.sort("Net Income", ascending=False).head(1)

net
# lowest_net
# highest_net


####### 1. Group all the levels of severity for each Illness... 2. What is the most expensive type of illness?

# In[10]:

df = pd.read_csv("../data/hospital-costs.csv")

# 1: Group all the levels of severity for each Illness...
double_group = df.groupby( (df["APR DRG Description"], df["APR Severity of Illness Description"]))

# 2: What is the most expensive type of illness?
answer = double_group["Mean Charge"].mean()
answer.head(1)


# In[ ]:




#### Now, lets focus in on "Viral Meningitis"

# In[13]:

df = pd.read_csv("../data/hospital-costs.csv")

# 1: Create a new dataframe that only contains the data corresponding to "Viral Meningitis"
newdf = df[df["APR DRG Description"] == "Viral Meningitis"]

# 2: Now, with our newdf, only keep the data columns we care about...
newdf = newdf[["Facility Name", "APR DRG Description","Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]

# 3: Our newdf should look somewhat like this:
# newdf


#### Now that we have our new clean df, lets get into some gravy

# In[14]:

# 1: Find which hospital has the most cases of Viral Meningitis.
group_step = newdf.groupby("Facility Name")
apply_step = group_step["Discharges"].sum()
order_step = apply_step.order(ascending=False)
answer = order_step.head()
answer


# In[97]:

# 2: Find which hospital is the least expensive for treating Moderate cases of VM.
df = pd.read_csv("../data/hospital-costs.csv")
onlyVM = df[df["APR DRG Description"] == "Viral Meningitis"]
onlyModerate_VM = onlyVM[onlyVM["APR Severity of Illness Description"] == "Moderate"]

answer = onlyModerate_VM.sort("Mean Charge")
answer


# In[15]:

# Find which hospital is the least expensive for treating Moderate cases of VM that have at least 3 or more Discarges.
df = pd.read_csv("../data/hospital-costs.csv")
onlyVM = df[df["APR DRG Description"] == "Viral Meningitis"]
onlyModerate_VM = onlyVM[onlyVM["APR Severity of Illness Description"] == "Moderate"]
threeOrMore = onlyModerate_VM[onlyModerate_VM["Discharges"] >= 3]
threeOrMore


# In[89]:

# Find if there is a correlation between the severity of case, and the mean charge
df = pd.read_csv("../data/hospital-costs.csv")
grouped = df.groupby("APR Severity of Illness Description")
x = grouped[["APR Severity of Illness Code","Mean Charge"]].mean()
x = x.reset_index()
x[["APR Severity of Illness Code","Mean Charge"]].corr()


# In[18]:

# 2. Find which hospital has the best markup rate.
df = pd.read_csv("../data/hospital-costs.csv")
df['Total Charge'] = df['Discharges'] * df['Mean Charge']
df['Total Cost'] = df['Discharges'] * df['Mean Cost']
df['Markup'] = df['Mean Charge'] / df['Mean Cost']
df = df.groupby("Facility Name")
df = df["Markup"].mean()
answer = df.order()
answer.tail()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# NOTES AND OTHER WAYS TO DO IT INCASE YOU GIVE A WHOO


## ### CLOSER LOOK AT VIRAL MENINGITIS

# In[4]:

# 1. Create a new dataframe that only contains the data correspoindng to "Viral Meningitis"
df = pd.read_csv("../data/hospital-costs.csv")
meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR DRG Description","APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]


# In[6]:

# 2. Find which hospital has the most cases of Viral Meningitis.
meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR DRG Description","Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
grouped = meng.groupby(["Facility Name"])
grouped = grouped.sum()
grouped = grouped.sort("Discharges", ascending=False)
# grouped


# In[7]:

# 3. Find which hospital is the least expensive for treating Moderate cases of VM.

meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR DRG Description","APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
meng = meng[meng["APR Severity of Illness Description"] == "Moderate"]
meng = meng.sort("Mean Charge")
# meng.tail()


# In[14]:

# 4. Find which hospital is the least expensive for treating Moderate cases of VM **that have at least 3 or more Discarges**.
meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR DRG Description","APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
meng = meng[meng["APR Severity of Illness Description"] == "Moderate"]
meng = meng[meng["Discharges"] > 3]
meng = meng.sort("Mean Charge").head(3)
# meng.to_csv("../data/for-readme.csv", sep="|")


# In[3]:

# Find if there is a correlation between # of Discharges a hospital has and how much they charge. Hint use df.corr() 
df = pd.read_csv("../data/hospital-costs.csv")
meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
grouped = meng.groupby(["Facility Name"])
grouped = grouped.sum()
grouped = grouped.sort("Mean Charge", ascending=False)
grouped["Markup"] = grouped["Mean Charge"] / grouped["Mean Cost"]
grouped[["Discharges", "Mean Charge"]].corr()
### DONE WITH FINDING COST PER VIRAL MENINGITIS TREATMENT


# In[19]:

# Find if there is a correlation between # of Discharges a hospital has and how much they charge. Hint use df.corr() 
df = pd.read_csv("../data/hospital-costs.csv")


# meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = df
meng = meng[["Facility Name", "APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
meng["Total Charges"]  = meng["Discharges"] * meng["Mean Charge"]
meng["Total Costs"]  = meng["Discharges"] * meng["Mean Cost"]
meng["Markup"] = meng["Total Costs"] / meng["Total Charges"]

grouped = meng.groupby(["Facility Name"])
# grouped = grouped.sum()
# grouped = grouped.mean()

# grouped = grouped.sort("Mean Charge", ascending=False)
# grouped["Markup"] = grouped["Mean Charge"] / grouped["Mean Cost"]
grouped
# grouped[["Discharges", "Total Charges"]].corr()
# # ### DONE WITH FINDING COST PER VIRAL MENINGITIS TREATMENT


# In[10]:

### LETS FIND OUT HOW MUCH ONE WILL SAVE BY CATCHING MENINGIGITUS EARLIER AS SUPPOSED TO LATER
###  AND LETS SEE HOW MUCH MORE (IN MARKUP) THE HOPSITAL CHARGES PER CASE
### FIRST LETS LOOK AT ALL THE HOPSITALS AS ONE, THEN LETS BREAK THEM DOWN

meng = df[df["APR DRG Description"] == "Viral Meningitis"]

meng = meng[["Discharges", "Mean Charge", "Mean Cost"]].groupby(meng["APR Severity of Illness Description"])
mengDis = meng["Discharges"].mean()
mengDis
mengCharge = meng["Mean Charge"].median()
mengCharge
mengCost = meng["Mean Cost"].median()
mengCost

# theyCharge = meng["Mean Charge"].mean()
# theyCharge / itCosts


# In[13]:

#### LETS FIND OUT WHICH PROCEDURE IS MARKED UP THE MOST HOLISTICALLY
df = pd.read_csv('../data/hospital-costs-clean.csv')
df['Total Charge'] = df['Discharges'] * df['Mean Charge']
df['Total Cost'] = df['Discharges'] * df['Mean Cost']
df['Markup'] = df['Mean Charge'] / df['Mean Cost']
df = df[["APR DRG Description", "Total Charge", "Total Cost"]]
df = df.groupby("APR DRG Description").sum()
df["Markup"] = df["Total Charge"] / df["Total Cost"]
df.sort("Markup").tail()


# In[12]:

df = pd.read_csv('../exercise-data/hospital-costs-clean.csv')
df['Total Charge'] = df['Discharges'] * df['Mean Charge']
df['Total Cost'] = df['Discharges'] * df['Mean Cost']
df['Markup'] = df['Mean Charge'] / df['Mean Cost']
df


# In[23]:

### LETS FIND THE HOSPITAL WITH THE CHEAPEST TREATMENT FOR ALL Severity Levels of Viral Meningitis
meng = df[df["APR DRG Description"] == "Viral Meningitis"]
meng = meng[["Facility Name", "APR Severity of Illness Description", "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]
grouped = meng.groupby(["Facility Name"])
grouped = grouped.sum()
grouped = grouped.sort("Mean Charge", ascending=False)
grouped
# grouped["Markup"] = grouped["Mean Charge"] / grouped["Mean Cost"]
# grouped[["Discharges", "Markup"]].corr()
### DONE WITH FINDING COST PER VIRAL MENINGITIS TREATMENT


# In[ ]:



