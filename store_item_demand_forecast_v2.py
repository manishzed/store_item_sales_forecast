
#type-1---------------------


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from prophet import Prophet

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (16, 9)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

df_train = pd.read_csv(r"C:/Users/manish.kumar/Downloads/train.csv", parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv(r"C:/Users/manish.kumar/Downloads/test.csv", parse_dates=['date'], index_col=['date'])
df_train.shape, df_test.shape


df_train.head()



proph_results = df_test.reset_index()
proph_results['sales'] = 0

proph_results.head()


#a=df_train.loc[(df_train['store'] == 1) & (df_train['item'] == 2)].reset_index()
#a.head()


tic = time.time()

for s in proph_results['store'].unique():
    for i in proph_results['item'].unique():
        proph_train = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i)].reset_index()
        proph_train.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
        
        m = Prophet()
        m.fit(proph_train[['ds', 'y']])
        future = m.make_future_dataframe(periods=len(df_test.index.unique()), include_history=False)
        fcst = m.predict(future)
        
        proph_results.loc[(proph_results['store'] == s) & (proph_results['item'] == i), 'sales'] = fcst['yhat'].values
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}s".format(s, i, toc-tic))






#type-2--------------------------------------

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from prophet import Prophet

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (16, 9)

train_data = pd.read_csv(r"C:/Users/manish.kumar/Downloads/train.csv")


train_data.head()


train_data.isnull().sum()

#analysis


# Store ids
train_data.store.unique()
# item ids
train_data.item.unique()


pivot = train_data.pivot_table(index= 'item',columns='store',values='sales')

# Item wise sales average
item_pivot = pivot.mean(axis=1)


# Low volume item is defined as having average sales less than 30 , medium volume sales is defined as average sales between 30 and 60,
# High volume item is defined as having average sales greater than 60
items_low_vol = list(item_pivot[item_pivot<30].index)
items_med_vol = list(item_pivot[(30<=item_pivot)&(item_pivot<60)].index)
items_high_vol = list(item_pivot[60<=item_pivot].index)

print('Low volume stores list:',items_low_vol)
print('Medium volume stores list:',items_med_vol)
print('High volume stores list:',items_high_vol)

print('Count of Low volume stores list:',len(items_low_vol))
print('Count of Medium volume stores list:',len(items_med_vol))
print('Count of High volume stores list:',len(items_high_vol))

train_data_analysis = train_data.copy()
train_data_analysis['date'] = pd.to_datetime(train_data_analysis['date'])
train_data_analysis['dayofweek'] = train_data_analysis['date'].apply(lambda x: x.dayofweek)
train_data_analysis['month'] = train_data_analysis['date'].apply(lambda x: x.month)
#Observing the average sales on different days of week and different months using pivot

pivot_weekdays = train_data_analysis.pivot_table(index='store',columns='dayofweek',values='sales')
pivot_months = train_data_analysis.pivot_table(index='store',columns='month',values='sales')
pivot_weekdays



# Plotting the average sales daywise for everystore
%matplotlib inline
fig, axs = plt.subplots(10,figsize=(30,25))
for i in range(10):
    store = pivot_weekdays.index[i]
    value_list = pivot_weekdays[pivot_weekdays.index==store].values.T
    axs[i].plot(value_list)
    axs[i].set(xlabel='dayofweek', ylabel='Average Sales')
    axs[i].set_title(f'Store_{store}_Sales_average day wise')





fig, axs = plt.subplots(10,figsize=(30,25))
for i in range(10):
    store = pivot_months.index[i]
    value_list = pivot_months[pivot_months.index==store].values.T
    axs[i].plot(value_list)
    axs[i].set(xlabel='month', ylabel='Average Sales')
    axs[i].set_title(f'Store_{store}_Sales_average month wise')



train_data_analysis['year'] = train_data_analysis['date'].apply(lambda x: x.year)


import matplotlib as mpl
import matplotlib.pyplot as plt
# Plotting Seasonal plots for Store-1 , item 2(Low Volume)
store=1
item =2
df = train_data_analysis[(train_data_analysis['store']==1)&(train_data_analysis['item']==2)].reset_index(drop=True)
years = df['year'].unique()
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
no_plots = len(years)
fig,ax = plt.subplots(no_plots,figsize=(30,30))
for i, y in enumerate(years):
    ax[i].plot('date','sales', data=df.loc[df.year==y, :].reset_index(drop=True), color=mycolors[i], label=y)
    ax[i].set(xlabel='date', ylabel='Sales')
    ax[i].set_title(f'Store_{store}_item_{item}_Sales_value daily plot for year {y}')




def generate_prophet_forecast(data,test_length,store,item):
    prophet_forecast_obj = Prophet(yearly_seasonality=True)
    prophet_forecast_obj.fit(data)
    dateframes = prophet_forecast_obj.make_future_dataframe(periods=test_length,include_history=False)
    ypredict = prophet_forecast_obj.predict(dateframes)
    final_data = ypredict[['ds','yhat']]
    final_data['store'] = store
    final_data['item'] = item
    final_data = final_data[['ds','store','item','yhat']]
    final_data = final_data.rename(columns={'ds':'date','yhat':'sales_forecast_prophet'})
    final_data = final_data.sort_values(by='date').reset_index(drop=True)
    return final_data


def submission_file(final_result,test_data):
    test_data['date'] = pd.to_datetime(test_data['date'])
    merged_file = test_data.merge(final_result,on=['store','item','date'],suffixes=('','_drop'))
    merged_new = merged_file.sort_values(by=['store','item','date']).reset_index(drop=True)
    merged_part = merged_new[['id','sales_forecast_prophet']]
    merged_part = merged_part.rename(columns={'sales_forecast_prophet':'sales'})
    merged_part = merged_part.sort_values(by='id').reset_index(drop=True)
    return merged_part


def get_time_series_prophet(data,store,item):
    data_store = data[(data.store==store)&(data.item==item)].reset_index(drop=True)
    data_prophet = data_store[['date','sales']]
    data_prophet = data_prophet.rename(columns={'date':'ds','sales':'y'})
    return data_prophet

def generate_all_stores_forecast(data,test_length):
    final_result = pd.DataFrame()
    for store in data.store.unique():
        for item in data.item.unique():
            data_part = get_time_series_prophet(data,store,item)
            final_data = generate_prophet_forecast(data_part,test_length,store,item)
            final_result = final_result.append(final_data)
            print(f'Store Number :{store} , item number :{item} done')
    
    final_result = final_result.reset_index(drop=True)
    final_result = final_result.sort_values(by=['store','item','date']).reset_index(drop=True)
    return final_result




test_length = 90
final_result = generate_all_stores_forecast(train_data,test_length)
final_result.head()