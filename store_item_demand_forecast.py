import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import seaborn as sns
sns.set(
    rc={
        "figure.figsize": (20, 8)
    }
)

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Load train dataset
TRAIN_DATA_PATH = r"C:/Users/manish.kumar/Downloads/train.csv"
df_train = pd.read_csv(TRAIN_DATA_PATH)

#for 3 store data
#df_train =df_train[df_train['store'].isin([1, 2, 3])]

# Convert date column to datetime type
df_train["date"] = pd.to_datetime(df_train["date"])
# Check missing values
print(df_train.isna().sum())

# Check 0 in sales
print(df_train[df_train["sales"] == 0])
print(df_train.shape)
# Impute the zero value with minimum value
df_train.loc[df_train["sales"] == 0, "sales"] = df_train[df_train["sales"] != 0]["sales"].min()
sns.histplot(df_train, x="sales", bins=30)

#daily
# Check overall trend of sales
sales_agg = df_train.groupby("date")[["sales"]].mean()
sales_agg.plot()


#monthly
# Check monthly trend
sales_agg_monthly = df_train.resample("M", on="date")[["sales"]].mean()
sales_agg_monthly.plot()


#Yearly
# Check yearly trend
sales_agg_yearly = df_train.resample("Y", on="date")[["sales"]].mean()
sales_agg_yearly.plot()



#Growth by Store
sales_agg_yearly_store = df_train.groupby(by=["store", df_train["date"].dt.year])[["sales"]].mean().reset_index()
sales_agg_yearly_store.pivot(index="date", columns="store", values="sales").plot()



#Growth by Item
sales_agg_yearly_store = df_train.groupby(by=["item", df_train["date"].dt.year])[["sales"]].mean().reset_index()
sales_agg_yearly_store.pivot(index="date", columns="item", values="sales").plot(figsize=(16,16))


playoffs = pd.DataFrame(
{
    "holiday": "playoff",
    "ds": ['2013-07-12', '2014-07-12', '2014-07-19',
                 '2014-07-02', '2015-07-11', '2016-07-17',
                 '2016-07-24', '2016-07-07','2016-07-24'],
    "lower_window": 0,
    "upper_window": 1
}
)
superbowls = pd.DataFrame(
{
    "holiday": "superbowl",
    "ds": ['2013-01-01', '2013-12-25', '2014-01-01', '2014-12-25','2015-01-01', '2015-12-25','2016-01-01', '2016-12-25',
                '2017-01-01', '2017-12-25'],
    "lower_window": 0,
    "upper_window": 1
}
)
holidays = pd.concat([playoffs, superbowls], ignore_index=True)



def train(data, holidays=None):
    # Log transformation
    data["y"] = np.log(data["y"])
    # Train the model
    model = Prophet(interval_width=0.95, holidays=holidays)
    model.fit(data)
    
    # Get prediction
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Cross validation
#     df_cv = cross_validation(model, initial="365 days", period="30 days", horizon="90 days")
#     df_p = performance_metrics(df_cv, rolling_window=1)
#     print(df_p)
    
    # Inverse log transformation
    data["y"] = np.exp(data["y"]).astype(int)
    forecast["yhat"] = np.exp(forecast["yhat"]).astype(int)
    
    # Calculate SMAPE
    train_predict = data.merge(forecast[["ds", "yhat"]], on="ds")
    smape = (2 * np.abs(train_predict["y"] - train_predict["yhat"]) / (train_predict["y"].abs() + train_predict["yhat"].abs())).mean()
    print(smape)
    
    return model, forecast[["ds", "yhat"]]




forecast_list = []
for store in df_train.store.unique():
    for item in df_train.item.unique():
        print(f"store {store}, item {item}")
        data = df_train[(df_train["store"]==store) & (df_train["item"]==item)][["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
        model, forecast = train(data, holidays)
        forecast_list.append(forecast)




max_train_date = df_train["date"].max()

forecast_list_future = [forecast_df[forecast_df["ds"] > max_train_date] for forecast_df in forecast_list]



for i in range(0, len(forecast_list_future)):
    forecast_list_future[i]["store"] = (i // 50) + 1
    forecast_list_future[i]["item"] = (i % 50) + 1
    
    
    

df_forecast = pd.concat(forecast_list_future, ignore_index=True)
df_forecast
