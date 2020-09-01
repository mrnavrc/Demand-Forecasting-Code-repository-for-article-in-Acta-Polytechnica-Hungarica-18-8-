
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import time
start_time = time.time()

df = pd.read_csv("DATASET.csv")
df.Date = pd.to_datetime(df.Date, format="%d/%m/%y")
df.Transakce = df['Demand'].astype(float)
df = df.sort_index()
y = df

y_to_train = y.iloc[:(len(y)-90)]
y_to_test = y.iloc[(len(y)-90):]

from tbats import BATS, TBATS

estimator = TBATS(seasonal_periods=(7, 365))
model = estimator.fit(y_to_train["Demand"])
y_forecast = model.forecast(steps=90)

y_test = y_to_test["Demand"]
y_test = y_test.reset_index()

plt.plot(y_forecast, label="Pred", color="black", zorder=1)
plt.plot(y_test["Demand"], label="True", color="lightgray", zorder=0)
plt.legend(loc="upper right")
plt.xlabel('Days', fontsize=10)
plt.ylabel('Demand', fontsize=10)

Y_true = y_test["Demand"]
Y_pred = y_forecast

from sklearn.metrics import mean_squared_error,mean_absolute_error
MSE = mean_squared_error(Y_true,Y_pred) 
MAE = mean_absolute_error(Y_true,Y_pred)

print(MSE)
print(MAE)

print(model.summary())
print("--- %s seconds ---" % (time.time() - start_time))