import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


df = pd.read_csv("DATASET.csv")
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index, format="%d/%m/%y")
df.Transakce = df['Demand'].astype(float)
y = df

y_to_train = y.iloc[:(len(y)-365)]
y_to_test = y.iloc[(len(y)-365):]
train_len=len(y_to_train)

model = pm.auto_arima(y_to_train, 
                      seasonal=True, m=7,
                      d=0, D=1, 
                      start_p=0, start_q=0,
                      max_p=2, max_q=2,
                      max_P=2, max_Q=2,
                      stepwise=True,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      ) 
print(model.summary())

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y["Demand"],
                order=(0, 0, 2),  
                seasonal_order=(0, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
                )
results = model.fit()
results.plot_diagnostics(figsize=(6, 6))
plt.show()
results.summary()

pred_dynamic = results.get_prediction(start=train_len, dynamic=True)
Y_pred = np.array(pred_dynamic.predicted_mean)
Y_true = np.array(y_to_test)

plt.plot(Y_pred, label="Pred", color="black", zorder=1)
plt.plot(Y_true, label="True", color="lightgray", zorder=0)
plt.legend(loc="upper right")
plt.xlabel('Days', fontsize=10)
plt.ylabel('Demand', fontsize=10)

from sklearn.metrics import mean_squared_error,mean_absolute_error
MSE = mean_squared_error(Y_true,Y_pred) 
MAE = mean_absolute_error(Y_true,Y_pred)

print(MSE)
print(MAE)