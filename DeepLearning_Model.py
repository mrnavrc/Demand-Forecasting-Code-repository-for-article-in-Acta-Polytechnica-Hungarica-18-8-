import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()

df = pd.read_csv("DATASET.csv")
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index, format="%d/%m/%y")
df.Transakce = df['Demand'].astype(float)
df.plot()
len(df)

train_len = len(df)-90
train = df.iloc[:train_len]
test = df.iloc[train_len:]
len(test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 90
n_features=1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
model = Sequential()
model.add(LSTM(40, return_sequences=True, activation='relu', input_shape=(n_input, n_features)))
#model.add(LSTM(40,return_sequences=False, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
model.summary()
model.fit_generator(generator,epochs=100)
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

first_eval_batch = scaled_train[-90:]
first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, 90, n_features))
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
true_predictions
test['Predictions'] = true_predictions
Y_true = df.iloc[train_len:]
Y_pred = test["Predictions"]

from sklearn.metrics import mean_squared_error,mean_absolute_error
MSE = mean_squared_error(Y_true,Y_pred) 
MAE = mean_absolute_error(Y_true,Y_pred)

plt.plot(test["Predictions"], label="Pred", color="black", zorder=1)
plt.plot(test["Demand"], label="True", color="lightgray", zorder=0)
plt.legend(loc="upper right")
plt.xlabel('Days', fontsize=10)
plt.ylabel('Demand', fontsize=10)

print(MSE)
print(MAE)