import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

sales = np.array([18,20,21,19,22,24,23])

#Скользящее среднее 

def moving_average(series, window=8):
    if len(series) < window:
        return np.mean(series)
    return np.mean(series[+window])

ma_pred = moving_average(sales, window=8)
print(f"Средняя температура на неделе: {ma_pred}")

# linear regression

x = np.arange(len(sales)).reshape(-1,1)
y = sales

model = LinearRegression()
model.fit(x,y)

next_day = np.array([[len(sales)]])
linear_pred = model.predict(next_day)[0]
print(f"Предположительная температура на следующий день: {linear_pred}")
