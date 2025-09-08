import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

# linear reggression

x = np.array([[10], [2], [7]])#часы подготовки
y = np.array([1, 0, 1])#результат экзамена

model = LogisticRegression()
model.fit(x,y)

amount_hours = np.array([[5]])
prediction = model.predict(amount_hours)

if prediction[0] == 1:
    print(f"Скорее всего сдаст")
else:
    print(f"Скорее всего не сдаст")
