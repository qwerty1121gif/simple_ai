from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
text = [
    "Купи афон за 100 рублей", #spam
    "Встреча в офисе в 10.00", #not spam
    "Вы выиграли миллион, пришлите денег",#spam
    "Отчет: продажи за квартал 2025",#not spam
    "Срочно! Позвони сейчас и получи приз",#spam
    "Напоминаем про оплату счета"#not spam
]

lable = [1,0,1,0,1,0] # 1 - spam, 0 - not spam

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=56)

pipe = make_pipeline(
    CountVectorizer(), 
    MultinomialNB() 
)
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)

print(f"Accurcy:{accuracy_score(y_test, y_pred)}")

print("New:", pipe.predict(["Добрый вечер! Встреча завтра в 12"])[0])
print("New:", pipe.predict(["Поздравляем! Вы победитель. Пришлите нам свой CVV код"])[0])
