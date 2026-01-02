from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Program start ho gaya")

news = [
    "Government announces new education policy",
    "Aliens landed in Delhi yesterday",
    "Scientists discovered new planet",
    "Drinking cow urine cures cancer"
]

labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news)

model = LogisticRegression()
model.fit(X, labels)

user_news = input("News likho yahan: ")

test = vectorizer.transform([user_news])
result = model.predict(test)

if result[0] == 1:
    print("REAL NEWS")

    print("FAKE NEWS")
