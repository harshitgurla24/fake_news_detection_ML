import pickle
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

test_news = "Government launches new education policy"

cleaned = clean_text(test_news)
vector = vectorizer.transform([cleaned])
prediction = model.predict(vector)

print("Real" if prediction[0] == 1 else "Fake")
