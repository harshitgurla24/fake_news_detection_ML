from flask import Flask, render_template, request
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
