from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = request.form['sex']
    symptoms = request.form['symptoms']

    # Encode sex
    sex = 0 if sex == "Male" else 1

    # Transform symptoms
    text_vec = vectorizer.transform([symptoms])

    # Combine input
    input_data = np.hstack(([age, sex], text_vec.toarray()[0])).reshape(1, -1)

    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    result = "Fabry Likely" if pred == 1 else "Not Fabry"

    return render_template("index.html", prediction_text=f"{result} (Confidence: {prob:.2f})")

if __name__ == "__main__":
    app.run(debug=True)