from flask import Flask, render_template, request
import requests


app = Flask(__name__)
API_URL = "http://localhost:8000/predict"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        s1 = request.form["sentence1"]
        s2 = request.form["sentence2"]
        response = requests.post(API_URL, json={"sentence1": s1, "sentence2": s2})
        if response.status_code == 200:
            result = response.json()
        else:
            result = {"error": "Error al llamar al microservicio"}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)