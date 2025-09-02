from flask import Flask, request, jsonify
from classifier import TextClassifier

app = Flask(__name__)
classifier = TextClassifier()

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    texts = data.get("texts")
    labels = data.get("labels")

    if not texts or not labels:
        return jsonify({"error": "texts and labels are required"}), 400

    classifier.train(texts, labels)
    return jsonify({"message": "Model trained successfully!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "text is required"}), 400

    label = classifier.predict(text)
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
