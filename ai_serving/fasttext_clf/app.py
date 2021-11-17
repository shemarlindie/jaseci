import os

from flask import Flask, request, jsonify

from .train import load_model
from .predict import predict

app = Flask(__name__)
model = None


@app.route("/", methods=['POST'])
def index():
    global model
    if not model:
        model = load_model()

    params = request.json
    sentences = params.get('sentences')

    if sentences:
        response = predict(sentences, model)
    else:
        response = {'error': '"sentences" not found in request'}

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
