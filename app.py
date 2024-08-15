from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Paths to models and vectorizers
MODEL_PATH = './model/'
logistic_model_file = MODEL_PATH + 'logistic_regression_model.pkl'
word2vec_model_file = MODEL_PATH + 'word2vec_model.pkl'
tfidf_vectorizer_file = MODEL_PATH + 'tfidf_vectorizer.pkl'

# Load models and vectorizers
with open(logistic_model_file, 'rb') as file:
    logistic_model = pickle.load(file)

with open(word2vec_model_file, 'rb') as file:
    word2vec_model = pickle.load(file)

with open(tfidf_vectorizer_file, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Define function for Word2Vec embeddings
def get_word2vec_embeddings(texts, model, size):
    embeddings = []
    for text in texts:
        words = text.split()
        vecs = [model.wv[word] for word in words if word in model.wv]
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
        else:
            mean_vec = np.zeros(size)
        embeddings.append(mean_vec)
    return np.array(embeddings)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the request
        data = request.json
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        # Process the texts
        X_word2vec = get_word2vec_embeddings(texts, word2vec_model, 100)

        # Make predictions
        tfidf_predictions = logistic_model.predict(X_word2vec)
        
        # Convert predictions to readable labels
        category_labels = {0: 'Politics', 1: 'Technology', 2: 'Entertainment', 3: 'Business'}
        predictions = [category_labels[pred] for pred in tfidf_predictions]

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Run the app with specified port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
