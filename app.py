from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

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

# Function to compute Word2Vec embeddings
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

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the form data
        text = request.form['text']
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Process the text
        X_word2vec = get_word2vec_embeddings([text], word2vec_model, 100)

        # Make predictions
        predictions = logistic_model.predict(X_word2vec)
        
        # Convert predictions to readable labels
        category_labels = {0: 'Politics', 1: 'Technology', 2: 'Entertainment', 3: 'Business'}
        prediction_label = category_labels.get(predictions[0], 'Unknown')

        # Render the template with the prediction result
        return render_template('public/index.html', prediction=prediction_label)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for the index page
@app.route('/')
def index():
    return render_template('public/index.html')  # Ensure index.html is in the correct folder

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
