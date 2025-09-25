from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load dataset and prepare headline lookup dictionary
data = pd.read_csv('data/combined.csv')
headline_lookup = dict()
for _, row in data.iterrows():
    headline_lookup[row['Headline'].strip().lower()] = {
        'label': row['Label'],
        'news_snippet': row['NewsSnippet']
    }

# Load ML model and vectorizer (optional fallback if needed)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fake-news', methods=['POST'])
def fake_news():
    req_data = request.get_json()
    headline = req_data.get('text', '').strip().lower()

    if not headline:
        return jsonify({'error': 'No headline provided'}), 400

    if headline in headline_lookup:
        record = headline_lookup[headline]
        label = 'REAL' if record['label'] == 1 else 'FAKE'
        confidence = 1.0
        snippet = record['news_snippet']
    else:
        # Optional ML fallback uncomment if needed
        # vect_text = vectorizer.transform([headline])
        # prediction = model.predict(vect_text)[0]
        # probabilities = model.predict_proba(vect_text)[0]
        # confidence = max(probabilities)
        # label = 'REAL' if prediction == 1 else 'FAKE'
        label = 'UNKNOWN'
        confidence = 0.0
        snippet = ''

    return jsonify({
        'label': label,
        'confidence': confidence,
        'snippet': snippet
    })

if __name__ == '__main__':
    app.run(debug=True)
