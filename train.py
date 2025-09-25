import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset and clean
data = pd.read_csv('data/combined.csv')
data = data.dropna(subset=['Headline', 'NewsSnippet', 'Label'])

# Print label distribution - debugging
print("Label distribution:\n", data['Label'].value_counts())

# Combine headline and snippet
data['Text'] = data['Headline'] + " " + data['NewsSnippet']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=80)
X = vectorizer.fit_transform(data['Text'])
y = data['Label']

# Split train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training completed and model/vectorizer saved.")
