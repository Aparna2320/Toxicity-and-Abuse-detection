import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("cleaned_toxic_dataset.csv")

X = df['text']
y = df['label'].map({'safe': 0, 'toxic': 1})  # Convert labels to binary

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000, min_df=5)),
    ('clf', LogisticRegression(class_weight='balanced')) 
])

pipeline.fit(X, y)

joblib.dump(pipeline, 'models/model.pkl')

print(" Model trained and saved as models/model.pkl")
