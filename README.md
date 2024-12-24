import logging
import re
import sqlite3
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ProcessPoolExecutor
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration (e.g., frames, database settings, etc.)
with open("config.json") as f:
    config = json.load(f)

frames = config["frames"]
db_name = config["db_name"]

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Compile regex pattern
pattern = re.compile(r"\bgovernment\b", re.IGNORECASE)

# Function for text preprocessing: tokenization, stopwords removal, and lemmatization
def preprocess_text(text):
    try:
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        logging.error(f"Error in preprocess_text: {e}")
        return ""

# Function to classify frames based on text and predefined categories
def classify_frame(text, frames):
    try:
        for frame, keywords in frames.items():
            if any(keyword in text for keyword in keywords):
                return frame
        return 'neutral'
    except Exception as e:
        logging.error(f"Error in classify_frame: {e}")
        return 'neutral'

# Function to match a pattern in the text
def match_pattern_in_text(text, pattern):
    try:
        return bool(pattern.search(text))
    except re.error as e:
        logging.error(f"Regex error: {e}")
        return False

# Function to insert batch data into the SQLite database
def insert_batch_to_db(db_name, data):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO documents (text, frame, preprocessed_text, pattern_matched) VALUES (?, ?, ?, ?)", data)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")

# Function to process a single document (preprocess, classify frame, match pattern)
def process_document(doc):
    try:
        preprocessed_text = preprocess_text(doc)
        frame = classify_frame(preprocessed_text, frames)
        matched = match_pattern_in_text(doc, pattern)
        return (doc, frame, preprocessed_text, matched)
    except Exception as e:
        logging.error(f"Error in process_document: {e}")
        return (doc, 'neutral', '', False)

# Parallel processing for handling large datasets
def process_documents_in_parallel(documents):
    try:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_document, documents))
        return results
    except Exception as e:
        logging.error(f"Error in process_documents_in_parallel: {e}")
        return []

# Visualization function for word cloud
def visualize_word_cloud(documents):
    try:
        all_text = ' '.join(documents)
        wordcloud = WordCloud(width=800, height=400).generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    except Exception as e:
        logging.error(f"Error in visualize_word_cloud: {e}")

# Topic modeling with LDA
def topic_modeling(documents, num_topics=5):
    try:
        # Create a document-term matrix
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(documents)

        # Apply LDA (Latent Dirichlet Allocation)
        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='batch', random_state=42)
        lda.fit(doc_term_matrix)

        # Display topics
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            print(f"Topic #{topic_idx}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
            print("\n")
    except Exception as e:
        logging.error(f"Error in topic_modeling: {e}")

# Function to save results to the database after processing
def save_results_to_db(documents):
    try:
        processed_data = process_documents_in_parallel(documents)
        insert_batch_to_db(db_name, processed_data)
    except Exception as e:
        logging.error(f"Error in save_results_to_db: {e}")

# Main function to run the entire pipeline
def main():
    try:
        # Sample documents for demonstration (replace with real data)
        documents = [
            "The government should lead this initiative to improve equity.",
            "There are harmful impacts associated with the proposed energy project.",
            "Collaboration between communities and companies is necessary for sustainable solutions."
        ]

        # Process documents and classify frames
        processed_results = process_documents_in_parallel(documents)

        # Save results to database
        save_results_to_db(documents)

        # Visualize word cloud
        visualize_word_cloud(documents)

        # Run topic modeling
        topic_modeling(documents)

    except Exception as e:
        logging.error(f"Error in main: {e}")

# Run the script
if __name__ == "__main__":
    main()

