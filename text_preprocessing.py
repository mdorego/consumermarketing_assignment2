import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def simple_tokenize(text):
    # Simple word tokenization using regex
    return re.findall(r'\b\w+\b', text.lower())

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def get_sentiment(text):
    if not isinstance(text, str):
        return 0
    return TextBlob(text).sentiment.polarity

def main():
    try:
        # Read the CSV file
        df = pd.read_csv('google_search_results_with_extracted_content.csv')
        
        # Combine title, main content, and meta description
        df['combined_text'] = df['titles'].fillna('') + ' ' + df['main_contents'].fillna('') + ' ' + df['meta_descriptions'].fillna('')
        
        # Preprocess the combined text
        df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)
        
        # Get sentiment scores
        df['sentiment_score'] = df['combined_text'].apply(get_sentiment)
        
        # Save the results
        df.to_csv('preprocessed_data_for_nlp.csv', index=False)
        print("Preprocessing complete. Results saved to 'preprocessed_data_for_nlp.csv'")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()