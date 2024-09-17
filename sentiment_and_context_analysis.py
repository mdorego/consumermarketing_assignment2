import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from wordcloud import WordCloud

# Define a set of stop words
STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

EXCLUDE_FROM_WORDCLOUD = ['mother', 'mom', 'mum', 'mommy', 'maternal', 'father', 'dad', 'daddy', 'paternal']

def simple_tokenize(text):
    return [word for word in re.findall(r'\b\w+\b', text.lower()) 
            if word not in STOP_WORDS and word not in EXCLUDE_FROM_WORDCLOUD]

def get_context(text, term, window=5):
    words = simple_tokenize(text)
    contexts = []
    for i, word in enumerate(words):
        if word == term:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            context = ' '.join(words[start:end])
            contexts.append(context)
    return contexts

def analyze_gender_context_and_sentiment(df):
    mother_terms = ['mother', 'mom', 'mum', 'mommy', 'maternal']
    father_terms = ['father', 'dad', 'daddy', 'paternal']
    
    mother_contexts = []
    father_contexts = []
    mother_sentiments = []
    father_sentiments = []
    
    for _, row in df.iterrows():
        text = row['combined_text'].lower()
        for term in mother_terms:
            contexts = get_context(text, term)
            mother_contexts.extend(contexts)
            mother_sentiments.extend([TextBlob(context).sentiment.polarity for context in contexts])
        
        for term in father_terms:
            contexts = get_context(text, term)
            father_contexts.extend(contexts)
            father_sentiments.extend([TextBlob(context).sentiment.polarity for context in contexts])
    
    return mother_contexts, father_contexts, mother_sentiments, father_sentiments

def safe_average(lst):
    return sum(lst) / len(lst) if lst else 0

def safe_plot_sentiment_distribution(mother_sentiments, father_sentiments):
    plt.figure(figsize=(10, 6))
    if mother_sentiments:
        sns.kdeplot(mother_sentiments, fill=True, label='Mother')
    if father_sentiments:
        sns.kdeplot(father_sentiments, fill=True, label='Father')
    plt.title('Sentiment Distribution: Mother vs Father')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Density')
    if mother_sentiments or father_sentiments:
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.savefig('sentiment_distribution.png')
    plt.close()

def preprocess_for_lda(text):
    tokens = simple_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    return [token for token in tokens if token not in GENSIM_STOPWORDS]

def perform_lda(texts, num_topics=3):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    return lda_model, dictionary, corpus

def analyze_gender_topics(df, mother_terms, father_terms):
    mother_texts = []
    father_texts = []
    
    for _, row in df.iterrows():
        text = row['combined_text'].lower()
        mother_contexts = [context for term in mother_terms for context in get_context(text, term, window=10)]
        father_contexts = [context for term in father_terms for context in get_context(text, term, window=10)]
        
        mother_texts.append(preprocess_for_lda(' '.join(mother_contexts)))
        father_texts.append(preprocess_for_lda(' '.join(father_contexts)))
    
    mother_lda, mother_dict, _ = perform_lda(mother_texts)
    father_lda, father_dict, _ = perform_lda(father_texts)
    
    return mother_lda, father_lda, mother_dict, father_dict

def safe_plot_word_frequency(word_freq, title, filename):
    plt.figure(figsize=(12, 6))
    words, counts = zip(*word_freq.most_common(20)) if word_freq else ([], [])
    if words and counts:
        sns.barplot(x=list(counts), y=list(words))
        plt.title(title)
        plt.xlabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_wordcloud(text, title, filename):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=set(STOP_WORDS).union(set(EXCLUDE_FROM_WORDCLOUD))).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(filename)
        plt.close()
        return True
    return False

def plot_topic_distribution(lda_model, dictionary, title, filename):
    topic_weights = []
    for i, topic_dist in enumerate(lda_model[lda_model.corpus]):
        topic_weights.append([w for i, w in topic_dist])
    
    df_topic_weights = pd.DataFrame(topic_weights)
    
    topic_top_words = [[word for word, prob in lda_model.show_topic(topic_idx, topn=5)] for topic_idx in range(lda_model.num_topics)]
    df_top_words = pd.DataFrame(topic_top_words).T
    df_top_words.columns = ['Topic '+str(i) for i in range(lda_model.num_topics)]
    
    plt.figure(figsize=(16, 6))
    sns.heatmap(df_topic_weights.transpose(), annot=df_top_words, fmt='', cmap='YlGnBu')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # Load the preprocessed data
    df = pd.read_csv('preprocessed_data_for_nlp.csv')
    
    mother_terms = ['mother', 'mom', 'mum', 'mommy', 'maternal']
    father_terms = ['father', 'dad', 'daddy', 'paternal']
    
    # Analyze gender context and sentiment
    mother_contexts, father_contexts, mother_sentiments, father_sentiments = analyze_gender_context_and_sentiment(df)
    
    # Print some example contexts
    print("Example contexts for mother-related terms:")
    for context in mother_contexts[:5]:
        print(f"- {context}")
    
    print("\nExample contexts for father-related terms:")
    for context in father_contexts[:5]:
        print(f"- {context}")
    
    # Calculate average sentiments
    avg_mother_sentiment = safe_average(mother_sentiments)
    avg_father_sentiment = safe_average(father_sentiments)
    
    print(f"\nAverage sentiment for mother-related contexts: {avg_mother_sentiment:.4f}")
    print(f"Average sentiment for father-related contexts: {avg_father_sentiment:.4f}")
    
    # Plot sentiment distribution
    safe_plot_sentiment_distribution(mother_sentiments, father_sentiments)
    print("Sentiment distribution plot saved as 'sentiment_distribution.png'")
    
    # Analyze most common words in contexts
    mother_words = [word for context in mother_contexts for word in simple_tokenize(context)]
    father_words = [word for context in father_contexts for word in simple_tokenize(context)]
    
    mother_freq = Counter(mother_words)
    father_freq = Counter(father_words)
    
    # Plot word frequency
    safe_plot_word_frequency(mother_freq, 'Most Common Words in Mother-related Contexts', 'mother_word_frequency.png')
    safe_plot_word_frequency(father_freq, 'Most Common Words in Father-related Contexts', 'father_word_frequency.png')
    print("Word frequency plots saved as 'mother_word_frequency.png' and 'father_word_frequency.png'")
    
    # Create word clouds
    if create_wordcloud(' '.join(mother_words), 'Mother-related Word Cloud', 'mother_wordcloud.png'):
        print("Mother-related word cloud saved as 'mother_wordcloud.png'")
    else:
        print("No data available for mother-related word cloud")
    
    if create_wordcloud(' '.join(father_words), 'Father-related Word Cloud', 'father_wordcloud.png'):
        print("Father-related word cloud saved as 'father_wordcloud.png'")
    else:
        print("No data available for father-related word cloud")

    # Perform gender-specific topic modeling
    if mother_contexts and father_contexts:
        mother_lda, father_lda, mother_dict, father_dict = analyze_gender_topics(df, mother_terms, father_terms)
        
        # Plot topic distributions
        plot_topic_distribution(mother_lda, mother_dict, 'Topic Distribution in Mother-related Contexts', 'mother_topic_distribution.png')
        plot_topic_distribution(father_lda, father_dict, 'Topic Distribution in Father-related Contexts', 'father_topic_distribution.png')
        print("Topic distribution plots saved as 'mother_topic_distribution.png' and 'father_topic_distribution.png'")
    else:
        print("Insufficient data for topic modeling")

if __name__ == "__main__":
    main()