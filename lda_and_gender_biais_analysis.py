import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models

# Expanded lists of mother-related and father-related terms
mother_terms = [
    'mother', 'mom', 'mum', 'mommy', 'mama', 'ma', 'mam',
    'maternal', 'matriarch', 'matriarchal',
    'motherhood', 'motherly', 'mothering',
    'birth mother', 'biological mother', 'adoptive mother', 'stepmother',
    'maternity', 'maternity leave',
    'working mother', 'stay-at-home mom',
    'single mother', 'soccer mom',
    'expectant mother',
    'mother-in-law',
]

father_terms = [
    'father', 'dad', 'daddy', 'papa', 'pa', 'pops',
    'paternal', 'patriarch', 'patriarchal',
    'fatherhood', 'fatherly', 'fathering',
    'birth father', 'biological father', 'adoptive father', 'stepfather',
    'grandfather', 'grandpa', 'granddad', 'gramps',
    'paternity', 'paternity leave',
    'working father', 'stay-at-home dad',
    'single father',
    'expectant father',
    'father-in-law', 'godfather'
]

def perform_lda(texts, num_topics=3, passes=15):
    # Create a dictionary from the texts
    dictionary = corpora.Dictionary(texts)
    
    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Build the LDA model
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, workers=2)
    
    return lda_model, dictionary, corpus

def analyze_gender_bias(lda_model, dictionary):
    mother_counts = sum(dictionary.cfs[dictionary.token2id[term]] for term in mother_terms if term in dictionary.token2id)
    father_counts = sum(dictionary.cfs[dictionary.token2id[term]] for term in father_terms if term in dictionary.token2id)
    
    total_counts = mother_counts + father_counts
    mother_percentage = (mother_counts / total_counts) * 100 if total_counts > 0 else 0
    father_percentage = (father_counts / total_counts) * 100 if total_counts > 0 else 0
    
    return mother_percentage, father_percentage, mother_counts, father_counts

def get_top_terms(dictionary, term_list, top_n=10):
    term_counts = [(term, dictionary.cfs[dictionary.token2id[term]]) 
                   for term in term_list if term in dictionary.token2id]
    return sorted(term_counts, key=lambda x: x[1], reverse=True)[:top_n]

def plot_gender_bias(mother_percentage, father_percentage, mother_terms, father_terms):
    labels = ['Mother', 'Father']
    sizes = [mother_percentage, father_percentage]
    colors = ['pink', 'lightblue']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Pie chart
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Representation of Mother vs Father Terms')
    
    # Bar chart for top terms
    top_mother_terms = mother_terms[:5]  # Get top 5 mother terms
    top_father_terms = father_terms[:5]  # Get top 5 father terms
    
    all_terms = top_mother_terms + top_father_terms
    term_names = [term[0] for term in all_terms]
    term_counts = [term[1] for term in all_terms]
    
    bar_colors = ['pink'] * 5 + ['lightblue'] * 5  # Correct way to specify colors
    bars = ax2.bar(range(len(all_terms)), term_counts, color=bar_colors)
    ax2.set_xticks(range(len(all_terms)))
    ax2.set_xticklabels(term_names, rotation=45, ha='right')
    ax2.set_title('Top 5 Mother and Father Terms')
    ax2.set_ylabel('Frequency')
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gender_bias_pie_chart.png')
    plt.close()

def visualize_lda_results(lda_model, corpus, dictionary):
    # Prepare the visualization
    lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    
    # Save the visualization to an HTML file
    pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')
    print("LDA visualization saved as 'lda_visualization.html'")

def main():
    # Load the preprocessed data
    df = pd.read_csv('preprocessed_data_for_nlp.csv')
    
    # Prepare the texts for LDA
    texts = [text.split() for text in df['preprocessed_text'] if isinstance(text, str)]
    
    # Perform LDA
    lda_model, dictionary, corpus = perform_lda(texts)
    
    # Print the topics
    print("Top 10 words for each topic:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}\n")
    
    # Analyze gender bias
    mother_percentage, father_percentage, mother_counts, father_counts = analyze_gender_bias(lda_model, dictionary)
    
    print(f"\nGender Representation:")
    print(f"Mother-related terms: {mother_percentage:.2f}% ({mother_counts} occurrences)")
    print(f"Father-related terms: {father_percentage:.2f}% ({father_counts} occurrences)")
    
    mother_top_terms = get_top_terms(dictionary, mother_terms)
    father_top_terms = get_top_terms(dictionary, father_terms)
    
    print("\nTop 10 mother-related terms:")
    for term, count in mother_top_terms:
        print(f"{term}: {count}")

    print("\nTop 10 father-related terms:")
    for term, count in father_top_terms:
        print(f"{term}: {count}")
    
    # Plot gender bias with top terms
    plot_gender_bias(mother_percentage, father_percentage, mother_top_terms, father_top_terms)
    print("Gender bias chart with top terms saved as 'gender_bias_pie_chart.png'")
    
    # Calculate and print coherence score
    coherence_model = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"\nCoherence Score: {coherence_score}")

if __name__ == "__main__":
    main()