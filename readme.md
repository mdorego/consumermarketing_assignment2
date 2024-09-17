# Gender Representation in Parenting Content Analysis

This project analyzes gender representation in parenting-related content by scraping Google search results, performing text extraction, and conducting various Natural Language Processing (NLP) analyses.

## Project Structure

The project consists of several Python scripts that work together to perform the analysis:

1. `url_scraping.py`: Scrapes Google search results for given keywords.
2. `text_extraction.py`: Extracts content from the scraped URLs.
3. `text_preprocessing.py`: Preprocesses the extracted text data.
4. `sentiment_and_context_analysis.py`: Performs sentiment analysis and context extraction.
5. `lda_and_gender_bias_analysis.py`: Conducts topic modeling and gender bias analysis.

## Installation

To run this project, you need Python 3.7+ and the following libraries:

```
pip install pandas requests beautifulsoup4 nltk textblob matplotlib seaborn gensim pyLDAvis wordcloud
```

You may also need to download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

Run the scripts in the following order:

1. `python url_scraping.py`
2. `python text_extraction.py`
3. `python text_preprocessing.py`
4. `python sentiment_and_context_analysis.py`
5. `python lda_and_gender_bias_analysis.py`

## Output

The scripts generate several output files:

- `google_search_results_20.csv`: Contains scraped URLs for each keyword.
- `google_search_results_with_extracted_content.csv`: Contains extracted content from the URLs.
- `preprocessed_data_for_nlp.csv`: Contains preprocessed text data.
- Various PNG files: Visualizations of sentiment distribution, word frequencies, and gender bias.
- `lda_visualization.html`: Interactive visualization of LDA results.

## Analysis Details

- **Sentiment Analysis**: Compares sentiment in mother-related vs. father-related contexts.
- **Word Frequency**: Identifies most common words in mother-related and father-related contexts.
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to identify main topics in the content.
- **Gender Bias Analysis**: Quantifies representation of mother-related vs. father-related terms.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
