import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random

def extract_content_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ""

        # Extract main content (this might need adjustment based on the website structure)
        main_content = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            main_content += tag.get_text() + "\n"

        # Extract meta description
        meta_description = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_description = meta_tag.get('content', '')

        return {
            'title': title.strip(),
            'main_content': main_content.strip(),
            'meta_description': meta_description.strip()
        }
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return {
            'title': '',
            'main_content': '',
            'meta_description': ''
        }

def main():
    # Read the CSV file
    df = pd.read_csv('google_search_results_20.csv')
    
    # Create new columns for the extracted content
    df['titles'] = ''
    df['main_contents'] = ''
    df['meta_descriptions'] = ''
    
    for index, row in df.iterrows():
        urls = row['URLs'].split(', ')
        titles = []
        main_contents = []
        meta_descriptions = []
        
        for url in urls:
            print(f"Processing URL: {url}")
            content = extract_content_from_url(url)
            titles.append(content['title'])
            main_contents.append(content['main_content'])
            meta_descriptions.append(content['meta_description'])
            
            # Add a random delay between requests
            time.sleep(random.uniform(1, 3))
        
        # Join all extracted content for this keyword
        df.at[index, 'titles'] = '|||'.join(titles)
        df.at[index, 'main_contents'] = '|||'.join(main_contents)
        df.at[index, 'meta_descriptions'] = '|||'.join(meta_descriptions)
        
        print(f"Completed processing for keyword: {row['Keyword']}")
    
    # Save the results
    df.to_csv('google_search_results_with_extracted_content.csv', index=False)
    print("Processing complete. Results saved to 'google_search_results_with_extracted_content.csv'")

if __name__ == "__main__":
    main()