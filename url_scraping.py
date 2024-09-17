import csv
import time
import random
import requests
from bs4 import BeautifulSoup

def get_search_results(query, num_results=20):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    results = []
    
    for start in range(0, num_results, 10):
        url = f"https://www.google.com/search?q={query}&start={start}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for g in soup.find_all('div', class_='yuRUbf'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                results.append(link)
        
        if len(results) >= num_results:
            break
        
        # Add a delay between page requests
        time.sleep(random.uniform(1, 3))
    
    return results[:num_results]

def main():
    input_file = 'cleaned_input_keywords.csv'
    output_file = 'google_search_results_20.csv'
    
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['Keyword', 'Category', 'Average Monthly Searches', 'URLs']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            keyword = row['Keyword']
            print(f"Searching for: {keyword}")
            
            urls = get_search_results(keyword, num_results=20)
            row['URLs'] = ', '.join(urls)
            writer.writerow(row)
            
            # Add a delay between keyword searches
            time.sleep(random.uniform(2, 5))
    
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()