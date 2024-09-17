import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files
df1 = pd.read_csv('parenting-keywords-avgMonthlySearches.csv')
df2 = pd.read_csv('new-parent-keywords-with-categories.csv')

# Convert keywords to lowercase and replace hyphens and slashes with spaces in both dataframes
def clean_keyword(keyword):
    return keyword.lower().replace('-', ' ').replace('/', ' ')

df1['keyword'] = df1['keyword'].apply(clean_keyword)
df2['Keyword'] = df2['Keyword'].apply(clean_keyword)

# Merge the dataframes
merged_df = pd.merge(df1, df2, left_on='keyword', right_on='Keyword', how='outer')

# Clean up the merged dataframe
merged_df['Avg. Monthly Search'] = merged_df['Avg. Monthly Search'].fillna('<10')
merged_df['Category'] = merged_df['Category'].fillna('Uncategorized')

# Convert search volume to numeric
def convert_search_volume(value):
    if value == '<10':
        return 5  # Midpoint of 0-10
    elif '-' in value:
        low, high = map(lambda x: int(x.replace('K', '000')), value.split('-'))
        return (low + high) / 2
    else:
        return int(value.replace('K', '000'))

merged_df['Search Volume'] = merged_df['Avg. Monthly Search'].apply(convert_search_volume)

# Identify uncategorized keywords
uncategorized_keywords = merged_df[merged_df['Category'] == 'Uncategorized']

print("Uncategorized Keywords:")
for index, row in uncategorized_keywords.iterrows():
    print(f"Keyword: {row['keyword']}, Avg. Monthly Search: {row['Avg. Monthly Search']}")

print(f"\nTotal uncategorized keywords: {len(uncategorized_keywords)}")

# Visualization 1: Bar plot of top 10 search volumes
plt.figure(figsize=(12, 6))
top_10 = merged_df.nlargest(10, 'Search Volume')
sns.barplot(x='Search Volume', y='keyword', data=top_10)
plt.title('Top 10 Parenting Keywords by Average Monthly Searches')
plt.xlabel('Average Monthly Searches')
plt.ylabel('Keyword')
plt.tight_layout()
plt.savefig('top_10_keywords.png')
plt.close()

# Visualization 2: Pie chart of categories
plt.figure(figsize=(10, 10))
category_counts = merged_df['Category'].value_counts()
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Parenting Keywords by Category')
plt.axis('equal')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.close()

# Visualization 3: Box plot of search volumes by category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Search Volume', data=merged_df)
plt.title('Distribution of Search Volumes by Category')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Monthly Searches')
plt.tight_layout()
plt.savefig('search_volume_by_category.png')
plt.close()

# Visualization 4: Heatmap of search volumes by category
pivot_df = merged_df.pivot_table(values='Search Volume', index='Category', columns='Avg. Monthly Search', aggfunc='count', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Heatmap of Search Volume Ranges by Category')
plt.xlabel('Search Volume Range')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('search_volume_heatmap.png')
plt.close()

# Create cleaned_input_keywords.csv
cleaned_df = merged_df[merged_df['Avg. Monthly Search'] != '<10'].copy()
cleaned_df = cleaned_df.sort_values('Search Volume', ascending=False)
cleaned_df = cleaned_df[['Keyword', 'Category', 'Avg. Monthly Search']]
cleaned_df = cleaned_df.rename(columns={'Avg. Monthly Search': 'Average Monthly Searches'})
cleaned_df.to_csv('cleaned_input_keywords.csv', index=False)

print("Analysis complete. Visualizations have been saved as PNG files.")
print(f"Created new CSV file: cleaned_input_keywords.csv")
print(f"Total keywords in cleaned file: {len(cleaned_df)}")
print("\nTop 5 keywords in cleaned file:")
print(cleaned_df.head().to_string(index=False))