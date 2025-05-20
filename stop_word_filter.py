import nltk
import pandas as pd
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter CSV file based on most common words")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="filtered_csv_file.csv", help="Path to output filtered CSV file")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes in the dataset")
    parser.add_argument("--top_n_multiplier", type=int, default=2, help="Multiplier for top words threshold (top_n = num_classes * multiplier)")
    return parser.parse_args()

def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')

def count_words(df):
    """Count word occurrences in the dataframe text columns"""
    stop_words = set(stopwords.words('english'))
    word_counts = {}
    
    print("Counting word occurrences...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Assuming text columns are at positions 1, 2, 3
        for text in [row[1], row[2], row[3]]:
            word_tokens = word_tokenize(text)
            for word in word_tokens:
                word_lower = word.lower()
                if word_lower not in stop_words and word_lower.isalpha():
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    return word_counts

def filter_by_top_words(df, top_n_words):
    """Filter rows in dataframe based on presence of top N words"""
    filtered_data = []
    
    print("Filtering texts based on top words...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        label = row[0].split('/')[-2]
        row_texts = [row[1], row[2], row[3]]  # Text columns
        
        for sentence in row_texts:
            # Check if any of the top N words are in the sentence
            if any(word in sentence.lower() for word in top_n_words):
                filtered_data.append([label, sentence])
    
    return filtered_data

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure necessary NLTK resources are downloaded
    download_nltk_resources()
    
    # Calculate top N threshold
    top_n_threshold = args.num_classes * args.top_n_multiplier
    
    # Read input CSV file
    print(f"Reading CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Count word occurrences
    word_counts = count_words(df)
    
    # Get top N words
    top_n_words = [word for word, count in sorted(
        word_counts.items(), key=lambda item: item[1], reverse=True
    )[:top_n_threshold]]
    
    print(f"Top {top_n_threshold} words: {', '.join(top_n_words)}")
    
    # Filter dataframe by top words
    filtered_data = filter_by_top_words(df, top_n_words)
    
    # Create new dataframe with filtered data
    filtered_df = pd.DataFrame(filtered_data, columns=['Label', 'GeneratedTexts'])
    
    # Save filtered dataframe to CSV
    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Filtered data saved to: {args.output_csv}")
    print(f"Total rows in filtered data: {len(filtered_df)}")
    
    # Print word counts for reference
    print("\nWord frequency counts:")
    for word, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()