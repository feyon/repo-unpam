import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def feature_extraction_and_display(input_filename='mental_health_dataset.json', output_filename='mental_health_features.csv'):
    """
    Loads, preprocesses, and performs TF-IDF feature extraction on a CSV file.
    It prints the results at each major step for review.
    """
   try:
        # 1. Load the dataset
        # --- MODIFIED LINE ---
        df = pd.read_json(input_filename, lines=True)
        print(f"--- 1. Successfully loaded '{input_filename}'. ---")
        print("Original DataFrame head:")
        print(df.head())
        print("-" * 50)

        # --- 2. Text Preprocessing ---
        STOPWORDS = set([
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
            'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could',
            "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
            'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
            "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into',
            'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not',
            'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan',
            "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the',
            'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
            "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll",
            "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
            "who's", 'whom', 'why', "why's", 'with', 'won', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
            'your', 'yours', 'yourself', 'yourselves'
        ])

        def preprocess_text_simple(text):
            if not isinstance(text, str): return ""
            text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#', '', text, flags=re.MULTILINE)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            tokens = text.split()
            filtered_tokens = [word for word in tokens if word not in STOPWORDS]
            return " ".join(filtered_tokens)

        print("--- 2. Starting text preprocessing... ---")
        df.dropna(subset=['text'], inplace=True)
        df['processed_text'] = df['text'].apply(preprocess_text_simple)
        
        # --- Display Preprocessing Result ---
        print("\nDisplaying original text vs. processed text:")
        print(df[['text', 'processed_text']].head())
        print("-" * 50)

        # --- 3. Feature Extraction using TfidfVectorizer ---
        print("--- 3. Starting feature extraction with TF-IDF... ---")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

        # --- Display Vocabulary ---
        feature_names = tfidf_vectorizer.get_feature_names_out()
        print(f"\nVocabulary size: {len(feature_names)} features")
        print("Sample of 20 features (vocabulary):")
        print(feature_names[:20])
        print("-" * 50)
        
        # --- 4. Create Final DataFrame and Display It ---
        print("--- 4. Creating final DataFrame with TF-IDF features... ---")
        df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
        
        df.reset_index(drop=True, inplace=True)
        df_tfidf.reset_index(drop=True, inplace=True)
        
        df_final = pd.concat([df['label'], df_tfidf], axis=1)
        df_final.dropna(subset=['label'], inplace=True)
        
        # --- Display Final Result ---
        print("\nDisplaying the final DataFrame with features and labels (first 5 rows):")
        # Using .iloc to show the first few feature columns as it can be very wide
        print(df_final.iloc[:, :8].head())
        print("-" * 50)

        # 5. Save the results to a new CSV file
        df_final.to_csv(output_filename, index=False)
        print(f"--- 5. Successfully saved results to '{output_filename}' ---")

    except FileNotFoundError:
        print(f"Error: '{input_filename}' not found. Please ensure the file is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the pipeline ---
if __name__ == "__main__":
    feature_extraction_and_display() 