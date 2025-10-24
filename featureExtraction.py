import re
import nltk
import string
import numpy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

sentences = [
"Prakiraan cuaca maritim penting untuk pelayaran. ",
"Meteorologi maritim mempelajari cuaca di laut. ",
"Data maritim akurat mencegah kecelakaan kapal. ",
"Informasi maritim memandu aktivitas lepas pantai. ",
"Kondisi cuaca maritim sangat dinamis dan berubah cepat. "
]

def preprocess_text(text):
    text = text.lower()
    text = text.translate(text.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    pos_tagged_tokens1 = pos_tag(stemmed_tokens)
    pos_tagged_tokens2 = pos_tag(lemmatized_tokens)

    return {
        "original_text": text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "pos_tagged_tokens1": pos_tagged_tokens1,
        "pos_tagged_tokens2": pos_tagged_tokens2,
    }

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
nltk.download('punkt_tab')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

import nltk
nltk.download('punkt_tab')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(preprocessed_sentences)

print("Vocabulary:\n", vectorizer.get_feature_names_out())
print('\nVector vocabulary:\n',vectorizer.vocabulary_)
print("\nBag of Words:\n", X_bow.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_sentences)

print("Vocabulary:\n", tfidf_vectorizer.get_feature_names_out())
print('\nVector vocabulary:\n',tfidf_vectorizer.vocabulary_)
print("\nTF-IDF:\n", X_tfidf.toarray())

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = "maritim"
query_vector = tfidf_vectorizer.transform([query])
cosine_similarities = cosine_similarity(query_vector, X_tfidf)
similarity_scores = cosine_similarities[0]
sorted_indices = np.argsort(similarity_scores)[::-1]
top_5_indices = sorted_indices[:5]
print("Top 5 sentences similar to 'maritim':")
for index in top_5_indices:
    print(f"Document {index + 1}: {sentences[index]} (Similarity score: {similarity_scores[index]})")