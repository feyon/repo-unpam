import pandas as pd
import re

# 1. Load Data
# Data akan diambil dari kolom 'text'
df = pd.read_csv("mental_health.csv")
corpus = df['text'].tolist()

def build_inverted_index(documents):
    """
    Membangun Inverted Index dari koleksi dokumen (corpus).

    Args:
        documents (list): Daftar string dokumen (corpus).

    Returns:
        dict: Inverted Index, di mana kunci adalah istilah (term) dan nilainya
              adalah daftar ID dokumen (indeks baris) tempat istilah itu muncul.
    """
    inverted_index = {}
    
    # Iterasi melalui setiap dokumen dengan indeksnya
    for doc_id, document in enumerate(documents):
        # Preprocessing: Lowercasing dan tokenisasi sederhana (menghapus tanda baca)
        # Kami menggunakan regex untuk memisahkan kata-kata dan menghilangkan yang bukan huruf/angka
        tokens = re.findall(r'\b\w+\b', str(document).lower())
        
        # Ambil hanya istilah unik dalam dokumen ini (untuk menghemat ruang)
        # Jika tujuannya menghitung frekuensi (TF-IDF), langkah ini tidak dilakukan.
        unique_tokens = set(tokens)
        
        # Membangun indeks
        for term in unique_tokens:
            if term not in inverted_index:
                # Jika istilah baru, buat entri baru dengan doc_id
                inverted_index[term] = [doc_id]
            else:
                # Jika istilah sudah ada, tambahkan doc_id ke daftar
                inverted_index[term].append(doc_id)
                
    return inverted_index

def retrieve_documents(query_term, index):
    """
    Mengambil daftar ID dokumen yang berisi istilah kueri.
    
    Args:
        query_term (str): Istilah yang dicari (akan di-lowercase).
        index (dict): Inverted Index yang telah dibangun.

    Returns:
        list: Daftar ID dokumen yang relevan.
    """
    # Preprocessing kueri: lowercase
    normalized_query = str(query_term).lower()
    
    # Retrieval
    return index.get(normalized_query, [])

# 2. Bangun Inverted Index
inverted_index = build_inverted_index(corpus)

# 3. Demonstrasi Retrieval
query_1 = "suicidal"
doc_ids_1 = retrieve_documents(query_1, inverted_index)

query_2 = "mother"
doc_ids_2 = retrieve_documents(query_2, inverted_index)

# Tampilkan hasil pencarian pertama
print(f"\n--- Demonstrasi Inverted Index Retrieval ---")
print(f"Total dokumen dalam corpus: {len(corpus)}")
print(f"Total istilah unik (terms) dalam index: {len(inverted_index)}")

print(f"\n[QUERY 1] Mencari istilah: '{query_1}'")
print(f"Ditemukan di {len(doc_ids_1)} dokumen.")
print(f"ID Dokumen (Indeks Baris) yang relevan: {doc_ids_1[:10]}... (menampilkan 10 ID pertama)")

# Tampilkan hasil pencarian kedua
print(f"\n[QUERY 2] Mencari istilah: '{query_2}'")
print(f"Ditemukan di {len(doc_ids_2)} dokumen.")
print(f"ID Dokumen (Indeks Baris) yang relevan: {doc_ids_2[:10]}... (menampilkan 10 ID pertama)")