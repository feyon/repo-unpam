import pandas as pd
import re

# Mengambial Data
file_name = "mental_health_dataset.json"

try:
    # Membaca file JSON Lines (satu objek JSON per baris)
    df = pd.read_json(file_name, lines=True)

    # Melakukan eksplorasi data dasar
    print("--- Informasi Data (df.info()) ---")
    df.info()
    print("\n--- Awal Data (df.head()) ---")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # Memastikan kolom 'text' ada dalam DataFrame
    if 'text' not in df.columns:
        raise KeyError("Kolom 'text' tidak ditemukan dalam file JSON. Kolom yang tersedia: " + ", ".join(df.columns))

    # Data akan diambil dari kolom 'text'
    # Melakukan Konversi ke list, pastikan semua data adalah string
    corpus = df['text'].astype(str).tolist()

    # 1. Fungsi untuk membangun Inverted Index dan Retrieval
    def build_inverted_index(documents):
        inverted_index = {}
        
        # Lakukan Iterasi melalui setiap dokumen dengan indeksnya
        for doc_id, document in enumerate(documents):
            # Preprocessing: Lowercasing dan tokenisasi sederhana
            tokens = re.findall(r'\b\w+\b', str(document).lower())
            
            # Ambil hanya istilah unik dalam dokumen ini
            unique_tokens = set(tokens)
            
            # Membangun indeks
            for term in unique_tokens:
                if term not in inverted_index:
                    inverted_index[term] = [doc_id]
                else:
                    inverted_index[term].append(doc_id)
                    
        return inverted_index

    def retrieve_documents(query_term, index):
        """
        Mengambil daftar ID dokumen yang berisi istilah Query.
        Args:
            query_term (str): Istilah yang dicari.
            index (dict): Inverted index.   
        Returns:
            list: Daftar ID dokumen yang mengandung istilah Query.
        """
        # Preprocessing Query untuk  lowercase
        normalized_query = str(query_term).lower()
        
        # Information Retrieval
        return index.get(normalized_query, [])

    # 2. Bangun Inverted Index
    inverted_index = build_inverted_index(corpus)

    # 3. Information Retrieval
    # Query yang ada sudah sangat relevan dengan konteks data
    query_1 = "suicidal"
    doc_ids_1 = retrieve_documents(query_1, inverted_index)

    query_2 = "mother"
    doc_ids_2 = retrieve_documents(query_2, inverted_index)
    
    query_3 = "anxiety"
    doc_ids_3 = retrieve_documents(query_3, inverted_index)


    # Tampilkan hasil pencarian
    print(f"\n--- Mulai Inverted Index Retrieval ---")
    print(f"Total dokumen dalam corpus: {len(corpus)}")
    print(f"Total istilah unik (terms) dalam index: {len(inverted_index)}")

    print(f"\n[QUERY 1] Mencari istilah: '{query_1}'")
    print(f"Ditemukan di {len(doc_ids_1)} dokumen.")
    print(f"ID Dokumen (Indeks Baris) yang relevan: {doc_ids_1[:20]}... (menampilkan 10 ID pertama)")

    print(f"\n[QUERY 2] Mencari istilah: '{query_2}'")
    print(f"Ditemukan di {len(doc_ids_2)} dokumen.")
    print(f"ID Dokumen (Indeks Baris) yang relevan: {doc_ids_2[:20]}... (menampilkan 10 ID pertama)")
    
    print(f"\n[QUERY 3] Mencari istilah: '{query_3}'")
    print(f"Ditemukan di {len(doc_ids_3)} dokumen.")
    print(f"ID Dokumen (Indeks Baris) yang relevan: {doc_ids_3[:20]}... (menampilkan 10 ID pertama)")

    # Tampilkan contoh data yang diambil untuk memverifikasi
    if doc_ids_1:
        print(f"\n--- Contoh Teks untuk Query '{query_1}' (Dokumen {doc_ids_1[0]}) ---")
        print(df.iloc[doc_ids_1[0]]['text'])
        
    if doc_ids_2:
        print(f"\n--- Contoh Teks untuk Query '{query_2}' (Dokokumen {doc_ids_2[0]}) ---")
        print(df.iloc[doc_ids_2[0]]['text'])

except FileNotFoundError:
    print(f"Error: File '{file_name}' tidak ditemukan.")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Terjadi error saat memproses file JSON: {e}")