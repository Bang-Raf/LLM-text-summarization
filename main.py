#!/usr/bin/env python3
"""
Script utama untuk menjalankan sistem summarization berita
Menggunakan Gemma2 9B Sahabat AI
"""

import json
import sys
from data_processor import NewsDataProcessor
from summarizer import NewsSummarizer

def create_sample_data():
    """
    Membuat sample data untuk testing
    """
    sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True], [False, False, False], [False, False], [False, False], [False, False], [False, False], [False], [False, False]],
            "id": "1501893029-lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun",
            "paragraphs": [
                [["Jakarta", ",", "CNN", "Indonesia", "-", "-", "Dokter", "Ryan", "Thamrin", ",", "yang", "terkenal", "lewat", "acara", "Dokter", "Oz", "Indonesia", ",", "meninggal", "dunia", "pada", "Jumat", "(", "4", "/", "8", ")", "dini", "hari", "."], ["Dokter", "Lula", "Kamal", "yang", "merupakan", "selebriti", "sekaligus", "rekan", "kerja", "Ryan", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."]],
                [["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", ",", "termasuk", "menjadi", "pembawa", "acara", "Dokter", "Oz", "Indonesia", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "di", "Pekanbaru", ",", "Riau", "untuk", "menjalani", "istirahat", "."]],
                [["\"", "Setahu", "saya", "dia", "orangnya", "sehat", ",", "tapi", "tahun", "lalu", "saya", "dengar", "dia", "sakit", "."], ["(", "Karena", ")", "sakitnya", ",", "ia", "langsung", "pulang", "ke", "Pekanbaru", ",", "jadi", "kami", "yang", "mau", "jenguk", "juga", "susah", "."], ["Barangkali", "mau", "istirahat", ",", "ya", "betul", "juga", ",", "kalau", "di", "Jakarta", "susah", "isirahatnya", ",", "\"", "kata", "Lula", "kepada", "CNNIndonesia.com", ",", "Jumat", "(", "4", "/", "8", ")", "."]],
                [["Lula", "yang", "mengenal", "Ryan", "sejak", "sebelum", "aktif", "berkarier", "di", "televisi", "mengaku", "belum", "sempat", "membesuk", "Ryan", "lantaran", "lokasi", "yang", "jauh", "."], ["Dia", "juga", "tak", "tahu", "penyakit", "apa", "yang", "diderita", "Ryan", "."]],
                [["\"", "Itu", "saya", "enggak", "tahu", ",", "belum", "sempat", "jenguk", "dan", "enggak", "selamanya", "bisa", "dijenguk", "juga", "."], ["Enggak", "tahu", "berat", "sekali", "apa", "bagaimana", ",", "\"", "tutur", "Ryan", "."]],
                [["Walau", "sudah", "setahun", "menderita", "sakit", ",", "Lula", "tak", "mengetahui", "apa", "penyebab", "pasti", "kematian", "Dr", "Oz", "Indonesia", "itu", "."], ["Meski", "demikian", ",", "ia", "mendengar", "beberapa", "kabar", "yang", "menyebut", "bahwa", "penyebab", "Ryan", "meninggal", "adalah", "karena", "jatuh", "di", "kamar", "mandi", "."]],
                [["\u201c", "Saya", "tidak", "tahu", ",", "barangkali", "penyakit", "yang", "dulu", "sama", "yang", "sekarang", "berbeda", ",", "atau", "penyebab", "kematiannya", "beda", "dari", "penyakit", "sebelumnya", "."], ["Kita", "kan", "enggak", "bisa", "mengambil", "kesimpulan", ",", "\"", "kata", "Lula", "."]],
                [["Ryan", "Thamrin", "terkenal", "sebagai", "dokter", "yang", "rutin", "membagikan", "tips", "dan", "informasi", "kesehatan", "lewat", "tayangan", "Dokter", "Oz", "Indonesia", "."]],
                [["Ryan", "menempuh", "Pendidikan", "Dokter", "pada", "tahun", "2002", "di", "Fakultas", "Kedokteran", "Universitas", "Gadjah", "Mada", "."], ["Dia", "kemudian", "melanjutkan", "pendidikan", "Klinis", "Kesehatan", "Reproduksi", "dan", "Penyakit", "Menular", "Seksual", "di", "Mahachulalongkornrajavidyalaya", "University", ",", "Bangkok", ",", "Thailand", "pada", "2004", "."]]
            ],
            "source": "cnn indonesia",
            "source_url": "https://www.cnnindonesia.com/hiburan/20170804120703-234-232443/lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun-lalu/",
            "summary": [
                [["Dokter", "Lula", "Kamal", "yang", "merupakan", "selebriti", "sekaligus", "rekan", "kerja", "Ryan", "Thamrin", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."], ["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", ",", "termasuk", "menjadi", "pembawa", "acara", "Dokter", "Oz", "Indonesia", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "di", "Pekanbaru", ",", "Riau", "untuk", "menjalani", "istirahat", "."]]
            ]
        }
    ]
    
    return sample_data

def main():
    """
    Fungsi utama untuk menjalankan sistem summarization
    """
    print("=== Sistem Summarization Berita dengan Gemma2 9B Sahabat AI ===\n")
    
    # 1. Inisialisasi data processor
    print("1. Memproses data...")
    processor = NewsDataProcessor()
    
    # Gunakan sample data untuk demo
    sample_data = create_sample_data()
    processed_data = processor.process_dataset(sample_data)
    
    print(f"Berhasil memproses {len(processed_data)} item berita")
    
    # Tampilkan sample data yang sudah diproses
    if processed_data:
        sample_item = processed_data[0]
        print(f"\nSample berita:")
        print(f"ID: {sample_item['id']}")
        print(f"Kategori: {sample_item['category']}")
        print(f"Teks lengkap: {sample_item['full_text'][:200]}...")
        print(f"Summary referensi: {sample_item['gold_summary']}")
    
    # 2. Inisialisasi summarizer
    print("\n2. Memuat model Gemma2 9B Sahabat AI...")
    summarizer = NewsSummarizer()
    
    try:
        # Load model (gunakan CPU jika GPU tidak tersedia)
        summarizer.load_model(device_map="auto")
        
        # 3. Test summarization untuk satu item
        print("\n3. Testing summarization...")
        if processed_data:
            test_item = processed_data[0]
            print(f"Memproses berita: {test_item['id']}")
            
            generated_summary = summarizer.generate_summary(test_item['full_text'])
            
            print(f"\nTeks asli: {test_item['full_text'][:300]}...")
            print(f"\nSummary yang dihasilkan: {generated_summary}")
            print(f"\nSummary referensi: {test_item['gold_summary']}")
            
            # Evaluasi summary
            if generated_summary and test_item['gold_summary']:
                metrics = summarizer.evaluate_summary(generated_summary, test_item['gold_summary'])
                print(f"\nMetrik evaluasi:")
                print(f"- Compression ratio: {metrics['compression_ratio']:.2f}")
                print(f"- Word overlap: {metrics['word_overlap']:.2f}")
                print(f"- Panjang summary: {metrics['generated_length']} kata")
                print(f"- Panjang referensi: {metrics['gold_length']} kata")
        
        # 4. Batch processing (opsional)
        print("\n4. Batch processing...")
        print("Apakah Anda ingin melakukan batch processing untuk semua data? (y/n): ", end="")
        
        # Untuk demo, kita skip batch processing
        print("n (skipped for demo)")
        
        # Jika ingin batch processing, uncomment baris berikut:
        # results = summarizer.batch_summarize(processed_data, "output_summaries.jsonl")
        # print(f"Berhasil memproses {len(results)} item")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTips:")
        print("1. Pastikan transformers==4.45.0 terinstall")
        print("2. Pastikan ada cukup memory untuk model 9B")
        print("3. Gunakan GPU jika tersedia untuk performa lebih baik")
        return
    
    print("\n=== Selesai ===")

if __name__ == "__main__":
    main()