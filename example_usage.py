#!/usr/bin/env python3
"""
Contoh penggunaan sistem summarization berita
"""

import json
from data_processor import NewsDataProcessor
from summarizer import NewsSummarizer
from utils import create_sample_dataset, evaluate_summaries, print_evaluation_results

def example_1_basic_usage():
    """
    Contoh penggunaan dasar
    """
    print("=== Contoh 1: Penggunaan Dasar ===\n")
    
    # 1. Buat sample dataset
    sample_data = create_sample_dataset("temp_sample.jsonl")
    
    # 2. Proses data
    processor = NewsDataProcessor()
    processed_data = processor.process_dataset(sample_data)
    
    print(f"Berhasil memproses {len(processed_data)} item")
    
    # 3. Load model dan generate summary
    summarizer = NewsSummarizer()
    summarizer.load_model()
    
    # 4. Generate summary untuk item pertama
    if processed_data:
        item = processed_data[0]
        print(f"\nMemproses: {item['id']}")
        print(f"Kategori: {item['category']}")
        
        summary = summarizer.generate_summary(item['full_text'])
        
        print(f"\nTeks asli: {item['full_text'][:150]}...")
        print(f"Summary yang dihasilkan: {summary}")
        print(f"Summary referensi: {item['gold_summary']}")

def example_2_batch_processing():
    """
    Contoh batch processing
    """
    print("\n=== Contoh 2: Batch Processing ===\n")
    
    # 1. Buat dataset yang lebih besar
    large_sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True]],
            "id": "batch-1",
            "paragraphs": [
                [["Jakarta", ",", "CNN", "Indonesia", "-", "Dokter", "Ryan", "Thamrin", "meninggal", "dunia", "."], ["Dokter", "Lula", "Kamal", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."]],
                [["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "."]]
            ],
            "source": "cnn indonesia",
            "source_url": "https://example.com",
            "summary": [
                [["Dokter", "Lula", "Kamal", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."], ["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", "."]]
            ]
        },
        {
            "category": "teknologi",
            "gold_labels": [[False, True], [True, True]],
            "id": "batch-2",
            "paragraphs": [
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "."], ["Kedua", "model", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."]],
                [["Mereka", "adalah", "Asus", "ZenFone", "4", "Selfie", "Pro", "dan", "ZenFone", "4", "Selfie", "."], ["Kedua", "model", "diracik", "sebagai", "jawaban", "atas", "kekurangan", "kompetitor", "."]]
            ],
            "source": "dailysocial.id",
            "source_url": "https://example.com",
            "summary": [
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "."], ["Kedua", "model", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."], ["Kedua", "model", "diracik", "sebagai", "jawaban", "atas", "kekurangan", "kompetitor", "."]]
            ]
        },
        {
            "category": "olahraga",
            "gold_labels": [[True, True], [False, True]],
            "id": "batch-3",
            "paragraphs": [
                [["Timnas", "Indonesia", "berhasil", "mengalahkan", "Timnas", "Malaysia", "dengan", "skor", "3-0", "."], ["Pertandingan", "berlangsung", "di", "Stadion", "Gelora", "Bung", "Karno", "Jakarta", "."]],
                [["Tiga", "gol", "dicetak", "oleh", "pemain", "Indonesia", "dalam", "babak", "pertama", "."], ["Pelatih", "Timnas", "Indonesia", "sangat", "puas", "dengan", "performa", "anak", "asuhnya", "."]]
            ],
            "source": "bola.com",
            "source_url": "https://example.com",
            "summary": [
                [["Timnas", "Indonesia", "berhasil", "mengalahkan", "Timnas", "Malaysia", "dengan", "skor", "3-0", "."], ["Pertandingan", "berlangsung", "di", "Stadion", "Gelora", "Bung", "Karno", "Jakarta", "."], ["Pelatih", "Timnas", "Indonesia", "sangat", "puas", "dengan", "performa", "anak", "asuhnya", "."]]
            ]
        }
    ]
    
    # 2. Proses data
    processor = NewsDataProcessor()
    processed_data = processor.process_dataset(large_sample_data)
    
    # 3. Load model dan batch processing
    summarizer = NewsSummarizer()
    summarizer.load_model()
    
    # 4. Batch summarization
    results = summarizer.batch_summarize(processed_data, "batch_output.jsonl")
    
    print(f"Berhasil memproses {len(results)} item")
    
    # 5. Tampilkan hasil
    for i, result in enumerate(results):
        print(f"\nItem {i+1}: {result['id']}")
        print(f"Kategori: {result['category']}")
        print(f"Summary: {result['generated_summary']}")
        print(f"Referensi: {result['gold_summary']}")
    
    # 6. Evaluasi
    metrics = evaluate_summaries(results)
    print_evaluation_results(metrics)

def example_3_custom_prompt():
    """
    Contoh dengan custom prompt
    """
    print("\n=== Contoh 3: Custom Prompt ===\n")
    
    # Buat custom summarizer dengan prompt yang berbeda
    class CustomSummarizer(NewsSummarizer):
        def create_summarization_prompt(self, text: str, max_length: int = 2000) -> str:
            """Custom prompt untuk summarization"""
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            prompt = f"""Ringkaslah berita berikut dalam 1-2 kalimat yang informatif dan mudah dipahami. Fokus pada informasi utama dan fakta penting.

Berita:
{text}

Ringkasan singkat:"""
            
            return prompt
    
    # Gunakan custom summarizer
    custom_summarizer = CustomSummarizer()
    custom_summarizer.load_model()
    
    # Sample text
    sample_text = """Jakarta, CNN Indonesia - - Dokter Ryan Thamrin, yang terkenal lewat acara Dokter Oz Indonesia, meninggal dunia pada Jumat (4/8) dini hari. Dokter Lula Kamal yang merupakan selebriti sekaligus rekan kerja Ryan menyebut kawannya itu sudah sakit sejak setahun yang lalu. Lula menuturkan, sakit itu membuat Ryan mesti vakum dari semua kegiatannya, termasuk menjadi pembawa acara Dokter Oz Indonesia."""
    
    summary = custom_summarizer.generate_summary(sample_text)
    
    print(f"Teks asli: {sample_text}")
    print(f"Summary dengan custom prompt: {summary}")

def example_4_error_handling():
    """
    Contoh penanganan error
    """
    print("\n=== Contoh 4: Error Handling ===\n")
    
    try:
        # Coba generate summary tanpa load model
        summarizer = NewsSummarizer()
        summary = summarizer.generate_summary("Test text")
    except ValueError as e:
        print(f"Error yang diharapkan: {e}")
        print("Solusi: Panggil load_model() terlebih dahulu")
    
    try:
        # Load model dan test dengan text kosong
        summarizer = NewsSummarizer()
        summarizer.load_model()
        
        summary = summarizer.generate_summary("")
        print(f"Summary untuk text kosong: '{summary}'")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Jalankan semua contoh
    """
    print("Contoh Penggunaan Sistem Summarization Berita\n")
    print("=" * 50)
    
    # Jalankan contoh-contoh
    example_1_basic_usage()
    example_2_batch_processing()
    example_3_custom_prompt()
    example_4_error_handling()
    
    print("\n" + "=" * 50)
    print("Semua contoh selesai!")

if __name__ == "__main__":
    main()