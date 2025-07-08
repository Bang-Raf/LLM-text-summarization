#!/usr/bin/env python3
"""
Script testing untuk sistem summarization berita
Dapat dijalankan tanpa memuat model untuk testing komponen lain
"""

import json
import sys
from data_processor import NewsDataProcessor
from utils import create_sample_dataset, evaluate_summaries, print_evaluation_results, calculate_rouge_scores
from config import get_config

def test_data_processor():
    """
    Test untuk data processor
    """
    print("=== Testing Data Processor ===\n")
    
    # 1. Test sample data creation
    print("1. Testing sample data creation...")
    sample_data = create_sample_dataset("test_sample.jsonl")
    print(f"✓ Berhasil membuat {len(sample_data)} sample data")
    
    # 2. Test data processing
    print("\n2. Testing data processing...")
    processor = NewsDataProcessor()
    processed_data = processor.process_dataset(sample_data)
    print(f"✓ Berhasil memproses {len(processed_data)} item")
    
    # 3. Test text extraction
    if processed_data:
        item = processed_data[0]
        print(f"\n3. Testing text extraction...")
        print(f"ID: {item['id']}")
        print(f"Category: {item['category']}")
        print(f"Full text length: {len(item['full_text'])} characters")
        print(f"Gold summary: {item['gold_summary']}")
        print(f"Original summary: {item['original_summary']}")
        print("✓ Text extraction berhasil")
    
    return processed_data

def test_utils():
    """
    Test untuk utility functions
    """
    print("\n=== Testing Utils ===\n")
    
    # 1. Test ROUGE calculation
    print("1. Testing ROUGE calculation...")
    summary1 = "Dokter Ryan Thamrin meninggal dunia pada Jumat dini hari."
    summary2 = "Dokter Ryan Thamrin, pembawa acara Dokter Oz Indonesia, meninggal dunia."
    
    rouge_scores = calculate_rouge_scores(summary1, summary2)
    print(f"ROUGE-1: {rouge_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']:.4f}")
    print("✓ ROUGE calculation berhasil")
    
    # 2. Test evaluation
    print("\n2. Testing evaluation...")
    test_results = [
        {
            'id': 'test-1',
            'generated_summary': summary1,
            'gold_summary': summary2
        },
        {
            'id': 'test-2', 
            'generated_summary': "Asus memperkenalkan ZenFone baru.",
            'gold_summary': "Asus meluncurkan smartphone ZenFone terbaru."
        }
    ]
    
    metrics = evaluate_summaries(test_results)
    print_evaluation_results(metrics)
    print("✓ Evaluation berhasil")

def test_config():
    """
    Test untuk konfigurasi
    """
    print("\n=== Testing Config ===\n")
    
    # 1. Test default config
    print("1. Testing default config...")
    config = get_config()
    print(f"Model ID: {config.MODEL_ID}")
    print(f"Max input length: {config.MAX_INPUT_LENGTH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("✓ Default config berhasil")
    
    # 2. Test prompt templates
    print("\n2. Testing prompt templates...")
    default_prompt = config.get_prompt_template("default")
    short_prompt = config.get_prompt_template("short")
    detailed_prompt = config.get_prompt_template("detailed")
    
    print(f"Default prompt length: {len(default_prompt)} characters")
    print(f"Short prompt length: {len(short_prompt)} characters")
    print(f"Detailed prompt length: {len(detailed_prompt)} characters")
    print("✓ Prompt templates berhasil")
    
    # 3. Test output directory
    print("\n3. Testing output directory...")
    output_path = config.get_output_path("test_output.jsonl")
    print(f"Output path: {output_path}")
    print("✓ Output directory berhasil")

def test_sample_data_processing():
    """
    Test dengan sample data yang diberikan user
    """
    print("\n=== Testing Sample Data Processing ===\n")
    
    # Sample data dari user
    user_sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True], [False, False, False], [False, False], [False, False], [False, False], [False, False], [False], [False, False]],
            "id": "1501893029-lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun",
            "paragraphs": [
                [["Jakarta", ",", "CNN", "Indonesia", "-", "-", "Dokter", "Ryan", "Thamrin", ",", "yang", "terkenal", "lewat", "acara", "Dokter", "Oz", "Indonesia", ",", "meninggal", "dunia", "pada", "Jumat", "(", "4", "/", "8", ")", "dini", "hari", "."], ["Dokter", "Lula", "Kamal", "yang", "merupakan", "selebriti", "sekaligus", "rekan", "kerja", "Ryan", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "di", "Pekanbaru", ",", "Riau", "untuk", "menjalani", "istirahat", "."]],
                [["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", ",", "termasuk", "menjadi", "pembawa", "acara", "Dokter", "Oz", "Indonesia", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "di", "Pekanbaru", ",", "Riau", "untuk", "menjalani", "istirahat", "."]],
                [["\"", "Setahu", "saya", "dia", "orangnya", "sehat", ",", "tapi", "tahun", "lalu", "saya", "dengar", "dia", "sakit", "."], ["(", "Karena", ")", "sakitnya", ",", "ia", "langsung", "pulang", "ke", "Pekanbaru", ",", "jadi", "kami", "yang", "mau", "jenguk", "juga", "susah", "."], ["Barangkali", "mau", "istirahat", ",", "ya", "betul", "juga", ",", "kalau", "di", "Jakarta", "susah", "isirahatnya", ",", "\"", "kata", "Lula", "kepada", "CNNIndonesia.com", ",", "Jumat", "(", "4", "/", "8", ")", "."]],
                [["Lula", "yang", "mengenal", "Ryan", "sejak", "sebelum", "aktif", "berkarier", "di", "televisi", "mengaku", "belum", "sempat", "membesuk", "Ryan", "lantaran", "lokasi", "yang", "jauh", "."], ["Dia", "juga", "tak", "tahu", "penyakit", "apa", "yang", "diderita", "Ryan", "."], ["\"", "Itu", "saya", "enggak", "tahu", ",", "belum", "sempat", "jenguk", "dan", "enggak", "selamanya", "bisa", "dijenguk", "juga", "."], ["Enggak", "tahu", "berat", "sekali", "apa", "bagaimana", ",", "\"", "tutur", "Ryan", "."], ["Walau", "sudah", "setahun", "menderita", "sakit", ",", "Lula", "tak", "mengetahui", "apa", "penyebab", "pasti", "kematian", "Dr", "Oz", "Indonesia", "itu", "."], ["Meski", "demikian", ",", "ia", "mendengar", "beberapa", "kabar", "yang", "menyebut", "bahwa", "penyebab", "Ryan", "meninggal", "adalah", "karena", "jatuh", "di", "kamar", "mandi", "."], ["\u201c", "Saya", "tidak", "tahu", ",", "barangkali", "penyakit", "yang", "dulu", "sama", "yang", "sekarang", "berbeda", ",", "atau", "penyebab", "kematiannya", "beda", "dari", "penyakit", "sebelumnya", "."], ["Kita", "kan", "enggak", "bisa", "mengambil", "kesimpulan", ",", "\"", "kata", "Lula", "."], ["Ryan", "Thamrin", "terkenal", "sebagai", "dokter", "yang", "rutin", "membagikan", "tips", "dan", "informasi", "kesehatan", "lewat", "tayangan", "Dokter", "Oz", "Indonesia", "."], ["Ryan", "menempuh", "Pendidikan", "Dokter", "pada", "tahun", "2002", "di", "Fakultas", "Kedokteran", "Universitas", "Gadjah", "Mada", "."], ["Dia", "kemudian", "melanjutkan", "pendidikan", "Klinis", "Kesehatan", "Reproduksi", "dan", "Penyakit", "Menular", "Seksual", "di", "Mahachulalongkornrajavidyalaya", "University", ",", "Bangkok", ",", "Thailand", "pada", "2004", "."]]
            ],
            "source": "cnn indonesia",
            "source_url": "https://www.cnnindonesia.com/hiburan/20170804120703-234-232443/lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun-lalu/",
            "summary": [
                [["Dokter", "Lula", "Kamal", "yang", "merupakan", "selebriti", "sekaligus", "rekan", "kerja", "Ryan", "Thamrin", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."], ["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", ",", "termasuk", "menjadi", "pembawa", "acara", "Dokter", "Oz", "Indonesia", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "di", "Pekanbaru", ",", "Riau", "untuk", "menjalani", "istirahat", "."]]
            ]
        }
    ]
    
    # Process user sample data
    processor = NewsDataProcessor()
    processed_data = processor.process_dataset(user_sample_data)
    
    if processed_data:
        item = processed_data[0]
        print(f"✓ Berhasil memproses data user")
        print(f"ID: {item['id']}")
        print(f"Category: {item['category']}")
        print(f"Source: {item['source']}")
        print(f"Full text length: {len(item['full_text'])} characters")
        print(f"Gold summary: {item['gold_summary']}")
        print(f"Original summary: {item['original_summary']}")
        
        # Test text extraction quality
        print(f"\nText preview (first 200 chars):")
        print(f"{item['full_text'][:200]}...")
        
        return processed_data
    
    return []

def run_all_tests():
    """
    Jalankan semua test
    """
    print("=== SISTEM TESTING SUMMARIZATION BERITA ===\n")
    
    try:
        # Test data processor
        processed_data = test_data_processor()
        
        # Test utils
        test_utils()
        
        # Test config
        test_config()
        
        # Test user sample data
        user_data = test_sample_data_processing()
        
        print("\n" + "=" * 50)
        print("✓ SEMUA TEST BERHASIL!")
        print("=" * 50)
        
        print("\nSistem siap digunakan untuk:")
        print("1. Memproses dataset berita dalam format yang diberikan")
        print("2. Melakukan summarization dengan model Gemma2 9B Sahabat AI")
        print("3. Evaluasi kualitas summary")
        print("4. Batch processing untuk dataset besar")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Silakan periksa error di atas dan perbaiki sebelum melanjutkan.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)