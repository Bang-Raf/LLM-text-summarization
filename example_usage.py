#!/usr/bin/env python3
"""
Contoh penggunaan modul evaluasi text summarization
"""

import os
import json
from data_loader import NewsDatasetLoader
from summarizer import GemmaSummarizer
from evaluator import SummarizationEvaluator
from visualizer import SummarizationVisualizer

def example_usage():
    """
    Contoh penggunaan lengkap dari semua modul
    """
    print("="*60)
    print("CONTOH PENGGUNAAN EVALUASI TEXT SUMMARIZATION")
    print("="*60)
    
    # Konfigurasi
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    SAMPLE_SIZE = 10  # Gunakan sampel kecil untuk demo
    
    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print("\n1. MEMUAT DATASET")
    print("-" * 30)
    
    data_loader = NewsDatasetLoader(data_dir=DATA_DIR)
    
    # Cek apakah direktori data ada
    if not os.path.exists(DATA_DIR):
        print(f"Direktori {DATA_DIR} tidak ditemukan!")
        print("Membuat contoh data untuk demo...")
        create_sample_data(DATA_DIR)
    
    # Load data
    try:
        raw_data = data_loader.load_all_train_files()
        processed_data = data_loader.preprocess_data(raw_data)
        print(f"Dataset berhasil dimuat: {len(processed_data)} artikel")
        
        # Sampling untuk demo
        if len(processed_data) > SAMPLE_SIZE:
            import random
            random.seed(42)
            evaluation_data = random.sample(processed_data, SAMPLE_SIZE)
            print(f"Menggunakan {len(evaluation_data)} sampel untuk demo")
        else:
            evaluation_data = processed_data
            
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return
    
    # 2. Initialize Model
    print("\n2. INISIALISASI MODEL")
    print("-" * 30)
    
    try:
        # Gunakan model yang lebih kecil untuk demo
        summarizer = GemmaSummarizer(
            model_name="google/gemma2-2b",  # Model yang lebih kecil
            device="cpu"  # Gunakan CPU untuk demo
        )
        print("Model berhasil diinisialisasi!")
    except Exception as e:
        print(f"Error saat inisialisasi model: {e}")
        print("Menggunakan mock summarizer untuk demo...")
        summarizer = MockSummarizer()
    
    # 3. Generate Summaries
    print("\n3. GENERATE SUMMARIES")
    print("-" * 30)
    
    try:
        results_with_summaries = summarizer.summarize_dataset(
            dataset=evaluation_data,
            max_length=256,  # Lebih pendek untuk demo
            temperature=0.7
        )
        print(f"Berhasil generate {len(results_with_summaries)} summaries")
        
        # Tampilkan contoh
        if results_with_summaries:
            sample = results_with_summaries[0]
            print(f"\nContoh hasil:")
            print(f"ID: {sample['id']}")
            print(f"Category: {sample['category']}")
            print(f"Reference: {sample['summary'][:100]}...")
            print(f"Generated: {sample['generated_summary'][:100]}...")
            
    except Exception as e:
        print(f"Error saat generate summaries: {e}")
        return
    
    # 4. Evaluate Results
    print("\n4. EVALUASI HASIL")
    print("-" * 30)
    
    evaluator = SummarizationEvaluator(lang="id")
    evaluation_results = evaluator.evaluate_dataset(results_with_summaries)
    
    # Print results
    evaluator.print_results(evaluation_results)
    
    # 5. Create Visualizations
    print("\n5. MEMBUAT VISUALISASI")
    print("-" * 30)
    
    visualizer = SummarizationVisualizer()
    evaluation_df = evaluator.create_evaluation_dataframe(results_with_summaries)
    
    # Generate plots
    try:
        # Metrics comparison
        metrics_path = os.path.join(OUTPUT_DIR, "demo_metrics_comparison.png")
        visualizer.plot_metrics_comparison(evaluation_results, save_path=metrics_path)
        print(f"Plot metrics tersimpan: {metrics_path}")
        
        # Category analysis
        category_path = os.path.join(OUTPUT_DIR, "demo_category_analysis.png")
        visualizer.plot_category_analysis(evaluation_df, save_path=category_path)
        print(f"Plot kategori tersimpan: {category_path}")
        
    except Exception as e:
        print(f"Error saat membuat visualisasi: {e}")
    
    # 6. Save Results
    print("\n6. MENYIMPAN HASIL")
    print("-" * 30)
    
    # Save evaluation results
    results_path = os.path.join(OUTPUT_DIR, "demo_evaluation_results.json")
    evaluator.save_results(evaluation_results, results_path)
    
    # Save results with summaries
    summaries_path = os.path.join(OUTPUT_DIR, "demo_results_with_summaries.jsonl")
    data_loader.save_processed_data(results_with_summaries, summaries_path)
    
    # Save evaluation dataframe
    df_path = os.path.join(OUTPUT_DIR, "demo_evaluation_dataframe.csv")
    evaluation_df.to_csv(df_path, index=False)
    
    print("Semua hasil demo tersimpan!")
    
    # 7. Summary
    print("\n7. RINGKASAN")
    print("-" * 30)
    print(f"Total artikel: {len(evaluation_data)}")
    print(f"ROUGE-1: {evaluation_results['summary']['rouge1']:.3f}")
    print(f"ROUGE-2: {evaluation_results['summary']['rouge2']:.3f}")
    print(f"ROUGE-L: {evaluation_results['summary']['rougeL']:.3f}")
    print(f"BLEU: {evaluation_results['summary']['bleu']:.3f}")
    print(f"BERTScore-F1: {evaluation_results['summary']['bertscore_f1']:.3f}")
    print(f"\nHasil tersimpan di: {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("DEMO SELESAI!")
    print("="*60)

def create_sample_data(data_dir):
    """
    Membuat contoh data untuk demo
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Contoh data berdasarkan format yang diberikan
    sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True], [False, False]],
            "id": "demo-001-lula-kamal-dokter-ryan-thamrin",
            "paragraphs": [
                [["Jakarta", ",", "CNN", "Indonesia", "-", "Dokter", "Ryan", "Thamrin", ",", "yang", "terkenal", "lewat", "acara", "Dokter", "Oz", "Indonesia", ",", "meninggal", "dunia", "pada", "Jumat", "dini", "hari", "."]],
                [["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", ",", "termasuk", "menjadi", "pembawa", "acara", "Dokter", "Oz", "Indonesia", "."]]
            ],
            "source": "cnn indonesia",
            "source_url": "https://example.com/demo-article-1",
            "summary": [
                [["Dokter", "Lula", "Kamal", "yang", "merupakan", "selebriti", "sekaligus", "rekan", "kerja", "Ryan", "Thamrin", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."]]
            ]
        },
        {
            "category": "teknologi",
            "gold_labels": [[False, False, True], [True, True], [False, False]],
            "id": "demo-002-dua-smartphone-zenfone-baru",
            "paragraphs": [
                [["Selfie", "ialah", "salah", "satu", "tema", "terpanas", "di", "kalangan", "produsen", "smartphone", ",", "bahkan", "menjadi", "senjata", "andalan", "beberapa", "brand", "terkenal", "."]],
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "dan", "keduanya", "sama-sama", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."]]
            ],
            "source": "dailysocial.id",
            "source_url": "https://example.com/demo-article-2",
            "summary": [
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "dan", "keduanya", "sama-sama", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."]]
            ]
        }
    ]
    
    # Simpan ke file train.01.jsonl
    train_file = os.path.join(data_dir, "train.01.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Contoh data dibuat di: {train_file}")

class MockSummarizer:
    """
    Mock summarizer untuk demo tanpa model
    """
    def __init__(self):
        print("Menggunakan Mock Summarizer untuk demo")
    
    def summarize_dataset(self, dataset, max_length=256, temperature=0.7):
        """
        Generate mock summaries
        """
        results = []
        for item in dataset:
            # Buat mock summary berdasarkan text
            mock_summary = f"Ringkasan artikel tentang {item['category']}: {item['summary'][:50]}..."
            
            result_item = item.copy()
            result_item['generated_summary'] = mock_summary
            results.append(result_item)
        
        return results

if __name__ == "__main__":
    example_usage()