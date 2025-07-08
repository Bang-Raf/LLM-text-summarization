#!/usr/bin/env python3
"""
Script evaluasi text summarization dengan Gemma2 9B
Versi Python dari Jupyter notebook
"""

import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import NewsDatasetLoader
from summarizer import GemmaSummarizer
from evaluator import SummarizationEvaluator
from visualizer import SummarizationVisualizer

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def main():
    """
    Fungsi utama untuk evaluasi text summarization
    """
    print("="*60)
    print("EVALUASI TEXT SUMMARIZATION DENGAN GEMMA2 9B")
    print("="*60)
    
    # Konfigurasi
    CONFIG = {
        'data_dir': 'data',  # Direktori yang berisi file dataset train.XX.jsonl
        'model_name': 'google/gemma2-9b',  # Model yang akan digunakan
        'device': None,  # None untuk auto-detect, atau 'cuda'/'cpu'
        'max_length': 512,  # Panjang maksimal summary
        'temperature': 0.7,  # Temperature untuk sampling
        'sample_size': 100,  # Jumlah sampel untuk evaluasi (None untuk semua)
        'output_dir': 'results'  # Direktori untuk menyimpan hasil
    }
    
    print("Konfigurasi:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Buat direktori output jika belum ada
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. Load Dataset
    print("\n1. MEMUAT DATASET")
    print("-" * 30)
    
    data_loader = NewsDatasetLoader(data_dir=CONFIG['data_dir'])
    
    # Cek apakah direktori data ada
    if not os.path.exists(CONFIG['data_dir']):
        print(f"Direktori {CONFIG['data_dir']} tidak ditemukan!")
        print("Pastikan file dataset train.XX.jsonl ada di direktori tersebut.")
        return
    
    # Load dan preprocess data
    raw_data = data_loader.load_all_train_files()
    processed_data = data_loader.preprocess_data(raw_data)
    
    print(f"Dataset berhasil dimuat: {len(processed_data)} artikel")
    
    # 2. Sampling Data
    print("\n2. SAMPLING DATA")
    print("-" * 30)
    
    if CONFIG['sample_size'] and CONFIG['sample_size'] < len(processed_data):
        import random
        random.seed(42)
        evaluation_data = random.sample(processed_data, CONFIG['sample_size'])
        print(f"Sampling {len(evaluation_data)} artikel untuk evaluasi")
    else:
        evaluation_data = processed_data
        print(f"Menggunakan semua {len(evaluation_data)} artikel")
    
    # 3. Initialize Model
    print("\n3. INISIALISASI MODEL")
    print("-" * 30)
    
    try:
        summarizer = GemmaSummarizer(
            model_name=CONFIG['model_name'],
            device=CONFIG['device']
        )
        print("Model berhasil diinisialisasi!")
    except Exception as e:
        print(f"Error saat inisialisasi model: {e}")
        print("Pastikan Anda memiliki akses internet dan ruang disk yang cukup.")
        return
    
    # 4. Generate Summaries
    print("\n4. GENERATE SUMMARIES")
    print("-" * 30)
    
    try:
        results_with_summaries = summarizer.summarize_dataset(
            dataset=evaluation_data,
            max_length=CONFIG['max_length'],
            temperature=CONFIG['temperature']
        )
        print(f"Berhasil generate {len(results_with_summaries)} summaries")
        
        # Tampilkan contoh hasil
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
    
    # 5. Evaluate Results
    print("\n5. EVALUASI HASIL")
    print("-" * 30)
    
    evaluator = SummarizationEvaluator(lang="id")
    evaluation_results = evaluator.evaluate_dataset(results_with_summaries)
    
    # Print results
    evaluator.print_results(evaluation_results)
    
    # 6. Create Visualizations
    print("\n6. MEMBUAT VISUALISASI")
    print("-" * 30)
    
    visualizer = SummarizationVisualizer()
    evaluation_df = evaluator.create_evaluation_dataframe(results_with_summaries)
    
    # Generate plots
    plots = [
        ('metrics_comparison.png', lambda: visualizer.plot_metrics_comparison(evaluation_results)),
        ('category_analysis.png', lambda: visualizer.plot_category_analysis(evaluation_df)),
        ('source_analysis.png', lambda: visualizer.plot_source_analysis(evaluation_df)),
        ('length_analysis.png', lambda: visualizer.plot_length_analysis(evaluation_df)),
        ('summary_report.png', lambda: visualizer.create_summary_report(evaluation_results, evaluation_df))
    ]
    
    for plot_name, plot_func in plots:
        try:
            plot_path = os.path.join(CONFIG['output_dir'], plot_name)
            plot_func()
            print(f"Plot tersimpan: {plot_name}")
        except Exception as e:
            print(f"Error saat membuat {plot_name}: {e}")
    
    # 7. Save Results
    print("\n7. MENYIMPAN HASIL")
    print("-" * 30)
    
    # Save evaluation results
    results_path = os.path.join(CONFIG['output_dir'], 'evaluation_results.json')
    evaluator.save_results(evaluation_results, results_path)
    
    # Save results with summaries
    summaries_path = os.path.join(CONFIG['output_dir'], 'results_with_summaries.jsonl')
    data_loader.save_processed_data(results_with_summaries, summaries_path)
    
    # Save evaluation dataframe
    df_path = os.path.join(CONFIG['output_dir'], 'evaluation_dataframe.csv')
    evaluation_df.to_csv(df_path, index=False)
    
    # Create final report
    report = {
        'config': CONFIG,
        'dataset_info': {
            'total_articles': len(evaluation_data),
            'categories': evaluation_df['category'].nunique(),
            'sources': evaluation_df['source'].nunique(),
            'avg_text_length': evaluation_df['reference_length'].mean(),
            'avg_summary_length': evaluation_df['prediction_length'].mean()
        },
        'evaluation_results': evaluation_results,
        'files_generated': [
            'evaluation_results.json',
            'results_with_summaries.jsonl',
            'evaluation_dataframe.csv',
            'metrics_comparison.png',
            'category_analysis.png',
            'source_analysis.png',
            'length_analysis.png',
            'summary_report.png'
        ]
    }
    
    report_path = os.path.join(CONFIG['output_dir'], 'final_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("Semua hasil tersimpan!")
    
    # 8. Print Summary
    print("\n8. RINGKASAN")
    print("-" * 30)
    print(f"Total artikel: {len(evaluation_data)}")
    print(f"ROUGE-1: {evaluation_results['summary']['rouge1']:.3f}")
    print(f"ROUGE-2: {evaluation_results['summary']['rouge2']:.3f}")
    print(f"ROUGE-L: {evaluation_results['summary']['rougeL']:.3f}")
    print(f"BLEU: {evaluation_results['summary']['bleu']:.3f}")
    print(f"BERTScore-F1: {evaluation_results['summary']['bertscore_f1']:.3f}")
    print(f"\nHasil tersimpan di: {CONFIG['output_dir']}")
    
    print("\n" + "="*60)
    print("EVALUASI SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()