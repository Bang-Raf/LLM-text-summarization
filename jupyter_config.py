#!/usr/bin/env python3
"""
Konfigurasi untuk Jupyter Notebook
"""

import os
import sys
from pathlib import Path

# Tambahkan direktori proyek ke Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

# Import semua modul yang diperlukan
try:
    from data_loader import NewsDatasetLoader
    from summarizer import GemmaSummarizer
    from evaluator import SummarizationEvaluator
    from visualizer import SummarizationVisualizer
    print("✅ Semua modul berhasil diimport!")
except ImportError as e:
    print(f"⚠️  Beberapa modul tidak dapat diimport: {e}")
    print("Pastikan semua dependencies sudah terinstall dengan menjalankan:")
    print("pip install -r requirements.txt")

# Konfigurasi untuk matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style untuk plot
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Konfigurasi untuk menampilkan plot di notebook
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    print("✅ Matplotlib dan Seaborn berhasil dikonfigurasi!")
except ImportError:
    print("⚠️  Matplotlib/Seaborn tidak tersedia. Visualisasi mungkin tidak berfungsi.")

# Konfigurasi untuk pandas
try:
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', None)
    print("✅ Pandas berhasil dikonfigurasi!")
except ImportError:
    print("⚠️  Pandas tidak tersedia.")

# Konfigurasi untuk numpy
try:
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    print("✅ NumPy berhasil dikonfigurasi!")
except ImportError:
    print("⚠️  NumPy tidak tersedia.")

# Fungsi helper untuk notebook
def setup_notebook():
    """
    Setup awal untuk notebook
    """
    print("="*60)
    print("SETUP NOTEBOOK EVALUASI TEXT SUMMARIZATION")
    print("="*60)
    
    # Cek direktori data
    data_dir = Path("data")
    if data_dir.exists():
        train_files = list(data_dir.glob("train.*.jsonl"))
        print(f"📁 Direktori data ditemukan dengan {len(train_files)} file training")
        for file in train_files:
            print(f"   - {file.name}")
    else:
        print("⚠️  Direktori data tidak ditemukan!")
        print("   Buat direktori 'data' dan letakkan file train.XX.jsonl di dalamnya")
    
    # Cek direktori results
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*"))
        print(f"📊 Direktori results ditemukan dengan {len(result_files)} file")
    else:
        print("📊 Direktori results akan dibuat otomatis saat menjalankan evaluasi")
    
    # Cek CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA tersedia: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA tidak tersedia, akan menggunakan CPU")
    except ImportError:
        print("⚠️  PyTorch tidak tersedia")
    
    print("\n✅ Setup notebook selesai!")
    print("Anda dapat mulai menjalankan cell-cell di notebook")

# Fungsi untuk quick test
def quick_test():
    """
    Quick test untuk memastikan semua komponen berfungsi
    """
    print("\n🧪 QUICK TEST")
    print("-" * 30)
    
    try:
        # Test data loader
        data_loader = NewsDatasetLoader()
        print("✅ DataLoader berhasil dibuat")
        
        # Test evaluator
        evaluator = SummarizationEvaluator()
        print("✅ Evaluator berhasil dibuat")
        
        # Test visualizer
        visualizer = SummarizationVisualizer()
        print("✅ Visualizer berhasil dibuat")
        
        print("\n🎉 Semua komponen berfungsi dengan baik!")
        
    except Exception as e:
        print(f"❌ Error saat testing: {e}")

if __name__ == "__main__":
    setup_notebook()
    quick_test()