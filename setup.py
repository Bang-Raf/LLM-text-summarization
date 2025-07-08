#!/usr/bin/env python3
"""
Setup script untuk sistem summarization berita
"""

import subprocess
import sys
import os

def install_requirements():
    """
    Install requirements dari requirements.txt
    """
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements berhasil diinstall")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_imports():
    """
    Test import modules
    """
    print("\nTesting imports...")
    
    try:
        import json
        print("✓ json")
    except ImportError as e:
        print(f"❌ json: {e}")
        return False
    
    try:
        import re
        print("✓ re")
    except ImportError as e:
        print(f"❌ re: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"❌ tqdm: {e}")
        return False
    
    return True

def test_data_processing():
    """
    Test data processing tanpa model
    """
    print("\nTesting data processing...")
    
    try:
        from data_processor import NewsDataProcessor
        
        # Sample data
        sample_data = [
            {
                "category": "test",
                "gold_labels": [[False, True]],
                "id": "test-1",
                "paragraphs": [
                    [["Test", "kalimat", "1", "."], ["Test", "kalimat", "2", "."]]
                ],
                "source": "test",
                "source_url": "https://test.com",
                "summary": [
                    [["Test", "summary", "."]]
                ]
            }
        ]
        
        processor = NewsDataProcessor()
        processed_data = processor.process_dataset(sample_data)
        
        if processed_data and len(processed_data) > 0:
            item = processed_data[0]
            print(f"✓ Berhasil memproses data")
            print(f"  - ID: {item['id']}")
            print(f"  - Category: {item['category']}")
            print(f"  - Full text: {item['full_text']}")
            print(f"  - Gold summary: {item['gold_summary']}")
            return True
        else:
            print("❌ Gagal memproses data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data processing: {e}")
        return False

def test_utils():
    """
    Test utility functions
    """
    print("\nTesting utilities...")
    
    try:
        from utils import clean_text, calculate_rouge_scores
        
        # Test clean_text
        test_text = "  Test   text  with   extra   spaces  "
        cleaned = clean_text(test_text)
        print(f"✓ clean_text: '{cleaned}'")
        
        # Test ROUGE calculation
        summary1 = "Test summary one"
        summary2 = "Test summary two"
        rouge_scores = calculate_rouge_scores(summary1, summary2)
        print(f"✓ ROUGE calculation: {rouge_scores}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing utilities: {e}")
        return False

def main():
    """
    Main setup function
    """
    print("=== SETUP SISTEM SUMMARIZATION BERITA ===\n")
    
    # 1. Install requirements
    if not install_requirements():
        print("Gagal install requirements. Silakan install manual:")
        print("pip install -r requirements.txt")
        return False
    
    # 2. Test imports
    if not test_imports():
        print("Gagal import modules. Silakan periksa instalasi.")
        return False
    
    # 3. Test data processing
    if not test_data_processing():
        print("Gagal test data processing.")
        return False
    
    # 4. Test utilities
    if not test_utils():
        print("Gagal test utilities.")
        return False
    
    print("\n" + "=" * 50)
    print("✓ SETUP BERHASIL!")
    print("=" * 50)
    
    print("\nSistem siap digunakan. Untuk menjalankan:")
    print("1. python3 main.py - untuk demo lengkap")
    print("2. python3 test_system.py - untuk testing")
    print("3. python3 example_usage.py - untuk contoh penggunaan")
    
    print("\nCatatan:")
    print("- Model Gemma2 9B akan di-download saat pertama kali dijalankan")
    print("- Pastikan ada cukup memory (minimal 16GB RAM)")
    print("- GPU direkomendasikan untuk performa lebih baik")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)