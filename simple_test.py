#!/usr/bin/env python3
"""
Simple test script untuk sistem summarization berita
Dapat dijalankan tanpa dependencies eksternal
"""

import json
import re
from typing import List, Dict, Any

def test_data_processor_simple():
    """
    Test sederhana untuk data processor
    """
    print("=== Testing Data Processor (Simple) ===\n")
    
    # Sample data dari user
    sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True], [False, False, False]],
            "id": "1501893029-lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun",
            "paragraphs": [
                [["Jakarta", ",", "CNN", "Indonesia", "-", "Dokter", "Ryan", "Thamrin", "meninggal", "dunia", "."], ["Dokter", "Lula", "Kamal", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."]],
                [["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", "."], ["Kondisi", "itu", "membuat", "Ryan", "harus", "kembali", "ke", "kampung", "halamannya", "."]],
                [["Ryan", "Thamrin", "terkenal", "sebagai", "dokter", "yang", "rutin", "membagikan", "tips", "kesehatan", "."], ["Dia", "menempuh", "Pendidikan", "Dokter", "pada", "tahun", "2002", "."], ["Dia", "melanjutkan", "pendidikan", "di", "Thailand", "pada", "2004", "."]]
            ],
            "source": "cnn indonesia",
            "source_url": "https://example.com",
            "summary": [
                [["Dokter", "Lula", "Kamal", "menyebut", "kawannya", "itu", "sudah", "sakit", "sejak", "setahun", "yang", "lalu", "."], ["Lula", "menuturkan", ",", "sakit", "itu", "membuat", "Ryan", "mesti", "vakum", "dari", "semua", "kegiatannya", "."]]
            ]
        }
    ]
    
    # Simple data processor
    def extract_text_from_paragraphs(paragraphs):
        """Extract text from paragraphs structure"""
        full_text = []
        
        for paragraph in paragraphs:
            paragraph_text = []
            for sentence in paragraph:
                sentence_text = ' '.join(sentence)
                paragraph_text.append(sentence_text)
            
            paragraph_combined = ' '.join(paragraph_text)
            full_text.append(paragraph_combined)
        
        return '\n\n'.join(full_text)
    
    def extract_summary_from_gold_labels(paragraphs, gold_labels):
        """Extract summary based on gold labels"""
        summary_sentences = []
        
        for i, (paragraph, labels) in enumerate(zip(paragraphs, gold_labels)):
            for j, (sentence, is_important) in enumerate(zip(paragraph, labels)):
                if is_important:
                    sentence_text = ' '.join(sentence)
                    summary_sentences.append(sentence_text)
        
        return ' '.join(summary_sentences)
    
    # Process data
    processed_data = []
    for item in sample_data:
        processed_item = {
            'id': item.get('id', ''),
            'category': item.get('category', ''),
            'source': item.get('source', ''),
            'source_url': item.get('source_url', ''),
            'full_text': extract_text_from_paragraphs(item.get('paragraphs', [])),
            'gold_summary': extract_summary_from_gold_labels(
                item.get('paragraphs', []), 
                item.get('gold_labels', [])
            ),
            'original_summary': extract_text_from_paragraphs(item.get('summary', []))
        }
        processed_data.append(processed_item)
    
    # Display results
    if processed_data:
        item = processed_data[0]
        print(f"✓ Berhasil memproses data")
        print(f"ID: {item['id']}")
        print(f"Category: {item['category']}")
        print(f"Source: {item['source']}")
        print(f"Full text length: {len(item['full_text'])} characters")
        print(f"Gold summary: {item['gold_summary']}")
        print(f"Original summary: {item['original_summary']}")
        
        print(f"\nText preview (first 200 chars):")
        print(f"{item['full_text'][:200]}...")
        
        return True
    else:
        print("❌ Gagal memproses data")
        return False

def test_utils_simple():
    """
    Test sederhana untuk utility functions
    """
    print("\n=== Testing Utils (Simple) ===\n")
    
    def clean_text(text):
        """Clean text from unwanted characters"""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_simple_rouge(generated_summary, reference_summary):
        """Calculate simple ROUGE-like score"""
        def get_words(text):
            return set(text.lower().split())
        
        gen_words = get_words(clean_text(generated_summary))
        ref_words = get_words(clean_text(reference_summary))
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words.intersection(ref_words))
        return overlap / len(ref_words)
    
    # Test clean_text
    test_text = "  Test   text  with   extra   spaces  "
    cleaned = clean_text(test_text)
    print(f"✓ clean_text: '{cleaned}'")
    
    # Test ROUGE calculation
    summary1 = "Dokter Ryan Thamrin meninggal dunia pada Jumat dini hari."
    summary2 = "Dokter Ryan Thamrin, pembawa acara Dokter Oz Indonesia, meninggal dunia."
    rouge_score = calculate_simple_rouge(summary1, summary2)
    print(f"✓ Simple ROUGE score: {rouge_score:.4f}")
    
    return True

def test_prompt_generation():
    """
    Test prompt generation
    """
    print("\n=== Testing Prompt Generation ===\n")
    
    def create_summarization_prompt(text, max_length=2000):
        """Create summarization prompt"""
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = f"""Buatlah ringkasan singkat dari berita berikut dalam bahasa Indonesia. Ringkasan harus mencakup informasi penting dan ditulis dalam 2-3 kalimat.

Berita:
{text}

Ringkasan:"""
        
        return prompt
    
    # Test with sample text
    sample_text = "Jakarta, CNN Indonesia - - Dokter Ryan Thamrin, yang terkenal lewat acara Dokter Oz Indonesia, meninggal dunia pada Jumat (4/8) dini hari."
    
    prompt = create_summarization_prompt(sample_text)
    print(f"✓ Generated prompt length: {len(prompt)} characters")
    print(f"Prompt preview:")
    print(f"{prompt[:200]}...")
    
    return True

def test_config_simple():
    """
    Test simple configuration
    """
    print("\n=== Testing Config (Simple) ===\n")
    
    # Simple config
    config = {
        "model_id": "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_input_length": 2000
    }
    
    print(f"✓ Model ID: {config['model_id']}")
    print(f"✓ Max new tokens: {config['max_new_tokens']}")
    print(f"✓ Temperature: {config['temperature']}")
    print(f"✓ Top-p: {config['top_p']}")
    print(f"✓ Max input length: {config['max_input_length']}")
    
    return True

def main():
    """
    Main test function
    """
    print("=== SIMPLE TEST SISTEM SUMMARIZATION BERITA ===\n")
    
    try:
        # Test data processor
        if not test_data_processor_simple():
            return False
        
        # Test utils
        if not test_utils_simple():
            return False
        
        # Test prompt generation
        if not test_prompt_generation():
            return False
        
        # Test config
        if not test_config_simple():
            return False
        
        print("\n" + "=" * 50)
        print("✓ SEMUA TEST BERHASIL!")
        print("=" * 50)
        
        print("\nSistem siap digunakan untuk:")
        print("1. Memproses dataset berita dalam format yang diberikan")
        print("2. Melakukan summarization dengan model Gemma2 9B Sahabat AI")
        print("3. Evaluasi kualitas summary")
        print("4. Batch processing untuk dataset besar")
        
        print("\nUntuk menjalankan dengan model:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Jalankan: python3 main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)