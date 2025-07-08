#!/usr/bin/env python3
"""
Demo sederhana untuk evaluasi text summarization
Versi yang tidak memerlukan dependencies eksternal
"""

import json
import os
from typing import List, Dict, Any

class SimpleDataLoader:
    """Data loader sederhana tanpa pandas"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Memuat file JSONL"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_all_train_files(self) -> List[Dict[str, Any]]:
        """Memuat semua file train.XX.jsonl"""
        all_data = []
        
        # Cari semua file train.XX.jsonl
        train_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith('train.') and file.endswith('.jsonl'):
                train_files.append(file)
        
        train_files.sort()
        
        print(f"Menemukan {len(train_files)} file training:")
        for file in train_files:
            print(f"  - {file}")
        
        # Muat setiap file
        for file in train_files:
            file_path = os.path.join(self.data_dir, file)
            data = self.load_jsonl_file(file_path)
            all_data.extend(data)
            
        print(f"Total {len(all_data)} artikel berita dimuat")
        return all_data
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocessing data"""
        processed_data = []
        
        for item in data:
            # Gabungkan paragraphs menjadi teks lengkap
            full_text = self._combine_paragraphs(item['paragraphs'])
            
            # Gabungkan summary menjadi teks lengkap
            full_summary = self._combine_paragraphs(item['summary'])
            
            processed_item = {
                'id': item['id'],
                'category': item['category'],
                'source': item['source'],
                'source_url': item['source_url'],
                'text': full_text,
                'summary': full_summary,
                'gold_labels': item['gold_labels'],
                'paragraphs': item['paragraphs'],
                'summary_paragraphs': item['summary']
            }
            
            processed_data.append(processed_item)
            
        return processed_data
    
    def _combine_paragraphs(self, paragraphs: List[List[List[str]]]) -> str:
        """Menggabungkan paragraphs menjadi satu teks"""
        full_text = ""
        
        for paragraph in paragraphs:
            paragraph_text = ""
            for sentence in paragraph:
                sentence_text = " ".join(sentence)
                paragraph_text += sentence_text + " "
            full_text += paragraph_text.strip() + "\n\n"
            
        return full_text.strip()

class SimpleSummarizer:
    """Mock summarizer untuk demo"""
    
    def __init__(self):
        print("Menggunakan Simple Summarizer untuk demo")
    
    def summarize_dataset(self, dataset: List[Dict[str, Any]], max_length: int = 512, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """Generate mock summaries"""
        results = []
        
        for item in dataset:
            # Buat mock summary berdasarkan text dan category
            mock_summary = self._generate_mock_summary(item)
            
            result_item = item.copy()
            result_item['generated_summary'] = mock_summary
            results.append(result_item)
        
        return results
    
    def _generate_mock_summary(self, item: Dict[str, Any]) -> str:
        """Generate mock summary berdasarkan kategori"""
        category = item['category']
        text_words = item['text'].split()[:50]  # Ambil 50 kata pertama
        
        if category == "tajuk utama":
            return f"Berita utama: {item['summary'][:100]}... (Ringkasan otomatis)"
        elif category == "teknologi":
            return f"Berita teknologi: {item['summary'][:100]}... (Ringkasan otomatis)"
        else:
            return f"Ringkasan artikel {category}: {item['summary'][:100]}... (Ringkasan otomatis)"

class SimpleEvaluator:
    """Evaluator sederhana untuk demo"""
    
    def __init__(self):
        print("Simple Evaluator berhasil diinisialisasi!")
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluasi sederhana"""
        total_items = len(dataset)
        
        # Hitung rata-rata panjang
        total_ref_length = sum(len(item['summary'].split()) for item in dataset)
        total_pred_length = sum(len(item['generated_summary'].split()) for item in dataset)
        
        avg_ref_length = total_ref_length / total_items
        avg_pred_length = total_pred_length / total_items
        
        # Mock scores untuk demo
        results = {
            'rouge': {
                'rouge1': 0.3245,
                'rouge2': 0.1567,
                'rougeL': 0.2987,
                'rouge1_std': 0.1234,
                'rouge2_std': 0.0891,
                'rougeL_std': 0.1156
            },
            'bleu': {
                'bleu': 0.2345
            },
            'bertscore': {
                'bertscore_precision': 0.3456,
                'bertscore_recall': 0.3123,
                'bertscore_f1': 0.3287,
                'bertscore_precision_std': 0.0789,
                'bertscore_recall_std': 0.0823,
                'bertscore_f1_std': 0.0801
            },
            'summary': {
                'rouge1': 0.3245,
                'rouge2': 0.1567,
                'rougeL': 0.2987,
                'bleu': 0.2345,
                'bertscore_f1': 0.3287
            },
            'statistics': {
                'total_items': total_items,
                'avg_reference_length': avg_ref_length,
                'avg_prediction_length': avg_pred_length
            }
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print hasil evaluasi"""
        print("\n" + "="*50)
        print("HASIL EVALUASI SUMMARIZATION")
        print("="*50)
        
        # ROUGE Scores
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {results['rouge']['rouge1']:.4f} ± {results['rouge']['rouge1_std']:.4f}")
        print(f"  ROUGE-2: {results['rouge']['rouge2']:.4f} ± {results['rouge']['rouge2_std']:.4f}")
        print(f"  ROUGE-L: {results['rouge']['rougeL']:.4f} ± {results['rouge']['rougeL_std']:.4f}")
        
        # BLEU Score
        print(f"\nBLEU Score:")
        print(f"  BLEU: {results['bleu']['bleu']:.4f}")
        
        # BERTScore
        print(f"\nBERTScore:")
        print(f"  Precision: {results['bertscore']['bertscore_precision']:.4f} ± {results['bertscore']['bertscore_precision_std']:.4f}")
        print(f"  Recall: {results['bertscore']['bertscore_recall']:.4f} ± {results['bertscore']['bertscore_recall_std']:.4f}")
        print(f"  F1: {results['bertscore']['bertscore_f1']:.4f} ± {results['bertscore']['bertscore_f1_std']:.4f}")
        
        # Statistics
        print(f"\nStatistik:")
        print(f"  Total artikel: {results['statistics']['total_items']}")
        print(f"  Rata-rata panjang reference: {results['statistics']['avg_reference_length']:.1f} kata")
        print(f"  Rata-rata panjang prediction: {results['statistics']['avg_prediction_length']:.1f} kata")
        
        print("\n" + "="*50)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Simpan hasil evaluasi"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Hasil evaluasi tersimpan di: {output_path}")

def main():
    """Fungsi utama demo"""
    print("="*60)
    print("DEMO EVALUASI TEXT SUMMARIZATION")
    print("="*60)
    
    # Konfigurasi
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    
    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print("\n1. MEMUAT DATASET")
    print("-" * 30)
    
    data_loader = SimpleDataLoader(data_dir=DATA_DIR)
    
    if not os.path.exists(DATA_DIR):
        print(f"Direktori {DATA_DIR} tidak ditemukan!")
        return
    
    # Load dan preprocess data
    raw_data = data_loader.load_all_train_files()
    processed_data = data_loader.preprocess_data(raw_data)
    
    print(f"Dataset berhasil dimuat: {len(processed_data)} artikel")
    
    # Tampilkan contoh data
    if processed_data:
        sample = processed_data[0]
        print(f"\nContoh data:")
        print(f"ID: {sample['id']}")
        print(f"Category: {sample['category']}")
        print(f"Source: {sample['source']}")
        print(f"Text length: {len(sample['text'])} characters")
        print(f"Summary length: {len(sample['summary'])} characters")
        print(f"Text preview: {sample['text'][:200]}...")
        print(f"Summary: {sample['summary']}")
    
    # 2. Initialize Model
    print("\n2. INISIALISASI MODEL")
    print("-" * 30)
    
    summarizer = SimpleSummarizer()
    print("Model berhasil diinisialisasi!")
    
    # 3. Generate Summaries
    print("\n3. GENERATE SUMMARIES")
    print("-" * 30)
    
    results_with_summaries = summarizer.summarize_dataset(
        dataset=processed_data,
        max_length=256,
        temperature=0.7
    )
    print(f"Berhasil generate {len(results_with_summaries)} summaries")
    
    # Tampilkan contoh hasil
    if results_with_summaries:
        sample = results_with_summaries[0]
        print(f"\nContoh hasil:")
        print(f"ID: {sample['id']}")
        print(f"Category: {sample['category']}")
        print(f"Reference: {sample['summary'][:100]}...")
        print(f"Generated: {sample['generated_summary']}")
    
    # 4. Evaluate Results
    print("\n4. EVALUASI HASIL")
    print("-" * 30)
    
    evaluator = SimpleEvaluator()
    evaluation_results = evaluator.evaluate_dataset(results_with_summaries)
    
    # Print results
    evaluator.print_results(evaluation_results)
    
    # 5. Save Results
    print("\n5. MENYIMPAN HASIL")
    print("-" * 30)
    
    # Save evaluation results
    results_path = os.path.join(OUTPUT_DIR, "demo_evaluation_results.json")
    evaluator.save_results(evaluation_results, results_path)
    
    # Save results with summaries
    summaries_path = os.path.join(OUTPUT_DIR, "demo_results_with_summaries.jsonl")
    with open(summaries_path, 'w', encoding='utf-8') as f:
        for item in results_with_summaries:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Hasil dengan summaries tersimpan di: {summaries_path}")
    
    # 6. Summary
    print("\n6. RINGKASAN")
    print("-" * 30)
    print(f"Total artikel: {len(processed_data)}")
    print(f"ROUGE-1: {evaluation_results['summary']['rouge1']:.3f}")
    print(f"ROUGE-2: {evaluation_results['summary']['rouge2']:.3f}")
    print(f"ROUGE-L: {evaluation_results['summary']['rougeL']:.3f}")
    print(f"BLEU: {evaluation_results['summary']['bleu']:.3f}")
    print(f"BERTScore-F1: {evaluation_results['summary']['bertscore_f1']:.3f}")
    print(f"\nHasil tersimpan di: {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("DEMO SELESAI!")
    print("="*60)
    
    print("\nCATATAN:")
    print("- Ini adalah demo dengan mock data dan scores")
    print("- Untuk evaluasi yang sebenarnya, install dependencies dan gunakan model Gemma2 9B")
    print("- File lengkap tersedia di repository ini")

if __name__ == "__main__":
    main()