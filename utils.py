"""
Utility functions untuk sistem summarization berita
"""

import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Memuat file JSONL (JSON Lines)
    
    Args:
        file_path (str): Path ke file JSONL
        
    Returns:
        List[Dict[str, Any]]: List data dari file
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    
    return data

def save_jsonl_file(data: List[Dict[str, Any]], file_path: str):
    """
    Menyimpan data ke file JSONL
    
    Args:
        data (List[Dict[str, Any]]): Data yang akan disimpan
        file_path (str): Path file output
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari karakter yang tidak diinginkan
    
    Args:
        text (str): Teks mentah
        
    Returns:
        str: Teks yang sudah dibersihkan
    """
    # Hapus karakter kontrol
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalisasi whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Hapus spasi di awal dan akhir
    text = text.strip()
    
    return text

def calculate_rouge_scores(generated_summary: str, reference_summary: str) -> Dict[str, float]:
    """
    Menghitung skor ROUGE sederhana (implementasi dasar)
    
    Args:
        generated_summary (str): Summary yang dihasilkan
        reference_summary (str): Summary referensi
        
    Returns:
        Dict[str, float]: Skor ROUGE
    """
    def get_ngrams(text: str, n: int) -> set:
        """Mendapatkan n-grams dari teks"""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i:i+n]))
        return ngrams
    
    # Bersihkan teks
    gen_clean = clean_text(generated_summary)
    ref_clean = clean_text(reference_summary)
    
    if not gen_clean or not ref_clean:
        return {'rouge-1': 0.0, 'rouge-2': 0.0}
    
    # Hitung unigrams dan bigrams
    gen_1grams = get_ngrams(gen_clean, 1)
    ref_1grams = get_ngrams(ref_clean, 1)
    gen_2grams = get_ngrams(gen_clean, 2)
    ref_2grams = get_ngrams(ref_clean, 2)
    
    # Hitung overlap
    overlap_1 = len(gen_1grams.intersection(ref_1grams))
    overlap_2 = len(gen_2grams.intersection(ref_2grams))
    
    # Hitung precision dan recall
    precision_1 = overlap_1 / len(gen_1grams) if gen_1grams else 0
    recall_1 = overlap_1 / len(ref_1grams) if ref_1grams else 0
    precision_2 = overlap_2 / len(gen_2grams) if gen_2grams else 0
    recall_2 = overlap_2 / len(ref_2grams) if ref_2grams else 0
    
    # Hitung F1 score
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    f1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0
    
    return {
        'rouge-1': f1_1,
        'rouge-2': f1_2,
        'precision-1': precision_1,
        'recall-1': recall_1,
        'precision-2': precision_2,
        'recall-2': recall_2
    }

def evaluate_summaries(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluasi batch summaries
    
    Args:
        results (List[Dict[str, Any]]): Hasil summarization dengan gold summary
        
    Returns:
        Dict[str, Any]: Metrik evaluasi
    """
    metrics = {
        'total_items': len(results),
        'successful_summaries': 0,
        'rouge_scores': [],
        'compression_ratios': [],
        'summary_lengths': []
    }
    
    for item in results:
        if 'generated_summary' in item and 'gold_summary' in item:
            gen_summary = item['generated_summary']
            gold_summary = item['gold_summary']
            
            if gen_summary and gold_summary:
                metrics['successful_summaries'] += 1
                
                # Hitung ROUGE scores
                rouge_scores = calculate_rouge_scores(gen_summary, gold_summary)
                metrics['rouge_scores'].append(rouge_scores)
                
                # Hitung compression ratio
                gen_length = len(gen_summary.split())
                gold_length = len(gold_summary.split())
                if gold_length > 0:
                    compression_ratio = gen_length / gold_length
                    metrics['compression_ratios'].append(compression_ratio)
                
                metrics['summary_lengths'].append(gen_length)
    
    # Hitung rata-rata
    if metrics['rouge_scores']:
        avg_rouge_1 = np.mean([score['rouge-1'] for score in metrics['rouge_scores']])
        avg_rouge_2 = np.mean([score['rouge-2'] for score in metrics['rouge_scores']])
        metrics['avg_rouge_1'] = avg_rouge_1
        metrics['avg_rouge_2'] = avg_rouge_2
    
    if metrics['compression_ratios']:
        metrics['avg_compression_ratio'] = np.mean(metrics['compression_ratios'])
    
    if metrics['summary_lengths']:
        metrics['avg_summary_length'] = np.mean(metrics['summary_lengths'])
    
    return metrics

def print_evaluation_results(metrics: Dict[str, Any]):
    """
    Mencetak hasil evaluasi dengan format yang rapi
    
    Args:
        metrics (Dict[str, Any]): Metrik evaluasi
    """
    print("\n=== HASIL EVALUASI ===")
    print(f"Total item: {metrics['total_items']}")
    print(f"Berhasil diringkas: {metrics['successful_summaries']}")
    print(f"Success rate: {metrics['successful_summaries']/metrics['total_items']*100:.1f}%")
    
    if 'avg_rouge_1' in metrics:
        print(f"\nROUGE Scores:")
        print(f"- ROUGE-1: {metrics['avg_rouge_1']:.4f}")
        print(f"- ROUGE-2: {metrics['avg_rouge_2']:.4f}")
    
    if 'avg_compression_ratio' in metrics:
        print(f"\nCompression Ratio: {metrics['avg_compression_ratio']:.2f}")
    
    if 'avg_summary_length' in metrics:
        print(f"Rata-rata panjang summary: {metrics['avg_summary_length']:.1f} kata")

def create_sample_dataset(output_file: str = "sample_dataset.jsonl"):
    """
    Membuat dataset sample untuk testing
    
    Args:
        output_file (str): Nama file output
    """
    sample_data = [
        {
            "category": "tajuk utama",
            "gold_labels": [[False, True], [True, True], [False, False, False]],
            "id": "sample-1",
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
        },
        {
            "category": "teknologi",
            "gold_labels": [[False, True], [True, True], [False, False]],
            "id": "sample-2",
            "paragraphs": [
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "."], ["Kedua", "model", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."]],
                [["Mereka", "adalah", "Asus", "ZenFone", "4", "Selfie", "Pro", "dan", "ZenFone", "4", "Selfie", "."], ["Kedua", "model", "diracik", "sebagai", "jawaban", "atas", "kekurangan", "kompetitor", "."]],
                [["ZenFone", "4", "Selfie", "Pro", "dibanderol", "seharga", "Rp", "5", "juta", "."], ["ZenFone", "4", "Selfie", "dipatok", "di", "harga", "Rp", "3,5", "juta", "."], ["Kedua", "perangkat", "sudah", "mulai", "dipasarkan", "."]]
            ],
            "source": "dailysocial.id",
            "source_url": "https://example.com",
            "summary": [
                [["Asus", "memperkenalkan", "ZenFone", "generasi", "keempat", "."], ["Kedua", "model", "dibekali", "setup", "kamera", "ganda", "di", "depan", "."], ["Kedua", "model", "diracik", "sebagai", "jawaban", "atas", "kekurangan", "kompetitor", "."]]
            ]
        }
    ]
    
    save_jsonl_file(sample_data, output_file)
    print(f"Sample dataset berhasil dibuat: {output_file}")
    return sample_data