import json
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

class NewsDatasetLoader:
    """
    Class untuk memuat dataset berita dari file JSONL
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inisialisasi data loader
        
        Args:
            data_dir: Direktori yang berisi file dataset
        """
        self.data_dir = data_dir
        
    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Memuat file JSONL dan mengembalikan list of dictionaries
        
        Args:
            file_path: Path ke file JSONL
            
        Returns:
            List of dictionaries yang berisi data berita
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_all_train_files(self) -> List[Dict[str, Any]]:
        """
        Memuat semua file train.XX.jsonl
        
        Returns:
            List of dictionaries yang berisi semua data training
        """
        all_data = []
        
        # Cari semua file train.XX.jsonl
        train_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith('train.') and file.endswith('.jsonl'):
                train_files.append(file)
        
        train_files.sort()  # Urutkan berdasarkan nomor
        
        print(f"Menemukan {len(train_files)} file training:")
        for file in train_files:
            print(f"  - {file}")
        
        # Muat setiap file
        for file in tqdm(train_files, desc="Loading training files"):
            file_path = os.path.join(self.data_dir, file)
            data = self.load_jsonl_file(file_path)
            all_data.extend(data)
            
        print(f"Total {len(all_data)} artikel berita dimuat")
        return all_data
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocessing data untuk format yang lebih mudah digunakan
        
        Args:
            data: Raw data dari JSONL
            
        Returns:
            Data yang sudah dipreprocess
        """
        processed_data = []
        
        for item in tqdm(data, desc="Preprocessing data"):
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
        """
        Menggabungkan paragraphs menjadi satu teks
        
        Args:
            paragraphs: List of paragraphs yang berisi list of sentences yang berisi list of tokens
            
        Returns:
            Teks yang sudah digabungkan
        """
        full_text = ""
        
        for paragraph in paragraphs:
            paragraph_text = ""
            for sentence in paragraph:
                sentence_text = " ".join(sentence)
                paragraph_text += sentence_text + " "
            full_text += paragraph_text.strip() + "\n\n"
            
        return full_text.strip()
    
    def get_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Mengkonversi data menjadi pandas DataFrame
        
        Args:
            data: List of dictionaries
            
        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame(data)
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """
        Menyimpan data yang sudah dipreprocess
        
        Args:
            data: Data yang sudah dipreprocess
            output_path: Path untuk menyimpan file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Data tersimpan di: {output_path}")
    
    def load_processed_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Memuat data yang sudah dipreprocess
        
        Args:
            file_path: Path ke file data yang sudah dipreprocess
            
        Returns:
            List of dictionaries
        """
        return self.load_jsonl_file(file_path)