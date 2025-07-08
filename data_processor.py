import json
from typing import List, Dict, Any
import re

class NewsDataProcessor:
    """
    Kelas untuk memproses dataset berita dalam format yang diberikan
    """
    
    def __init__(self):
        self.processed_data = []
    
    def load_data_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Memuat data dari file JSON yang berisi dataset berita
        
        Args:
            file_path (str): Path ke file JSON
            
        Returns:
            List[Dict[str, Any]]: List data berita yang sudah diproses
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def extract_text_from_paragraphs(self, paragraphs: List[List[List[str]]]) -> str:
        """
        Mengekstrak teks dari struktur paragraphs yang kompleks
        
        Args:
            paragraphs (List[List[List[str]]]): Struktur paragraphs dari dataset
            
        Returns:
            str: Teks yang sudah digabungkan
        """
        full_text = []
        
        for paragraph in paragraphs:
            paragraph_text = []
            for sentence in paragraph:
                # Gabungkan token-token dalam satu kalimat
                sentence_text = ' '.join(sentence)
                paragraph_text.append(sentence_text)
            
            # Gabungkan kalimat dalam satu paragraf
            paragraph_combined = ' '.join(paragraph_text)
            full_text.append(paragraph_combined)
        
        # Gabungkan semua paragraf
        return '\n\n'.join(full_text)
    
    def extract_summary_from_gold_labels(self, paragraphs: List[List[List[str]]], 
                                       gold_labels: List[List[bool]]) -> str:
        """
        Mengekstrak summary berdasarkan gold_labels yang diberikan
        
        Args:
            paragraphs (List[List[List[str]]]): Struktur paragraphs
            gold_labels (List[List[bool]]): Label yang menandakan kalimat penting
            
        Returns:
            str: Summary yang diekstrak
        """
        summary_sentences = []
        
        for i, (paragraph, labels) in enumerate(zip(paragraphs, gold_labels)):
            for j, (sentence, is_important) in enumerate(zip(paragraph, labels)):
                if is_important:
                    sentence_text = ' '.join(sentence)
                    summary_sentences.append(sentence_text)
        
        return ' '.join(summary_sentences)
    
    def process_news_item(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Memproses satu item berita
        
        Args:
            news_item (Dict[str, Any]): Item berita mentah
            
        Returns:
            Dict[str, Any]: Item berita yang sudah diproses
        """
        processed_item = {
            'id': news_item.get('id', ''),
            'category': news_item.get('category', ''),
            'source': news_item.get('source', ''),
            'source_url': news_item.get('source_url', ''),
            'full_text': self.extract_text_from_paragraphs(news_item.get('paragraphs', [])),
            'gold_summary': self.extract_summary_from_gold_labels(
                news_item.get('paragraphs', []), 
                news_item.get('gold_labels', [])
            ),
            'original_summary': self.extract_text_from_paragraphs(news_item.get('summary', []))
        }
        
        return processed_item
    
    def process_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Memproses seluruh dataset
        
        Args:
            data (List[Dict[str, Any]]): Dataset mentah
            
        Returns:
            List[Dict[str, Any]]: Dataset yang sudah diproses
        """
        processed_data = []
        
        for item in data:
            try:
                processed_item = self.process_news_item(item)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
                continue
        
        self.processed_data = processed_data
        return processed_data
    
    def save_processed_data(self, output_path: str):
        """
        Menyimpan data yang sudah diproses
        
        Args:
            output_path (str): Path untuk menyimpan file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def get_sample_data(self, n_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Mengambil sample data untuk testing
        
        Args:
            n_samples (int): Jumlah sample yang diinginkan
            
        Returns:
            List[Dict[str, Any]]: Sample data
        """
        return self.processed_data[:n_samples] if self.processed_data else []