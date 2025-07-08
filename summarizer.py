import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import re
from tqdm import tqdm

class GemmaSummarizer:
    """
    Class untuk melakukan summarization menggunakan model Gemma2 9B
    """
    
    def __init__(self, model_name: str = "google/gemma2-9b", device: str = None):
        """
        Inisialisasi summarizer dengan model Gemma2 9B
        
        Args:
            model_name: Nama model yang akan digunakan
            device: Device untuk inference (cuda/cpu)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Menggunakan device: {self.device}")
        
        # Load tokenizer dan model
        print("Memuat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Memuat model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Set padding token jika belum ada
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model berhasil dimuat!")
    
    def generate_summary(self, text: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate summary untuk teks input
        
        Args:
            text: Teks yang akan diringkas
            max_length: Panjang maksimal summary
            temperature: Temperature untuk sampling
            
        Returns:
            Summary yang dihasilkan
        """
        # Prompt template untuk summarization dalam bahasa Indonesia
        prompt = f"""Berikut adalah artikel berita dalam bahasa Indonesia. Buatlah ringkasan yang singkat dan informatif dalam bahasa Indonesia.

Artikel:
{text}

Ringkasan:"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract summary (hapus prompt)
        summary = generated_text[len(prompt):].strip()
        
        return summary
    
    def batch_summarize(self, texts: List[str], max_length: int = 512, temperature: float = 0.7) -> List[str]:
        """
        Generate summary untuk batch teks
        
        Args:
            texts: List of teks yang akan diringkas
            max_length: Panjang maksimal summary
            temperature: Temperature untuk sampling
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for text in tqdm(texts, desc="Generating summaries"):
            try:
                summary = self.generate_summary(text, max_length, temperature)
                summaries.append(summary)
            except Exception as e:
                print(f"Error saat generate summary: {e}")
                summaries.append("")
                
        return summaries
    
    def summarize_dataset(self, dataset: List[Dict[str, Any]], max_length: int = 512, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate summary untuk seluruh dataset
        
        Args:
            dataset: Dataset yang berisi teks berita
            max_length: Panjang maksimal summary
            temperature: Temperature untuk sampling
            
        Returns:
            Dataset dengan summary yang dihasilkan
        """
        results = []
        
        for item in tqdm(dataset, desc="Processing dataset"):
            try:
                # Generate summary
                generated_summary = self.generate_summary(
                    item['text'], 
                    max_length=max_length, 
                    temperature=temperature
                )
                
                # Tambahkan hasil ke item
                result_item = item.copy()
                result_item['generated_summary'] = generated_summary
                results.append(result_item)
                
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
                result_item = item.copy()
                result_item['generated_summary'] = ""
                results.append(result_item)
                
        return results
    
    def clean_summary(self, summary: str) -> str:
        """
        Membersihkan summary dari karakter yang tidak diinginkan
        
        Args:
            summary: Summary yang akan dibersihkan
            
        Returns:
            Summary yang sudah dibersihkan
        """
        # Hapus karakter khusus
        summary = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', summary)
        
        # Hapus spasi berlebih
        summary = re.sub(r'\s+', ' ', summary)
        
        # Hapus baris kosong
        summary = summary.strip()
        
        return summary