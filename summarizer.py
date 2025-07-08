import torch
import transformers
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
import time

class NewsSummarizer:
    """
    Kelas untuk melakukan summarization berita menggunakan Gemma2 9B Sahabat AI
    """
    
    def __init__(self, model_id: str = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"):
        """
        Inisialisasi model summarizer
        
        Args:
            model_id (str): ID model yang akan digunakan
        """
        self.model_id = model_id
        self.pipeline = None
        self.terminators = None
        self.is_loaded = False
        
    def load_model(self, device_map: str = "auto"):
        """
        Memuat model ke memory
        
        Args:
            device_map (str): Device mapping untuk model
        """
        print(f"Loading model {self.model_id}...")
        
        try:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map=device_map,
            )
            
            self.terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            self.is_loaded = True
            print("Model berhasil dimuat!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_summarization_prompt(self, text: str, max_length: int = 2000) -> str:
        """
        Membuat prompt untuk summarization
        
        Args:
            text (str): Teks berita yang akan diringkas
            max_length (int): Panjang maksimal teks input
            
        Returns:
            str: Prompt yang sudah diformat
        """
        # Potong teks jika terlalu panjang
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = f"""Buatlah ringkasan singkat dari berita berikut dalam bahasa Indonesia. Ringkasan harus mencakup informasi penting dan ditulis dalam 2-3 kalimat.

Berita:
{text}

Ringkasan:"""
        
        return prompt
    
    def generate_summary(self, text: str, max_new_tokens: int = 256) -> str:
        """
        Menghasilkan summary dari teks berita
        
        Args:
            text (str): Teks berita
            max_new_tokens (int): Maksimal token baru yang dihasilkan
            
        Returns:
            str: Summary yang dihasilkan
        """
        if not self.is_loaded:
            raise ValueError("Model belum dimuat. Panggil load_model() terlebih dahulu.")
        
        prompt = self.create_summarization_prompt(text)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Ekstrak response dari output
            generated_text = outputs[0]["generated_text"][-1]["content"]
            
            # Bersihkan response
            summary = self.clean_summary(generated_text)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def clean_summary(self, summary: str) -> str:
        """
        Membersihkan summary dari karakter yang tidak diinginkan
        
        Args:
            summary (str): Summary mentah
            
        Returns:
            str: Summary yang sudah dibersihkan
        """
        # Hapus prefix yang tidak diinginkan
        if "Ringkasan:" in summary:
            summary = summary.split("Ringkasan:")[-1]
        
        # Bersihkan whitespace
        summary = summary.strip()
        
        # Hapus karakter khusus di awal dan akhir
        summary = summary.strip('"').strip("'").strip()
        
        return summary
    
    def batch_summarize(self, data: List[Dict[str, Any]], 
                       output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Melakukan summarization batch untuk dataset
        
        Args:
            data (List[Dict[str, Any]]): Dataset yang akan diringkas
            output_file (Optional[str]): File untuk menyimpan hasil
            
        Returns:
            List[Dict[str, Any]]: Dataset dengan summary yang ditambahkan
        """
        if not self.is_loaded:
            raise ValueError("Model belum dimuat. Panggil load_model() terlebih dahulu.")
        
        results = []
        
        print(f"Memulai summarization untuk {len(data)} item...")
        
        for i, item in enumerate(tqdm(data, desc="Summarizing")):
            try:
                # Generate summary
                generated_summary = self.generate_summary(item['full_text'])
                
                # Tambahkan summary ke item
                item_with_summary = item.copy()
                item_with_summary['generated_summary'] = generated_summary
                
                results.append(item_with_summary)
                
                # Simpan progress secara berkala
                if output_file and (i + 1) % 10 == 0:
                    self.save_results(results, output_file)
                
                # Jeda kecil untuk menghindari overload
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                # Tambahkan item tanpa summary jika gagal
                item_with_summary = item.copy()
                item_with_summary['generated_summary'] = ""
                results.append(item_with_summary)
                continue
        
        # Simpan hasil akhir
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Menyimpan hasil summarization
        
        Args:
            results (List[Dict[str, Any]]): Hasil summarization
            output_file (str): Path file output
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def evaluate_summary(self, generated_summary: str, gold_summary: str) -> Dict[str, float]:
        """
        Evaluasi summary yang dihasilkan (metrik sederhana)
        
        Args:
            generated_summary (str): Summary yang dihasilkan model
            gold_summary (str): Summary referensi
            
        Returns:
            Dict[str, float]: Metrik evaluasi
        """
        # Hitung panjang summary
        gen_length = len(generated_summary.split())
        gold_length = len(gold_summary.split())
        
        # Hitung compression ratio
        compression_ratio = gen_length / gold_length if gold_length > 0 else 0
        
        # Hitung overlap kata sederhana
        gen_words = set(generated_summary.lower().split())
        gold_words = set(gold_summary.lower().split())
        
        if len(gold_words) > 0:
            word_overlap = len(gen_words.intersection(gold_words)) / len(gold_words)
        else:
            word_overlap = 0
        
        return {
            'compression_ratio': compression_ratio,
            'word_overlap': word_overlap,
            'generated_length': gen_length,
            'gold_length': gold_length
        }