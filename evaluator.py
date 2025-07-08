import numpy as np
from typing import List, Dict, Any, Tuple
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from bert_score import score as bert_score_func
import pandas as pd
from tqdm import tqdm

class SummarizationEvaluator:
    """
    Class untuk mengevaluasi hasil summarization menggunakan ROUGE, BLEU, dan BERTScore
    """
    
    def __init__(self, lang: str = "id"):
        """
        Inisialisasi evaluator
        
        Args:
            lang: Bahasa untuk evaluasi (default: id untuk Indonesia)
        """
        self.lang = lang
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print("Evaluator berhasil diinisialisasi!")
    
    def calculate_rouge_scores(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Menghitung skor ROUGE
        
        Args:
            references: List of reference summaries (ground truth)
            predictions: List of predicted summaries
            
        Returns:
            Dictionary berisi skor ROUGE
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in tqdm(zip(references, predictions), desc="Calculating ROUGE scores", total=len(references)):
            if not pred.strip():  # Skip empty predictions
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
                continue
                
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores)
        }
    
    def calculate_bleu_score(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Menghitung skor BLEU
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            
        Returns:
            Dictionary berisi skor BLEU
        """
        # Filter out empty predictions
        valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if pred.strip()]
        
        if not valid_pairs:
            return {'bleu': 0.0}
        
        refs, preds = zip(*valid_pairs)
        
        # Convert to list of lists for BLEU calculation
        refs_list = [[ref] for ref in refs]
        
        # Calculate BLEU
        bleu = BLEU()
        result = bleu.corpus_score(preds, refs_list)
        
        return {
            'bleu': result.score,
            'bleu_details': {
                'precisions': result.precisions,
                'bp': result.bp,
                'sys_len': result.sys_len,
                'ref_len': result.ref_len
            }
        }
    
    def calculate_bertscore(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Menghitung skor BERTScore
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            
        Returns:
            Dictionary berisi skor BERTScore
        """
        # Filter out empty predictions
        valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if pred.strip()]
        
        if not valid_pairs:
            return {'bertscore': 0.0}
        
        refs, preds = zip(*valid_pairs)
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score_func(
                preds, 
                refs, 
                lang=self.lang, 
                verbose=True,
                batch_size=16
            )
            
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item(),
                'bertscore_precision_std': P.std().item(),
                'bertscore_recall_std': R.std().item(),
                'bertscore_f1_std': F1.std().item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
                'bertscore_precision_std': 0.0,
                'bertscore_recall_std': 0.0,
                'bertscore_f1_std': 0.0
            }
    
    def evaluate_summaries(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """
        Evaluasi lengkap menggunakan semua metrik
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            
        Returns:
            Dictionary berisi semua skor evaluasi
        """
        print("Memulai evaluasi summarization...")
        
        # Calculate ROUGE scores
        print("Menghitung skor ROUGE...")
        rouge_scores = self.calculate_rouge_scores(references, predictions)
        
        # Calculate BLEU score
        print("Menghitung skor BLEU...")
        bleu_scores = self.calculate_bleu_score(references, predictions)
        
        # Calculate BERTScore
        print("Menghitung skor BERTScore...")
        bert_scores = self.calculate_bertscore(references, predictions)
        
        # Combine all scores
        results = {
            'rouge': rouge_scores,
            'bleu': bleu_scores,
            'bertscore': bert_scores,
            'summary': {
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'bleu': bleu_scores['bleu'],
                'bertscore_f1': bert_scores['bertscore_f1']
            }
        }
        
        return results
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluasi dataset yang sudah berisi generated summaries
        
        Args:
            dataset: Dataset dengan field 'summary' dan 'generated_summary'
            
        Returns:
            Dictionary berisi hasil evaluasi
        """
        references = [item['summary'] for item in dataset]
        predictions = [item['generated_summary'] for item in dataset]
        
        return self.evaluate_summaries(references, predictions)
    
    def print_results(self, results: Dict[str, Any]):
        """
        Print hasil evaluasi dengan format yang rapi
        
        Args:
            results: Hasil evaluasi dari evaluate_summaries
        """
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
        
        print("\n" + "="*50)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Menyimpan hasil evaluasi ke file
        
        Args:
            results: Hasil evaluasi
            output_path: Path untuk menyimpan file
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Hasil evaluasi tersimpan di: {output_path}")
    
    def create_evaluation_dataframe(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Membuat DataFrame untuk analisis detail hasil evaluasi
        
        Args:
            dataset: Dataset dengan generated summaries
            
        Returns:
            DataFrame dengan skor per item
        """
        evaluation_data = []
        
        for item in dataset:
            ref = item['summary']
            pred = item['generated_summary']
            
            if not pred.strip():
                rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            else:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                }
            
            evaluation_data.append({
                'id': item['id'],
                'category': item['category'],
                'source': item['source'],
                'reference_length': len(ref.split()),
                'prediction_length': len(pred.split()),
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            })
        
        return pd.DataFrame(evaluation_data)