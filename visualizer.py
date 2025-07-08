import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SummarizationVisualizer:
    """
    Class untuk visualisasi hasil evaluasi summarization
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Inisialisasi visualizer
        
        Args:
            figsize: Ukuran default figure
        """
        self.figsize = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
    def plot_metrics_comparison(self, results: Dict[str, Any], save_path: str = None):
        """
        Plot perbandingan metrik evaluasi
        
        Args:
            results: Hasil evaluasi dari evaluator
            save_path: Path untuk menyimpan plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Perbandingan Metrik Evaluasi Summarization', fontsize=16, fontweight='bold')
        
        # ROUGE Scores
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        rouge_values = [results['rouge'][metric] for metric in rouge_metrics]
        rouge_stds = [results['rouge'][f'{metric}_std'] for metric in rouge_metrics]
        
        axes[0, 0].bar(rouge_metrics, rouge_values, yerr=rouge_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('ROUGE Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(rouge_values):
            axes[0, 0].text(i, v + rouge_stds[i] + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # BLEU Score
        bleu_value = results['bleu']['bleu']
        axes[0, 1].bar(['BLEU'], [bleu_value], alpha=0.7, color='orange')
        axes[0, 1].set_title('BLEU Score')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, max(bleu_value * 1.2, 0.1))
        axes[0, 1].text(0, bleu_value + 0.001, f'{bleu_value:.3f}', ha='center', va='bottom')
        
        # BERTScore
        bert_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        bert_values = [results['bertscore'][metric] for metric in bert_metrics]
        bert_stds = [results['bertscore'][f'{metric}_std'] for metric in bert_metrics]
        
        axes[1, 0].bar(bert_metrics, bert_values, yerr=bert_stds, capsize=5, alpha=0.7, color='green')
        axes[1, 0].set_title('BERTScore')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(bert_values):
            axes[1, 0].text(i, v + bert_stds[i] + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Summary metrics
        summary_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore-F1']
        summary_values = [
            results['summary']['rouge1'],
            results['summary']['rouge2'],
            results['summary']['rougeL'],
            results['summary']['bleu'],
            results['summary']['bertscore_f1']
        ]
        
        axes[1, 1].bar(summary_metrics, summary_values, alpha=0.7, color='purple')
        axes[1, 1].set_title('Ringkasan Semua Metrik')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(summary_values):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot tersimpan di: {save_path}")
        
        plt.show()
    
    def plot_category_analysis(self, evaluation_df: pd.DataFrame, save_path: str = None):
        """
        Plot analisis berdasarkan kategori berita
        
        Args:
            evaluation_df: DataFrame hasil evaluasi
            save_path: Path untuk menyimpan plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisis Hasil Berdasarkan Kategori Berita', fontsize=16, fontweight='bold')
        
        # ROUGE scores by category
        category_rouge = evaluation_df.groupby('category')[['rouge1', 'rouge2', 'rougeL']].mean()
        
        category_rouge.plot(kind='bar', ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('ROUGE Scores per Kategori')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Length comparison by category
        length_comparison = evaluation_df.groupby('category')[['reference_length', 'prediction_length']].mean()
        
        length_comparison.plot(kind='bar', ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title('Panjang Summary per Kategori')
        axes[0, 1].set_ylabel('Jumlah Kata')
        axes[0, 1].legend(['Reference', 'Prediction'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Distribution of ROUGE-1 scores
        axes[1, 0].hist(evaluation_df['rouge1'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribusi ROUGE-1 Scores')
        axes[1, 0].set_xlabel('ROUGE-1 Score')
        axes[1, 0].set_ylabel('Frekuensi')
        axes[1, 0].axvline(evaluation_df['rouge1'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {evaluation_df["rouge1"].mean():.3f}')
        axes[1, 0].legend()
        
        # Box plot ROUGE scores by category
        rouge_data = []
        categories = []
        for category in evaluation_df['category'].unique():
            cat_data = evaluation_df[evaluation_df['category'] == category]['rouge1']
            rouge_data.extend(cat_data.tolist())
            categories.extend([category] * len(cat_data))
        
        rouge_df = pd.DataFrame({'category': categories, 'rouge1': rouge_data})
        sns.boxplot(data=rouge_df, x='category', y='rouge1', ax=axes[1, 1])
        axes[1, 1].set_title('Box Plot ROUGE-1 per Kategori')
        axes[1, 1].set_ylabel('ROUGE-1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot tersimpan di: {save_path}")
        
        plt.show()
    
    def plot_source_analysis(self, evaluation_df: pd.DataFrame, save_path: str = None):
        """
        Plot analisis berdasarkan sumber berita
        
        Args:
            evaluation_df: DataFrame hasil evaluasi
            save_path: Path untuk menyimpan plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisis Hasil Berdasarkan Sumber Berita', fontsize=16, fontweight='bold')
        
        # ROUGE scores by source
        source_rouge = evaluation_df.groupby('source')[['rouge1', 'rouge2', 'rougeL']].mean()
        
        source_rouge.plot(kind='bar', ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('ROUGE Scores per Sumber')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Number of articles per source
        source_counts = evaluation_df['source'].value_counts()
        axes[0, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Distribusi Artikel per Sumber')
        
        # Average length by source
        source_length = evaluation_df.groupby('source')[['reference_length', 'prediction_length']].mean()
        
        source_length.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Panjang Summary per Sumber')
        axes[1, 0].set_ylabel('Jumlah Kata')
        axes[1, 0].legend(['Reference', 'Prediction'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Heatmap correlation between metrics
        metrics_corr = evaluation_df[['rouge1', 'rouge2', 'rougeL', 'reference_length', 'prediction_length']].corr()
        
        sns.heatmap(metrics_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Korelasi Antar Metrik')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot tersimpan di: {save_path}")
        
        plt.show()
    
    def plot_length_analysis(self, evaluation_df: pd.DataFrame, save_path: str = None):
        """
        Plot analisis hubungan panjang teks dengan performa
        
        Args:
            evaluation_df: DataFrame hasil evaluasi
            save_path: Path untuk menyimpan plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisis Hubungan Panjang Teks dengan Performa', fontsize=16, fontweight='bold')
        
        # ROUGE-1 vs Reference Length
        axes[0, 0].scatter(evaluation_df['reference_length'], evaluation_df['rouge1'], alpha=0.6)
        axes[0, 0].set_xlabel('Panjang Reference (kata)')
        axes[0, 0].set_ylabel('ROUGE-1 Score')
        axes[0, 0].set_title('ROUGE-1 vs Panjang Reference')
        
        # Add trend line
        z = np.polyfit(evaluation_df['reference_length'], evaluation_df['rouge1'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(evaluation_df['reference_length'], p(evaluation_df['reference_length']), "r--", alpha=0.8)
        
        # ROUGE-1 vs Prediction Length
        axes[0, 1].scatter(evaluation_df['prediction_length'], evaluation_df['rouge1'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Panjang Prediction (kata)')
        axes[0, 1].set_ylabel('ROUGE-1 Score')
        axes[0, 1].set_title('ROUGE-1 vs Panjang Prediction')
        
        # Add trend line
        z = np.polyfit(evaluation_df['prediction_length'], evaluation_df['rouge1'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(evaluation_df['prediction_length'], p(evaluation_df['prediction_length']), "r--", alpha=0.8)
        
        # Length ratio vs ROUGE-1
        length_ratio = evaluation_df['prediction_length'] / evaluation_df['reference_length']
        axes[1, 0].scatter(length_ratio, evaluation_df['rouge1'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Rasio Panjang (Prediction/Reference)')
        axes[1, 0].set_ylabel('ROUGE-1 Score')
        axes[1, 0].set_title('ROUGE-1 vs Rasio Panjang')
        
        # Add trend line
        z = np.polyfit(length_ratio, evaluation_df['rouge1'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(length_ratio, p(length_ratio), "r--", alpha=0.8)
        
        # Distribution of length ratios
        axes[1, 1].hist(length_ratio, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Rasio Panjang (Prediction/Reference)')
        axes[1, 1].set_ylabel('Frekuensi')
        axes[1, 1].set_title('Distribusi Rasio Panjang')
        axes[1, 1].axvline(length_ratio.mean(), color='red', linestyle='--', 
                          label=f'Mean: {length_ratio.mean():.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot tersimpan di: {save_path}")
        
        plt.show()
    
    def create_summary_report(self, results: Dict[str, Any], evaluation_df: pd.DataFrame, save_path: str = None):
        """
        Membuat laporan ringkasan lengkap
        
        Args:
            results: Hasil evaluasi
            evaluation_df: DataFrame hasil evaluasi
            save_path: Path untuk menyimpan laporan
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Laporan Lengkap Evaluasi Summarization', fontsize=18, fontweight='bold')
        
        # 1. Overall metrics
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore-F1']
        values = [
            results['summary']['rouge1'],
            results['summary']['rouge2'],
            results['summary']['rougeL'],
            results['summary']['bleu'],
            results['summary']['bertscore_f1']
        ]
        
        bars = axes[0, 0].bar(metrics, values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        axes[0, 0].set_title('Skor Metrik Keseluruhan')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Category performance
        category_perf = evaluation_df.groupby('category')['rouge1'].mean().sort_values(ascending=False)
        category_perf.plot(kind='bar', ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title('ROUGE-1 per Kategori')
        axes[0, 1].set_ylabel('ROUGE-1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Source performance
        source_perf = evaluation_df.groupby('source')['rouge1'].mean().sort_values(ascending=False)
        source_perf.plot(kind='bar', ax=axes[0, 2], alpha=0.7)
        axes[0, 2].set_title('ROUGE-1 per Sumber')
        axes[0, 2].set_ylabel('ROUGE-1 Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Length distribution
        axes[1, 0].hist(evaluation_df['reference_length'], bins=20, alpha=0.7, label='Reference', color='blue')
        axes[1, 0].hist(evaluation_df['prediction_length'], bins=20, alpha=0.7, label='Prediction', color='orange')
        axes[1, 0].set_title('Distribusi Panjang Summary')
        axes[1, 0].set_xlabel('Jumlah Kata')
        axes[1, 0].set_ylabel('Frekuensi')
        axes[1, 0].legend()
        
        # 5. ROUGE scores distribution
        axes[1, 1].hist(evaluation_df['rouge1'], bins=20, alpha=0.7, label='ROUGE-1', color='blue')
        axes[1, 1].hist(evaluation_df['rouge2'], bins=20, alpha=0.7, label='ROUGE-2', color='green')
        axes[1, 1].hist(evaluation_df['rougeL'], bins=20, alpha=0.7, label='ROUGE-L', color='red')
        axes[1, 1].set_title('Distribusi ROUGE Scores')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frekuensi')
        axes[1, 1].legend()
        
        # 6. Length vs Performance scatter
        axes[1, 2].scatter(evaluation_df['reference_length'], evaluation_df['rouge1'], alpha=0.6)
        axes[1, 2].set_xlabel('Panjang Reference')
        axes[1, 2].set_ylabel('ROUGE-1 Score')
        axes[1, 2].set_title('Panjang vs Performa')
        
        # 7. Category counts
        category_counts = evaluation_df['category'].value_counts()
        axes[2, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        axes[2, 0].set_title('Distribusi Kategori')
        
        # 8. Source counts
        source_counts = evaluation_df['source'].value_counts()
        axes[2, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        axes[2, 1].set_title('Distribusi Sumber')
        
        # 9. Statistics table
        axes[2, 2].axis('off')
        stats_text = f"""
        STATISTIK DATASET:
        
        Total Artikel: {len(evaluation_df)}
        Kategori: {evaluation_df['category'].nunique()}
        Sumber: {evaluation_df['source'].nunique()}
        
        RATA-RATA:
        ROUGE-1: {evaluation_df['rouge1'].mean():.3f}
        ROUGE-2: {evaluation_df['rouge2'].mean():.3f}
        ROUGE-L: {evaluation_df['rougeL'].mean():.3f}
        
        Panjang Reference: {evaluation_df['reference_length'].mean():.1f} kata
        Panjang Prediction: {evaluation_df['prediction_length'].mean():.1f} kata
        """
        axes[2, 2].text(0.1, 0.9, stats_text, transform=axes[2, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Laporan tersimpan di: {save_path}")
        
        plt.show()