# Panduan Penggunaan Evaluasi Text Summarization

Panduan lengkap untuk menggunakan sistem evaluasi text summarization dengan model Gemma2 9B.

## 📋 Daftar Isi

1. [Instalasi](#instalasi)
2. [Persiapan Dataset](#persiapan-dataset)
3. [Penggunaan Dasar](#penggunaan-dasar)
4. [Penggunaan Lanjutan](#penggunaan-lanjutan)
5. [Analisis Hasil](#analisis-hasil)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd text-summarization-evaluation
```

### 2. Install Dependencies
```bash
# Install semua dependencies
pip install -r requirements.txt

# Atau install dengan conda
conda env create -f environment.yml
conda activate text-summarization-eval
```

### 3. Verifikasi Instalasi
```bash
# Test instalasi
python demo_simple.py

# Atau test dengan dependencies lengkap
python example_usage.py
```

## 📊 Persiapan Dataset

### Format Dataset
Dataset harus dalam format JSONL dengan struktur:

```json
{
  "category": "kategori_berita",
  "gold_labels": [[false, true], [true, true], ...],
  "id": "unique_id",
  "paragraphs": [[["token1", "token2", ...], ["token1", "token2", ...]], ...],
  "source": "sumber_berita",
  "source_url": "url_berita",
  "summary": [["token1", "token2", ...], ["token1", "token2", ...], ...]
}
```

### Struktur Direktori
```
project/
├── data/
│   ├── train.01.jsonl
│   ├── train.02.jsonl
│   ├── train.03.jsonl
│   ├── train.04.jsonl
│   └── train.05.jsonl
├── results/          # Auto-generated
└── ...
```

### Contoh Dataset
File `data/train.01.jsonl` sudah disediakan sebagai contoh dengan 2 artikel berita bahasa Indonesia.

## 🔧 Penggunaan Dasar

### 1. Menggunakan Script Utama
```bash
# Evaluasi dengan semua artikel
python main.py

# Evaluasi dengan sampel tertentu
python main.py --sample_size 100

# Menggunakan model berbeda
python main.py --model_name "google/gemma2-9b-it"

# Mengatur parameter generation
python main.py --max_length 256 --temperature 0.8

# Menyimpan hasil di direktori tertentu
python main.py --output_dir "my_results"
```

### 2. Menggunakan Demo Sederhana
```bash
# Demo tanpa dependencies eksternal
python demo_simple.py
```

### 3. Menggunakan Jupyter Notebook
```python
# Import modul
from data_loader import NewsDatasetLoader
from summarizer import GemmaSummarizer
from evaluator import SummarizationEvaluator
from visualizer import SummarizationVisualizer

# Load dataset
data_loader = NewsDatasetLoader(data_dir="data")
raw_data = data_loader.load_all_train_files()
processed_data = data_loader.preprocess_data(raw_data)

# Initialize model
summarizer = GemmaSummarizer(model_name="google/gemma2-9b")

# Generate summaries
results = summarizer.summarize_dataset(processed_data)

# Evaluate results
evaluator = SummarizationEvaluator()
evaluation_results = evaluator.evaluate_dataset(results)

# Create visualizations
visualizer = SummarizationVisualizer()
visualizer.plot_metrics_comparison(evaluation_results)
```

## ⚙️ Penggunaan Lanjutan

### 1. Konfigurasi Model
```python
# Konfigurasi model
summarizer = GemmaSummarizer(
    model_name="google/gemma2-9b",  # Model yang digunakan
    device="cuda"  # Device: "cuda", "cpu", atau None untuk auto-detect
)

# Parameter generation
results = summarizer.summarize_dataset(
    dataset=processed_data,
    max_length=512,      # Panjang maksimal summary
    temperature=0.7      # Temperature untuk sampling
)
```

### 2. Evaluasi Kustom
```python
# Evaluasi dengan metrik tertentu
evaluator = SummarizationEvaluator(lang="id")

# Evaluasi manual
references = [item['summary'] for item in dataset]
predictions = [item['generated_summary'] for item in dataset]

results = evaluator.evaluate_summaries(references, predictions)
evaluator.print_results(results)
```

### 3. Visualisasi Kustom
```python
# Buat visualisasi khusus
visualizer = SummarizationVisualizer()

# Plot perbandingan metrik
visualizer.plot_metrics_comparison(evaluation_results, save_path="my_plot.png")

# Plot analisis kategori
evaluation_df = evaluator.create_evaluation_dataframe(dataset)
visualizer.plot_category_analysis(evaluation_df, save_path="category_analysis.png")

# Plot analisis sumber
visualizer.plot_source_analysis(evaluation_df, save_path="source_analysis.png")

# Plot analisis panjang
visualizer.plot_length_analysis(evaluation_df, save_path="length_analysis.png")

# Laporan lengkap
visualizer.create_summary_report(evaluation_results, evaluation_df, save_path="full_report.png")
```

## 📈 Analisis Hasil

### 1. File Output
Setelah menjalankan evaluasi, akan dihasilkan file-file berikut di direktori `results/`:

#### File Data
- `evaluation_results.json` - Hasil evaluasi dalam format JSON
- `results_with_summaries.jsonl` - Dataset dengan generated summaries
- `evaluation_dataframe.csv` - DataFrame untuk analisis detail
- `final_report.json` - Laporan lengkap

#### File Visualisasi
- `metrics_comparison.png` - Perbandingan semua metrik
- `category_analysis.png` - Analisis berdasarkan kategori berita
- `source_analysis.png` - Analisis berdasarkan sumber berita
- `length_analysis.png` - Analisis hubungan panjang teks dengan performa
- `summary_report.png` - Laporan ringkasan lengkap

### 2. Interpretasi Metrik

#### ROUGE Scores
- **ROUGE-1**: Overlap unigram antara reference dan prediction (0-1)
- **ROUGE-2**: Overlap bigram antara reference dan prediction (0-1)
- **ROUGE-L**: Longest Common Subsequence (0-1)

#### BLEU Score
- Mengukur akurasi n-gram dalam prediction terhadap reference (0-1)

#### BERTScore
- **Precision**: Akurasi semantic dari prediction
- **Recall**: Kelengkapan semantic dari prediction
- **F1**: Harmonic mean dari precision dan recall

### 3. Analisis Performa
```python
# Analisis performa per kategori
category_performance = evaluation_df.groupby('category').agg({
    'rouge1': ['mean', 'std', 'count'],
    'rouge2': ['mean', 'std'],
    'rougeL': ['mean', 'std']
}).round(3)

print(category_performance)

# Analisis korelasi
correlation_matrix = evaluation_df[['rouge1', 'rouge2', 'rougeL', 'reference_length', 'prediction_length']].corr()
print(correlation_matrix.round(3))
```

## 🛠️ Troubleshooting

### 1. Error: "CUDA out of memory"
```bash
# Gunakan CPU
python main.py --device cpu

# Atau kurangi batch size di summarizer.py
```

### 2. Error: "Model not found"
```bash
# Pastikan koneksi internet stabil
# Atau gunakan model lokal
python main.py --model_name "path/to/local/model"
```

### 3. Error: "Dataset not found"
```bash
# Pastikan direktori data/ ada dan berisi file train.XX.jsonl
mkdir data
# Letakkan file dataset di direktori data/
```

### 4. Error: "Dependencies not found"
```bash
# Install ulang dependencies
pip install -r requirements.txt

# Atau install satu per satu
pip install torch transformers datasets rouge-score sacrebleu bert-score
```

### 5. Error: "Import error"
```bash
# Pastikan Python path sudah benar
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Atau gunakan setup.py
pip install -e .
```

## ❓ FAQ

### Q: Apakah saya perlu GPU untuk menjalankan evaluasi?
A: Tidak wajib, tetapi GPU akan mempercepat proses generate summary. Model dapat berjalan di CPU dengan waktu yang lebih lama.

### Q: Berapa lama waktu yang dibutuhkan untuk evaluasi?
A: Tergantung pada:
- Jumlah artikel (1 artikel ≈ 10-30 detik di CPU, 2-5 detik di GPU)
- Panjang teks
- Model yang digunakan

### Q: Apakah saya bisa menggunakan model lain?
A: Ya, Anda dapat menggunakan model lain yang kompatibel dengan Hugging Face Transformers. Ganti parameter `model_name` saat inisialisasi.

### Q: Bagaimana cara menambah metrik evaluasi baru?
A: Tambahkan metrik baru di class `SummarizationEvaluator` dan update fungsi `evaluate_summaries()`.

### Q: Apakah hasil evaluasi dapat dipercaya?
A: Hasil evaluasi menggunakan metrik standar (ROUGE, BLEU, BERTScore) yang sudah divalidasi secara akademis. Namun, interpretasi hasil tetap memerlukan analisis kontekstual.

### Q: Bagaimana cara menggunakan dataset saya sendiri?
A: Konversi dataset Anda ke format JSONL yang sesuai dengan struktur yang telah ditentukan.

## 📞 Dukungan

Jika mengalami masalah atau memiliki pertanyaan:

1. Periksa bagian [Troubleshooting](#troubleshooting)
2. Baca [FAQ](#faq)
3. Buat issue di repository GitHub
4. Hubungi tim pengembang

## 📚 Referensi

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- [Gemma2: Open Models for Responsible AI](https://ai.google.dev/gemma)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)