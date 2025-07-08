# Evaluasi Text Summarization dengan Gemma2 9B

Proyek ini berisi implementasi lengkap untuk evaluasi model text summarization menggunakan dataset berita bahasa Indonesia dan model Gemma2 9B dari Google.

## 🎯 Tujuan

Mengevaluasi performa model Gemma2 9B dalam melakukan text summarization pada dataset berita bahasa Indonesia menggunakan metrik evaluasi standar: ROUGE, BLEU, dan BERTScore.

## 📁 Struktur Proyek

```
├── data/                          # Direktori dataset (buat manual)
│   ├── train.01.jsonl
│   ├── train.02.jsonl
│   ├── train.03.jsonl
│   ├── train.04.jsonl
│   └── train.05.jsonl
├── results/                       # Hasil evaluasi (auto-generated)
├── data_loader.py                 # Modul untuk memuat dataset
├── summarizer.py                  # Modul untuk generate summary
├── evaluator.py                   # Modul untuk evaluasi metrik
├── visualizer.py                  # Modul untuk visualisasi
├── main.py                        # Script utama
├── requirements.txt               # Dependencies
└── README.md                      # Dokumentasi
```

## 🚀 Instalasi

1. **Clone repository**
```bash
git clone <repository-url>
cd text-summarization-evaluation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Siapkan dataset**
   - Buat direktori `data/`
   - Letakkan file dataset `train.XX.jsonl` di dalamnya
   - Format dataset harus sesuai dengan contoh yang diberikan

## 📊 Format Dataset

Dataset harus dalam format JSONL dengan struktur sebagai berikut:

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

## 🔧 Penggunaan

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

### 2. Menggunakan Jupyter Notebook

Buka file `text_summarization_evaluation.ipynb` di Jupyter Notebook dan jalankan cell secara berurutan.

### 3. Menggunakan Modul Secara Terpisah

```python
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

## 📈 Metrik Evaluasi

### 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Overlap unigram antara reference dan prediction
- **ROUGE-2**: Overlap bigram antara reference dan prediction  
- **ROUGE-L**: Longest Common Subsequence

### 2. BLEU (Bilingual Evaluation Understudy)
- Mengukur akurasi n-gram dalam prediction terhadap reference

### 3. BERTScore
- Menggunakan contextual embeddings dari BERT untuk evaluasi semantic similarity
- Menghasilkan skor precision, recall, dan F1

## 📊 Output

Setelah menjalankan evaluasi, akan dihasilkan file-file berikut di direktori `results/`:

### File Data
- `evaluation_results.json` - Hasil evaluasi dalam format JSON
- `results_with_summaries.jsonl` - Dataset dengan generated summaries
- `evaluation_dataframe.csv` - DataFrame untuk analisis detail
- `final_report.json` - Laporan lengkap

### File Visualisasi
- `metrics_comparison.png` - Perbandingan semua metrik
- `category_analysis.png` - Analisis berdasarkan kategori berita
- `source_analysis.png` - Analisis berdasarkan sumber berita
- `length_analysis.png` - Analisis hubungan panjang teks dengan performa
- `summary_report.png` - Laporan ringkasan lengkap

## ⚙️ Konfigurasi

### Parameter Model
- `model_name`: Nama model yang digunakan (default: "google/gemma2-9b")
- `device`: Device untuk inference ("cuda", "cpu", atau None untuk auto-detect)
- `max_length`: Panjang maksimal summary (default: 512)
- `temperature`: Temperature untuk sampling (default: 0.7)

### Parameter Evaluasi
- `sample_size`: Jumlah sampel untuk evaluasi (None untuk semua)
- `data_dir`: Direktori dataset
- `output_dir`: Direktori untuk menyimpan hasil

## 🔍 Analisis Hasil

### 1. Performa Keseluruhan
- Skor ROUGE, BLEU, dan BERTScore untuk seluruh dataset
- Analisis statistik (mean, std, min, max)

### 2. Analisis Kategori
- Performa model per kategori berita
- Identifikasi kategori yang mudah/sulit diringkas

### 3. Analisis Sumber
- Performa model per sumber berita
- Analisis gaya penulisan yang mempengaruhi hasil

### 4. Analisis Panjang
- Hubungan panjang teks dengan performa
- Optimal length ratio untuk summary

## 🛠️ Troubleshooting

### 1. Error: "CUDA out of memory"
```bash
# Gunakan CPU atau kurangi batch size
python main.py --device cpu
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
```

## 📝 Contoh Hasil

### Output Terminal
```
============================================================
EVALUASI TEXT SUMMARIZATION DENGAN GEMMA2 9B
============================================================

1. MEMUAT DATASET
------------------------------
Dataset berhasil dimuat: 1000 artikel

2. SAMPLING DATA
------------------------------
Menggunakan semua 1000 artikel

3. INISIALISASI MODEL
------------------------------
Model berhasil diinisialisasi!

4. GENERATE SUMMARIES
------------------------------
Berhasil generate 1000 summaries

5. EVALUASI HASIL
------------------------------
==================================================
HASIL EVALUASI SUMMARIZATION
==================================================

ROUGE Scores:
  ROUGE-1: 0.3245 ± 0.1234
  ROUGE-2: 0.1567 ± 0.0891
  ROUGE-L: 0.2987 ± 0.1156

BLEU Score:
  BLEU: 0.2345

BERTScore:
  Precision: 0.3456 ± 0.0789
  Recall: 0.3123 ± 0.0823
  F1: 0.3287 ± 0.0801
==================================================
```

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Kontak

Jika ada pertanyaan atau masalah, silakan buat issue di repository ini.

## 🙏 Acknowledgments

- Dataset berita bahasa Indonesia
- Model Gemma2 9B dari Google
- Library evaluasi: ROUGE, BLEU, BERTScore
- Transformers library dari Hugging Face