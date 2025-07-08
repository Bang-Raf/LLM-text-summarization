# Sistem Summarization Berita dengan Gemma2 9B Sahabat AI

Sistem ini dirancang untuk melakukan summarization (peringkasan) teks berita dalam bahasa Indonesia menggunakan model Gemma2 9B Sahabat AI. Sistem dapat memproses dataset berita dalam format khusus dan menghasilkan ringkasan yang informatif.

## Fitur Utama

- **Pemrosesan Dataset**: Mendukung format dataset berita dengan struktur paragraphs dan gold_labels
- **Summarization Otomatis**: Menggunakan model Gemma2 9B Sahabat AI untuk menghasilkan ringkasan
- **Evaluasi Kualitas**: Menyediakan metrik evaluasi seperti ROUGE score dan compression ratio
- **Batch Processing**: Dapat memproses dataset dalam jumlah besar
- **Fokus Bahasa Indonesia**: Dioptimalkan untuk teks berita dalam bahasa Indonesia

## Struktur Dataset

Sistem ini mendukung format dataset berita dengan struktur berikut:

```json
{
  "category": "tajuk utama",
  "gold_labels": [[false, true], [true, true], [false, false, false]],
  "id": "unique-id",
  "paragraphs": [
    [["Token", "1", "dari", "kalimat", "1"], ["Token", "1", "dari", "kalimat", "2"]],
    [["Token", "1", "dari", "kalimat", "1", "paragraf", "2"]]
  ],
  "source": "nama sumber",
  "source_url": "https://example.com",
  "summary": [
    [["Token", "summary", "referensi"]]
  ]
}
```

## Instalasi

1. **Clone repository ini**
```bash
git clone <repository-url>
cd news-summarization-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Pastikan versi transformers yang benar**
```bash
pip install transformers==4.45.0
```

## Penggunaan

### 1. Menjalankan Demo

```bash
python main.py
```

Script ini akan:
- Memuat sample data
- Memproses data menggunakan `NewsDataProcessor`
- Memuat model Gemma2 9B Sahabat AI
- Melakukan test summarization
- Menampilkan hasil dan metrik evaluasi

### 2. Menggunakan dengan Dataset Sendiri

```python
from data_processor import NewsDataProcessor
from summarizer import NewsSummarizer

# 1. Proses dataset
processor = NewsDataProcessor()
raw_data = processor.load_data_from_file("your_dataset.jsonl")
processed_data = processor.process_dataset(raw_data)

# 2. Load model dan lakukan summarization
summarizer = NewsSummarizer()
summarizer.load_model()

# 3. Generate summary untuk satu item
summary = summarizer.generate_summary(processed_data[0]['full_text'])

# 4. Batch processing
results = summarizer.batch_summarize(processed_data, "output.jsonl")
```

### 3. Evaluasi Hasil

```python
from utils import evaluate_summaries, print_evaluation_results

# Evaluasi batch results
metrics = evaluate_summaries(results)
print_evaluation_results(metrics)
```

## Komponen Sistem

### 1. `data_processor.py`
- **NewsDataProcessor**: Kelas untuk memproses dataset berita
- Mengekstrak teks dari struktur paragraphs yang kompleks
- Membuat gold summary berdasarkan gold_labels
- Menyimpan data yang sudah diproses

### 2. `summarizer.py`
- **NewsSummarizer**: Kelas utama untuk summarization
- Menggunakan model Gemma2 9B Sahabat AI
- Membuat prompt yang dioptimalkan untuk bahasa Indonesia
- Mendukung batch processing dan evaluasi

### 3. `utils.py`
- Fungsi helper untuk evaluasi (ROUGE scores)
- Utility untuk loading/saving JSONL files
- Text cleaning dan preprocessing

### 4. `main.py`
- Script demo yang menunjukkan penggunaan lengkap sistem
- Menggunakan sample data untuk testing

## Konfigurasi Model

### Model yang Digunakan
- **Model**: `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct`
- **Framework**: Transformers 4.45.0
- **Precision**: bfloat16
- **Device**: Auto (CPU/GPU)

### Parameter Summarization
- **max_new_tokens**: 256
- **temperature**: 0.7
- **top_p**: 0.9
- **max_length**: 2000 (untuk input text)

## Metrik Evaluasi

Sistem menyediakan beberapa metrik evaluasi:

1. **ROUGE Scores**: ROUGE-1 dan ROUGE-2 untuk mengukur overlap kata
2. **Compression Ratio**: Rasio panjang summary terhadap teks asli
3. **Success Rate**: Persentase berhasil diringkas
4. **Summary Length**: Panjang rata-rata summary yang dihasilkan

## Persyaratan Sistem

### Minimum Requirements
- **RAM**: 16GB (untuk model 9B)
- **Storage**: 20GB free space
- **Python**: 3.8+

### Recommended
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU dengan 8GB+ VRAM
- **Storage**: SSD dengan 50GB+ free space

## Troubleshooting

### Error: "Model belum dimuat"
```python
# Pastikan memanggil load_model() sebelum generate_summary()
summarizer = NewsSummarizer()
summarizer.load_model()  # Panggil ini dulu
summary = summarizer.generate_summary(text)
```

### Error: "Out of memory"
- Kurangi batch size
- Gunakan CPU jika GPU memory tidak cukup
- Tutup aplikasi lain yang menggunakan memory

### Error: "Transformers version"
```bash
pip install transformers==4.45.0
```

## Contoh Output

```
=== Sistem Summarization Berita dengan Gemma2 9B Sahabat AI ===

1. Memproses data...
Berhasil memproses 1 item berita

Sample berita:
ID: 1501893029-lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun
Kategori: tajuk utama
Teks lengkap: Jakarta, CNN Indonesia - - Dokter Ryan Thamrin, yang terkenal lewat acara Dokter Oz Indonesia, meninggal dunia pada Jumat (4/8) dini hari. Dokter Lula Kamal yang merupakan selebriti sekaligus rekan kerja Ryan menyebut kawannya itu sudah sakit sejak setahun yang lalu...
Summary referensi: Dokter Lula Kamal yang merupakan selebriti sekaligus rekan kerja Ryan Thamrin menyebut kawannya itu sudah sakit sejak setahun yang lalu. Lula menuturkan, sakit itu membuat Ryan mesti vakum dari semua kegiatannya, termasuk menjadi pembawa acara Dokter Oz Indonesia.

2. Memuat model Gemma2 9B Sahabat AI...
Model berhasil dimuat!

3. Testing summarization...
Memproses berita: 1501893029-lula-kamal-dokter-ryan-thamrin-sakit-sejak-setahun

Teks asli: Jakarta, CNN Indonesia - - Dokter Ryan Thamrin, yang terkenal lewat acara Dokter Oz Indonesia, meninggal dunia pada Jumat (4/8) dini hari. Dokter Lula Kamal yang merupakan selebriti sekaligus rekan kerja Ryan menyebut kawannya itu sudah sakit sejak setahun yang lalu...

Summary yang dihasilkan: Dokter Ryan Thamrin, pembawa acara Dokter Oz Indonesia, meninggal dunia pada Jumat dini hari. Rekan kerjanya, Dokter Lula Kamal, menyebut Ryan sudah sakit sejak setahun lalu dan harus vakum dari semua kegiatannya.

Summary referensi: Dokter Lula Kamal yang merupakan selebriti sekaligus rekan kerja Ryan Thamrin menyebut kawannya itu sudah sakit sejak setahun yang lalu. Lula menuturkan, sakit itu membuat Ryan mesti vakum dari semua kegiatannya, termasuk menjadi pembawa acara Dokter Oz Indonesia.

Metrik evaluasi:
- Compression ratio: 0.85
- Word overlap: 0.45
- Panjang summary: 34 kata
- Panjang referensi: 40 kata
```

## Lisensi

Sistem ini menggunakan model Gemma2 9B Sahabat AI yang memiliki lisensi tersendiri. Pastikan untuk mematuhi ketentuan lisensi model yang digunakan.

## Kontribusi

Kontribusi untuk meningkatkan sistem ini sangat diterima. Silakan buat pull request atau issue untuk melaporkan bug atau saran perbaikan.