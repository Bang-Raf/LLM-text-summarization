# Struktur Sistem Summarization Berita

## Overview

Sistem ini dirancang untuk melakukan summarization (peringkasan) teks berita dalam bahasa Indonesia menggunakan model Gemma2 9B Sahabat AI. Sistem dapat memproses dataset berita dalam format khusus dan menghasilkan ringkasan yang informatif.

## Struktur File

```
news-summarization-system/
├── requirements.txt          # Dependencies yang diperlukan
├── README.md                # Dokumentasi utama
├── STRUCTURE.md             # Dokumentasi struktur (file ini)
├── config.py                # Konfigurasi sistem
├── data_processor.py        # Pemrosesan dataset berita
├── summarizer.py            # Model summarization utama
├── utils.py                 # Utility functions
├── main.py                  # Script demo utama
├── example_usage.py         # Contoh penggunaan
├── test_system.py           # Testing lengkap
├── simple_test.py           # Testing sederhana
└── setup.py                 # Setup script
```

## Komponen Utama

### 1. `data_processor.py` - NewsDataProcessor

**Fungsi**: Memproses dataset berita dalam format yang diberikan

**Fitur Utama**:
- `load_data_from_file()`: Memuat data dari file JSONL
- `extract_text_from_paragraphs()`: Mengekstrak teks dari struktur paragraphs
- `extract_summary_from_gold_labels()`: Membuat gold summary berdasarkan labels
- `process_dataset()`: Memproses seluruh dataset
- `save_processed_data()`: Menyimpan data yang sudah diproses

**Input Format**:
```json
{
  "category": "tajuk utama",
  "gold_labels": [[false, true], [true, true]],
  "id": "unique-id",
  "paragraphs": [
    [["Token", "1", "dari", "kalimat", "1"], ["Token", "1", "dari", "kalimat", "2"]]
  ],
  "source": "nama sumber",
  "source_url": "https://example.com",
  "summary": [
    [["Token", "summary", "referensi"]]
  ]
}
```

**Output Format**:
```json
{
  "id": "unique-id",
  "category": "tajuk utama",
  "source": "nama sumber",
  "source_url": "https://example.com",
  "full_text": "Teks lengkap yang sudah digabungkan",
  "gold_summary": "Summary berdasarkan gold_labels",
  "original_summary": "Summary referensi asli"
}
```

### 2. `summarizer.py` - NewsSummarizer

**Fungsi**: Melakukan summarization menggunakan model Gemma2 9B Sahabat AI

**Fitur Utama**:
- `load_model()`: Memuat model ke memory
- `create_summarization_prompt()`: Membuat prompt untuk summarization
- `generate_summary()`: Menghasilkan summary dari teks
- `batch_summarize()`: Batch processing untuk dataset besar
- `evaluate_summary()`: Evaluasi kualitas summary

**Model Configuration**:
- Model: `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct`
- Framework: Transformers 4.45.0
- Precision: bfloat16
- Parameters: temperature=0.7, top_p=0.9, max_new_tokens=256

**Prompt Template**:
```
Buatlah ringkasan singkat dari berita berikut dalam bahasa Indonesia. 
Ringkasan harus mencakup informasi penting dan ditulis dalam 2-3 kalimat.

Berita:
{text}

Ringkasan:
```

### 3. `utils.py` - Utility Functions

**Fungsi**: Helper functions untuk evaluasi dan preprocessing

**Fitur Utama**:
- `load_jsonl_file()`: Memuat file JSONL
- `save_jsonl_file()`: Menyimpan ke file JSONL
- `clean_text()`: Membersihkan teks dari karakter tidak diinginkan
- `calculate_rouge_scores()`: Menghitung ROUGE scores
- `evaluate_summaries()`: Evaluasi batch summaries
- `print_evaluation_results()`: Mencetak hasil evaluasi

**Metrik Evaluasi**:
- ROUGE-1: Overlap unigrams
- ROUGE-2: Overlap bigrams
- Compression Ratio: Rasio panjang summary/teks asli
- Word Overlap: Persentase kata yang sama

### 4. `config.py` - Configuration

**Fungsi**: Centralized configuration untuk seluruh sistem

**Fitur Utama**:
- Model parameters
- Text processing settings
- Prompt templates (default, short, detailed)
- Environment-specific configs (development, production, test)
- Output directory management

**Environment Support**:
- Development: Debug mode, batch size 5
- Production: Warning mode, batch size 20
- Test: Error mode, batch size 2

## Workflow Sistem

### 1. Data Processing Workflow
```
Raw Dataset → NewsDataProcessor → Processed Dataset
     ↓              ↓                    ↓
JSONL File → Extract Text → Clean Data
     ↓              ↓                    ↓
Gold Labels → Create Summary → Save Results
```

### 2. Summarization Workflow
```
Processed Data → NewsSummarizer → Generated Summaries
      ↓              ↓                    ↓
Full Text → Load Model → Generate Summary
      ↓              ↓                    ↓
Gold Summary → Evaluate → Save Results
```

### 3. Evaluation Workflow
```
Generated + Gold → Utils → Metrics
      ↓              ↓        ↓
Summaries → Calculate ROUGE → Report
      ↓              ↓        ↓
Results → Print Results → Save Metrics
```

## Penggunaan

### 1. Basic Usage
```python
from data_processor import NewsDataProcessor
from summarizer import NewsSummarizer

# Process data
processor = NewsDataProcessor()
processed_data = processor.process_dataset(raw_data)

# Generate summaries
summarizer = NewsSummarizer()
summarizer.load_model()
results = summarizer.batch_summarize(processed_data)
```

### 2. Evaluation
```python
from utils import evaluate_summaries, print_evaluation_results

metrics = evaluate_summaries(results)
print_evaluation_results(metrics)
```

### 3. Custom Configuration
```python
from config import get_config

config = get_config("production")
summarizer = NewsSummarizer()
summarizer.load_model(device_map=config.MODEL_PARAMS["device_map"])
```

## File Scripts

### 1. `main.py` - Demo Script
- Menampilkan penggunaan lengkap sistem
- Menggunakan sample data
- Menjalankan test summarization
- Menampilkan metrik evaluasi

### 2. `example_usage.py` - Contoh Penggunaan
- Basic usage example
- Batch processing example
- Custom prompt example
- Error handling example

### 3. `test_system.py` - Testing Lengkap
- Test data processor
- Test utils functions
- Test configuration
- Test user sample data

### 4. `simple_test.py` - Testing Sederhana
- Test tanpa dependencies eksternal
- Verifikasi core functionality
- Quick system check

### 5. `setup.py` - Setup Script
- Install dependencies
- Test imports
- Verify system readiness

## Dependencies

### Required Packages
```
transformers==4.45.0
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
json5>=0.9.0
tqdm>=4.64.0
```

### System Requirements
- **RAM**: Minimal 16GB (32GB+ recommended)
- **Storage**: 20GB+ free space
- **Python**: 3.8+
- **GPU**: Optional (8GB+ VRAM recommended)

## Output Files

### 1. Processed Data
- Format: JSONL
- Fields: id, category, source, full_text, gold_summary, original_summary

### 2. Generated Summaries
- Format: JSONL
- Fields: All processed data + generated_summary

### 3. Evaluation Results
- Format: Console output + JSON
- Metrics: ROUGE scores, compression ratio, success rate

## Error Handling

### 1. Data Processing Errors
- Invalid JSON format
- Missing required fields
- Malformed paragraphs structure

### 2. Model Errors
- Out of memory
- Model loading failure
- Generation timeout

### 3. File I/O Errors
- File not found
- Permission denied
- Disk space full

## Performance Considerations

### 1. Memory Management
- Model size: ~18GB (9B parameters)
- Batch processing untuk menghemat memory
- Automatic device mapping

### 2. Speed Optimization
- GPU acceleration
- Batch processing
- Parallel processing (future enhancement)

### 3. Quality vs Speed Trade-offs
- Temperature: 0.7 (balance creativity/consistency)
- Max tokens: 256 (adequate for Indonesian news)
- Top-p: 0.9 (maintains quality)

## Extensibility

### 1. Adding New Models
- Implement new model class
- Update config.py
- Add model-specific parameters

### 2. Adding New Metrics
- Extend utils.py
- Add new evaluation functions
- Update evaluation workflow

### 3. Adding New Data Formats
- Extend NewsDataProcessor
- Add format-specific extractors
- Update validation logic

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size, use CPU
2. **Model Loading Error**: Check transformers version
3. **Data Processing Error**: Validate input format
4. **Evaluation Error**: Check summary format

### Debug Mode
```python
from config import get_config
config = get_config("development")
# Enable debug logging
```

## Future Enhancements

### 1. Performance
- Multi-GPU support
- Model quantization
- Caching mechanisms

### 2. Features
- Multiple model support
- Advanced evaluation metrics
- Web interface
- API endpoints

### 3. Quality
- Fine-tuning capabilities
- Domain-specific prompts
- Quality filtering