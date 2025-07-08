"""
Konfigurasi untuk sistem summarization berita
"""

import os
from typing import Dict, Any

class Config:
    """
    Kelas konfigurasi untuk sistem summarization
    """
    
    # Model Configuration
    MODEL_ID = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
    TRANSFORMERS_VERSION = "4.45.0"
    
    # Model Parameters
    MODEL_PARAMS = {
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    # Text Processing
    MAX_INPUT_LENGTH = 2000
    MIN_SUMMARY_LENGTH = 10
    MAX_SUMMARY_LENGTH = 500
    
    # Prompt Templates
    DEFAULT_PROMPT_TEMPLATE = """Buatlah ringkasan singkat dari berita berikut dalam bahasa Indonesia. Ringkasan harus mencakup informasi penting dan ditulis dalam 2-3 kalimat.

Berita:
{text}

Ringkasan:"""

    SHORT_PROMPT_TEMPLATE = """Ringkaslah berita berikut dalam 1-2 kalimat yang informatif dan mudah dipahami. Fokus pada informasi utama dan fakta penting.

Berita:
{text}

Ringkasan singkat:"""

    DETAILED_PROMPT_TEMPLATE = """Buatlah ringkasan lengkap dari berita berikut dalam bahasa Indonesia. Ringkasan harus mencakup:
1. Siapa (who) - tokoh utama
2. Apa (what) - peristiwa utama
3. Kapan (when) - waktu kejadian
4. Di mana (where) - lokasi kejadian
5. Mengapa (why) - alasan/konteks
6. Bagaimana (how) - cara/akibat

Berita:
{text}

Ringkasan lengkap:"""
    
    # File Paths
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_LOG_FILE = "summarization.log"
    
    # Batch Processing
    BATCH_SIZE = 10
    SAVE_INTERVAL = 10  # Save every N items
    
    # Evaluation
    EVALUATION_METRICS = [
        'rouge-1',
        'rouge-2', 
        'compression_ratio',
        'summary_length',
        'word_overlap'
    ]
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Mendapatkan parameter model"""
        return cls.MODEL_PARAMS.copy()
    
    @classmethod
    def get_prompt_template(cls, template_type: str = "default") -> str:
        """Mendapatkan template prompt berdasarkan tipe"""
        templates = {
            "default": cls.DEFAULT_PROMPT_TEMPLATE,
            "short": cls.SHORT_PROMPT_TEMPLATE,
            "detailed": cls.DETAILED_PROMPT_TEMPLATE
        }
        return templates.get(template_type, cls.DEFAULT_PROMPT_TEMPLATE)
    
    @classmethod
    def create_output_dir(cls) -> str:
        """Membuat direktori output jika belum ada"""
        if not os.path.exists(cls.DEFAULT_OUTPUT_DIR):
            os.makedirs(cls.DEFAULT_OUTPUT_DIR)
        return cls.DEFAULT_OUTPUT_DIR
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Mendapatkan path lengkap untuk file output"""
        output_dir = cls.create_output_dir()
        return os.path.join(output_dir, filename)

# Konfigurasi untuk environment yang berbeda
class DevelopmentConfig(Config):
    """Konfigurasi untuk development"""
    LOG_LEVEL = "DEBUG"
    BATCH_SIZE = 5

class ProductionConfig(Config):
    """Konfigurasi untuk production"""
    LOG_LEVEL = "WARNING"
    BATCH_SIZE = 20
    SAVE_INTERVAL = 50

class TestConfig(Config):
    """Konfigurasi untuk testing"""
    LOG_LEVEL = "ERROR"
    BATCH_SIZE = 2
    MAX_INPUT_LENGTH = 500

# Factory function untuk mendapatkan konfigurasi berdasarkan environment
def get_config(env: str = "development") -> Config:
    """
    Mendapatkan konfigurasi berdasarkan environment
    
    Args:
        env (str): Environment ('development', 'production', 'test')
        
    Returns:
        Config: Konfigurasi yang sesuai
    """
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig
    }
    
    return configs.get(env, DevelopmentConfig)

# Default config
config = get_config(os.getenv("ENVIRONMENT", "development"))