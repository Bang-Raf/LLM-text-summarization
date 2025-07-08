#!/usr/bin/env python3
"""
Setup script untuk instalasi package evaluasi text summarization
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-summarization-evaluation",
    version="1.0.0",
    author="Text Summarization Evaluation Team",
    author_email="your.email@example.com",
    description="Evaluasi text summarization menggunakan Gemma2 9B dengan metrik ROUGE, BLEU, dan BERTScore",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-summarization-evaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-summarization-eval=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.jsonl", "*.txt", "*.md"],
    },
)