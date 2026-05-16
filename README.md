# 🌐 Aksharam - English to Malayalam Translator

Aksharam is an AI-powered English to Malayalam translation system built using transformer-based multilingual models, dataset augmentation, and parameter-efficient fine-tuning techniques.

The project focuses on improving translation quality for Malayalam using modern NLP architectures such as mBART50 and LoRA fine-tuning.

---

# 🚀 Features

- English to Malayalam translation
- Transformer-based multilingual architecture
- LoRA-based parameter-efficient fine-tuning
- Dataset preprocessing and alignment pipeline
- Dataset augmentation for improved training
- Evaluation using BLEU and COMET metrics
- GPU-enabled training and inference
- Flask API integration for serving translations

---

# 🧠 Technologies Used

## NLP & Deep Learning
- Hugging Face Transformers
- mBART50
- PyTorch
- LoRA (PEFT)
- Sentence Transformers

## Backend
- Flask

## Evaluation
- COMET
- BLEU Score

## Data Processing
- JSON
- CSV
- Python Data Pipelines

---

# 🏗️ Project Pipeline

The project follows the following pipeline:

1. Data Collection & Alignment
2. Dataset Cleaning
3. Dataset Augmentation
4. Dataset Merging
5. mBART Model Fine-tuning
6. Translation Evaluation
7. API Deployment

---

# 📁 Project Structure

```text
Aksharam-English-Malayalam-Translator/
│
├── aksharam_model/              # Trained model files
├── aksharam_model_final/        # Final model checkpoints
│
├── 01_collect_and_align.py
├── 02_augment_novel.py
├── 02_train.py
├── 03_evaluate.py
├── 03_merge_datasets.py
├── 04_train_mbart.py
├── 05_evaluate.py
├── 06_comet_eval.py
│
├── app.py                       # Flask application
├── collect.py
│
├── aligned_dataset.json
├── augmented_novel.json
├── final_dataset.json
├── paras.csv
│
├── CUDA_SETUP.md
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Aksharam-English-Malayalam-Translator.git

cd Aksharam-English-Malayalam-Translator
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔥 GPU Setup

Ensure CUDA and PyTorch GPU support are properly installed.

Check GPU availability:

```python
import torch
print(torch.cuda.is_available())
```

---

# 🧪 Model Training

## Dataset Preparation

```bash
python 01_collect_and_align.py
```

## Dataset Augmentation

```bash
python 02_augment_novel.py
```

## Merge Datasets

```bash
python 03_merge_datasets.py
```

## Train mBART Model

```bash
python 04_train_mbart.py
```

---

# 📊 Evaluation

## BLEU Evaluation

```bash
python 05_evaluate.py
```

## COMET Evaluation

```bash
python 06_comet_eval.py
```

---

# 🌐 Run Flask Application

```bash
python app.py
```

---

# 🧠 Model Details

## Base Model
- mBART50 Multilingual Transformer

## Fine-tuning Technique
- LoRA (Low-Rank Adaptation)

## Training Features
- Mixed Precision Training
- Gradient Accumulation
- GPU Acceleration

---

# 📈 Future Improvements

- Improved Malayalam dataset scaling
- Better low-resource translation optimization
- Real-time translation interface
- Speech-to-text integration
- Deployment using Docker and cloud services

---

# 👨‍💻 My Contributions

- Dataset preprocessing and alignment
- LoRA-based mBART fine-tuning
- Translation evaluation pipeline
- GPU optimization experiments
- Flask API integration
- COMET and BLEU evaluation setup

---

# 📝 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

- Hugging Face
- PyTorch
- COMET Evaluation Framework
- Open-source NLP community
