# Aksharam Translation System
Tone-aware English-Malayalam machine translation system optimized for resource-constrained environments using mBART-50 and LoRA.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run Data Alignment: `python 01_collect_and_align.py`
   - Downloads OPUS-100 dataset.
   - Computes LaBSE embeddings and creates a highly-aligned `aligned_dataset.json`.
3. Train Model: `python 02_train.py`
   - Fine-tunes `mBART-50` using PEFT (LoRA) on the aligned dataset.
   - Saves adapters to `./aksharam_model_final`.
4. Run Evaluation: `python 03_evaluate.py`
   - Generates translation hypotheses.
   - Calculates **SacreBLEU** metric for n-gram accuracy.
   - Calculates **LaBSE Cosine Similarity** for tone and semantic preservation.

## Future Improvements for Aksharam

To truly elevate this system to a state-of-the-art "tone-aware" translator for literary and cultural texts (like novels):

1. **Stylistic Conditioning Tokens (`<formal>`, `<informal>`, `<poetic>`)**
   - **Current State:** The model learns a generalized tone based on the arbitrary mix of OPUS subtitles/corpus data.
   - **Improvement:** Introduce special tokens during training prepended to the source English string. For instance, `<informal> How are you?` -> `സുഖമാണോ?` vs `<formal> How are you?` -> `തങ്ങൾക്ക് സുഖമാണോ?`.

2. **Expanded Novel-Specific Custom Datasets**
   - **Current State:** Using generic OPUS-100 to prove the technical pipeline. 
   - **Improvement:** Scrape copyright-free English and Malayalam classical literature (e.g., from Project Gutenberg and Malayalam wikisource). Use a dynamic sliding window approach alongside LaBSE to align paragraph-level chunks rather than strict 1:1 sentences, capturing flowing narrative tones better.

3. **RLHF (Reinforcement Learning from Human Feedback) for Cultural Nuance**
   - **Current State:** Pure Supervised Fine-Tuning (SFT) minimizing cross-entropy loss.
   - **Improvement:** Build a simple reward model that scores translations based on cultural appropriateness (e.g., using proper honorifics like "chettan" or "sir" where implied in English context). Fine-tune the mBART LoRA adapter using PPO.

4. **Larger Context Windows (Document-Level Translation)**
   - **Current State:** Translates isolated sentences. Tone often depends heavily on previous paragraphs.
   - **Improvement:** Migrate from `mBART-50` to a model with a larger context window (like `Llama-3-8B` quantized or `Mistral-v0.3` which have long context lengths) to translate an entire page of a novel at once, maintaining consistent character voices (Register/Tone).
