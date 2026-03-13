<div align="center">

# 🎙️ Whisper LoRA Fine-Tuning for Shami (Levantine Arabic)

### Fine-tune OpenAI's Whisper for Syrian, Lebanese, Jordanian & Palestinian Arabic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#-quick-start)
[![HuggingFace Model](https://img.shields.io/badge/🤗_Model-whisper--shami-yellow)](https://huggingface.co/mabahboh/whisper-shami)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

<br>

**Whisper struggles with Levantine Arabic dialects.** This notebook fixes that — using LoRA to fine-tune `whisper-small` on ~33K Shami speech samples while training only **1.4%** of the model's parameters.

<br>

[Getting Started](#-quick-start) · [Results](#-results) · [Datasets](#-datasets) · [Configuration](#%EF%B8%8F-configuration) · [Model](#-trained-model)

</div>

---

## 📌 Highlights

- **Parameter-Efficient**: LoRA adapters train only **3.5M / 245M params (1.4%)** — runs on a single **free Colab GPU (T4)**
- **Multi-Dataset Pipeline**: Combines 3 Arabic speech datasets with smart streaming to avoid huge downloads
- **Arabic Text Normalization**: Built-in diacritics removal, Alef/Ta-Marbuta unification, and cleaning pipeline
- **End-to-End**: From raw audio to a merged, deployable HuggingFace model in one notebook
- **Real Improvement**: **~17 percentage points WER reduction** over the base Whisper model on Levantine speech

---

## 🚀 Quick Start

### 1. Open in Google Colab

Upload `whisper_lora_shami_finetuning.ipynb` to Google Colab, select a **GPU runtime** (T4 or higher), and run all cells.

### 2. Or run locally

```bash
git clone https://github.com/<your-username>/whisper-shami-lora.git
cd whisper-shami-lora
pip install -r requirements.txt
jupyter notebook whisper_lora_shami_finetuning.ipynb
```

### 3. Use the pre-trained model directly

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

model_name = "mabahboh/whisper-shami"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load your audio
speech_array, sr = torchaudio.load("your_audio.wav")
speech_array = speech_array.mean(dim=0)  # mono

# Transcribe
inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")
pred_ids = model.generate(**inputs)
text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
print(text)
```

---

## 📊 Results

| Model | WER (%) | Params Trained |
|-------|---------|---------------|
| Whisper Small (base) | 100.29% | — |
| **+ LoRA Fine-Tuned (ours)** | **83.38%** | **1.4% (3.5M)** |
| **Improvement** | **↓ 16.91 pp** | — |

> **Note**: WER on Levantine Arabic is inherently high due to dialectal variation and limited standardized orthography. The base model essentially produces random outputs for Shami dialects — the fine-tuned model shows substantial qualitative improvement in generating coherent Arabic transcriptions.

---

## 📁 Datasets

This notebook combines **three open-source datasets** to maximize Levantine Arabic coverage:

| Dataset | Size | Strategy | Content |
|---------|------|----------|---------|
| [`halabi2016/arabic_speech_corpus`](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) | ~1 GB | Full download | Damascian Levantine Arabic |
| [`pain/MASC`](https://huggingface.co/datasets/pain/MASC) | ~200 GB total | **Streaming** (0 GB disk!) | Multi-dialect — filtered for Levantine only |
| [`fsicoli/common_voice_22_0`](https://huggingface.co/datasets/fsicoli/common_voice_22_0) | ~3 GB (Arabic) | Full download | Crowd-sourced Arabic |

**Final dataset**: ~33,044 training samples + 5,000 evaluation samples

### Smart Streaming for MASC

MASC is ~200 GB, but we only need the Levantine subset. The notebook **streams** MASC and filters on-the-fly — collecting up to 3,000 Levantine samples without downloading the full dataset.

---

## ⚙️ Configuration

All hyperparameters are in a single config cell (Section 2) for easy tweaking:

```python
# Model
MODEL_ID = "openai/whisper-small"
LANGUAGE = "arabic"

# LoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2    # effective batch = 16
LEARNING_RATE = 1e-3
WARMUP_STEPS = 500
FP16 = True
```

### Tips for Better Results

| Goal | What to Change |
|------|---------------|
| **More data** | Set `MAX_TRAIN_SAMPLES = None` and `MASC_MAX_LEVANTINE_SAMPLES = 10000` |
| **Better accuracy** | Increase `LORA_R=64`, add `"k_proj", "out_proj"` to targets, train 5-10 epochs |
| **Less VRAM** | Reduce batch to 4, increase gradient accumulation to 4, enable 8-bit quantization |
| **Bigger model** | Change `MODEL_ID` to `whisper-medium` or `whisper-large-v3` |

---

## 🏗️ Notebook Structure

| Section | Description |
|---------|-------------|
| **1. Setup** | Install dependencies (transformers, peft, datasets, etc.) |
| **2. Config** | All hyperparameters in one place |
| **3. Processor** | Load Whisper tokenizer + feature extractor |
| **4. Normalization** | Arabic text cleaning (diacritics, Alef forms, Ta-Marbuta) |
| **5. Datasets** | Load & combine 3 datasets with streaming support |
| **6. Features** | Extract Whisper mel-spectrograms + tokenize labels |
| **7. Collator** | Custom padding for Seq2Seq training |
| **8. Metrics** | WER evaluation with Arabic normalization |
| **9. Model + LoRA** | Load Whisper + apply LoRA adapters |
| **10. Train** | Launch training with Seq2SeqTrainer |
| **11. Evaluate** | Compute WER on held-out test set |
| **12. Inference** | Merge LoRA → full model, run transcription |
| **13. Comparison** | Side-by-side Base vs Fine-Tuned evaluation |
| **14. Push to Hub** | Upload merged model to HuggingFace Hub |

---

## 🤗 Trained Model

The fine-tuned model is available on HuggingFace Hub:

👉 [**mabahboh/whisper-shami**](https://huggingface.co/mabahboh/whisper-shami)

```python
# One-liner with pipeline
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mabahboh/whisper-shami")
result = pipe("path/to/shami_audio.wav", generate_kwargs={"language": "arabic"}, chunk_length_s=30)
print(result["text"])
```

---

## 🔧 Requirements

```
torch>=2.0
transformers>=4.39.0
datasets>=2.18.0,<4.0.0
accelerate>=0.26.0
peft>=0.9.0
bitsandbytes>=0.41.0
evaluate
jiwer
librosa
soundfile
tensorboard
```

**Hardware**: NVIDIA GPU with ≥16 GB VRAM recommended (T4, A100, etc.). Free Google Colab T4 works.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — base ASR model
- [HuggingFace PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [halabi2016/arabic_speech_corpus](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) — Damascian Levantine speech data
- [pain/MASC](https://huggingface.co/datasets/pain/MASC) — Multi-dialect Arabic speech corpus
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — Crowd-sourced Arabic speech

---

## ⭐ Citation

If you use this work, please consider citing:

```bibtex
@misc{whisper-shami-lora,
  title={LoRA Fine-Tuning Whisper for Shami (Levantine Arabic)},
  author={mabahboh},
  year={2025},
  url={https://huggingface.co/mabahboh/whisper-shami}
}
```

---

<div align="center">

**If this helped you, give it a ⭐!**

Made with ❤️ for the Arabic NLP community

</div>
