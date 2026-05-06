# 📝 Text Summarizer

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![T5](https://img.shields.io/badge/T5-Small-blueviolet?style=for-the-badge)](https://huggingface.co/t5-small)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A production-ready text summarization web application powered by the **T5 Transformer model** from HuggingFace. Paste any text — articles, dialogues, documents — and get an instant AI-generated summary via a clean REST API.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Important Note on Model File](#-important-note-on-model-file)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🔍 Overview

This application uses **Google's T5 (Text-to-Text Transfer Transformer)** model fine-tuned for summarization. The backend is built with **FastAPI** — providing a proper async REST API with automatic documentation. The model supports GPU (CUDA), Apple Silicon (MPS), and CPU inference automatically.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **T5 Transformer** | Google's T5-Small model for accurate, coherent text summarization |
| ⚡ **FastAPI Backend** | Async REST API with auto-generated Swagger docs at `/docs` |
| 🧹 **Text Preprocessing** | Cleans HTML tags, extra whitespace, and line breaks before inference |
| 🔢 **Beam Search** | Uses num_beams=4 for higher quality output over greedy decoding |
| 💻 **Auto Device Detection** | Automatically uses CUDA → MPS → CPU based on available hardware |
| 🌐 **Web Interface** | Simple HTML/JS frontend for browser-based usage |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI |
| **ML Model** | T5ForConditionalGeneration (HuggingFace Transformers) |
| **Tokenizer** | T5Tokenizer |
| **Deep Learning** | PyTorch |
| **Frontend** | HTML, CSS, JavaScript (Fetch API) |
| **Templates** | Jinja2 |

---

## 📁 Project Structure

```
Text-Summarizer/
│
├── 📄 app.py                        # FastAPI app — routes, model loading, inference
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Excludes model weights, __pycache__, .idea
│
├── 📂 saved_summary_model/          # T5 model files (NOT pushed to GitHub — see below)
│   ├── config.json                  # Model architecture config
│   ├── generation_config.json       # Generation parameters
│   ├── model.safetensors            # Model weights (231MB — use Git LFS or HuggingFace Hub)
│   ├── tokenizer.json               # Tokenizer vocab
│   └── tokenizer_config.json        # Tokenizer settings
│
└── 📂 templates/
    └── index.html                   # Web UI
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/jay51211/Text-Summarizer.git
cd Text-Summarizer
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the model

Since the model file is large (231MB), it is not stored in the repo directly.

**Option A — Download T5-Small from HuggingFace directly:**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")
```

**Option B — If you have the model files already**, place them in `saved_summary_model/`

### Step 4 — Run the application

```bash
uvicorn app:app --reload
```

### Step 5 — Open in browser

```
http://localhost:8000
```

### Bonus — View auto-generated API docs

```
http://localhost:8000/docs
```

---

## 🔌 API Reference

### `POST /summarize/`

Summarize any text input.

**Endpoint:** `/summarize/`
**Method:** `POST`
**Content-Type:** `application/json`

#### Request Body

```json
{
  "dialogue": "Your long text to summarize goes here..."
}
```

#### Response

**✅ Success** `200 OK`
```json
{
  "summary": "Concise AI-generated summary of your text."
}
```

#### Example — cURL

```bash
curl -X POST "http://localhost:8000/summarize/" \
     -H "Content-Type: application/json" \
     -d '{"dialogue": "The Amazon rainforest is the world largest tropical rainforest covering over 5.5 million square kilometres. It represents over half of the remaining rainforests on Earth and is home to an estimated 10% of all species in the world."}'
```

**Response:**
```json
{
  "summary": "the amazon rainforest is the world largest tropical rainforest and is home to an estimated 10% of all species in the world."
}
```

---

## ⚙️ How It Works

```
User Input (text)
      │
      ▼
 clean_data()
 ─────────────────────────────
 • Remove \r\n line breaks
 • Collapse extra whitespace
 • Strip HTML tags (<p>, <h1>)
 • Lowercase + strip
      │
      ▼
 T5Tokenizer.encode()
 ─────────────────────────────
 • max_length = 512 tokens
 • padding = max_length
 • truncation = True
 • return_tensors = "pt"
      │
      ▼
 model.generate()
 ─────────────────────────────
 • num_beams = 4 (beam search)
 • max_length = 150
 • early_stopping = True
      │
      ▼
 tokenizer.decode()
 ─────────────────────────────
 • skip_special_tokens = True
 • Removes EOS, SEP tokens
      │
      ▼
 JSON Response → {"summary": "..."}
```

**Why Beam Search (num_beams=4)?**

Instead of picking the single most likely next word (greedy decoding), beam search keeps track of the top 4 candidate sequences at each step and picks the one with the highest overall probability. This produces significantly more coherent summaries at a small computational cost.

---

## ⚠️ Important Note on Model File

The `model.safetensors` file is **231MB** — too large for a standard GitHub push (limit: 100MB).

**Do NOT push it directly.** Instead use one of these approaches:

**Option 1 — Git LFS (recommended for this repo):**
```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git add saved_summary_model/model.safetensors
git commit -m "Add model via Git LFS"
```

**Option 2 — Add to .gitignore and document download steps:**
```gitignore
saved_summary_model/model.safetensors
```
Then document how to download it in the README (as done above).

**Option 3 — Push model to HuggingFace Hub and load from there.**

---

## 🗺️ Future Roadmap

- [ ] Add `summarize:` prefix to input for proper T5 task conditioning
- [ ] Support for multiple summarization styles (short / detailed / bullet points)
- [ ] File upload support (PDF, DOCX)
- [ ] Deploy on Render or Railway with Docker
- [ ] Add character/word count display
- [ ] Add copy-to-clipboard button for summary output

---

## 📦 Dependencies

```txt
fastapi
uvicorn
transformers
torch
pydantic
jinja2
python-multipart
sentencepiece
```

```bash
pip install fastapi uvicorn transformers torch pydantic jinja2 python-multipart sentencepiece
```

---

## 👤 Author

<div align="center">

**Jay Sandip Kumbhar**
*B.Sc. Computer Science | CGPA 9.00 | ML Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-jay51211-181717?style=for-the-badge&logo=github)](https://github.com/jay51211)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jaykumbhar5121-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/jaykumbhar5121)
[![Email](https://img.shields.io/badge/Email-jaykumbhar518@gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:jaykumbhar518@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with ❤️ by Jay Kumbhar*

⭐ **Star this repo if you found it useful!** ⭐

</div>
