# 📘 EchoSense
### *Multimodal AI Accessibility Assistant*

EchoSense is an AI-powered accessibility tool that combines **speech**, **sign language**, and **legal knowledge** to support people with disabilities. It provides:

- 🎙️ **Speech → AI response → Braille**
- 🤟 **ASL hand signs → Text → AI response**
- 📜 **“Know Your Rights” chatbot over an accessibility PDF**

All modules are integrated into a single **Streamlit app**.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture & Modules](#architecture--modules)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Model & Data](#model--data)
- [Future Enhancements](#future-enhancements)

---

## 🔍 Overview

EchoSense is designed to reduce communication barriers for people with disabilities by combining:

- **Speech transcription + LLM + Braille conversion**
- **ASL alphabet recognition via webcam**
- **A legal-information chatbot over an accessibility PDF**

The app is implemented in **Python**, built with **Streamlit**, and uses **OpenAI, Whisper, LangChain, ChromaDB, PyTorch, and OpenCV**.

---

## ⭐ Features

### 🎙️ Speech to Braille
- Records audio from the microphone (PyAudio)
- Transcribes speech using **Whisper** (`small` model)
- Sends text to an **LLM agent (gpt-4o-mini via autogen `ConversableAgent`)**
- Converts the AI response to **Grade 1 Unicode Braille**
- Allows the user to **download Braille output** as a `.txt` file

### 🤟 ASL to Text (Live)
- Uses a **ResNet18** classifier trained on the **ASL Alphabet Kaggle dataset**
- Captures a **Region of Interest (ROI)** from the webcam via OpenCV
- Predicts letters: `A–Z`, plus `del`, `nothing`, `space`
- Builds words from stable predictions
- Sends the recognized word to an **AI agent** for a friendly, contextual response

### 📜 “Know Your Rights” Chatbot
- Loads `Data.pdf` using **pdfplumber**
- Splits text into chunks with **LangChain**’s `RecursiveCharacterTextSplitter`
- Builds a **Chroma** vector store with **OpenAIEmbeddings**
- Uses **RetrievalQA + OpenAI LLM** to answer user questions
- Always appends a legal disclaimer:
  > “Disclaimer: This response is based on the provided document and is for informational purposes only. Please consult a qualified lawyer for legal advice.”

---

## 🧱 Architecture & Modules

The app is organized into **three main modules**, all exposed via the Streamlit UI:

1. **Speech to Braille**
   - `record_audio` → `whisper.transcribe` → `ConversableAgent` → `text_to_braille`
   - Output shown in UI + downloadable

2. **ASL Translator**
   - Pretrained `ResNet18` (`asl_model.pth`)
   - OpenCV webcam feed → ROI → prediction → letters → word
   - Word passed to `ai_agent` for conversational response

3. **Know Your Rights**
   - `PDFChatbot` class:
     - extract PDF text → chunk → embed → Chroma DB
     - RetrievalQA over `Data.pdf`
   - Streamlit interface for Q&A

Navigation is provided via a **sidebar radio menu**:
- `Speech to Braille`
- `ASL Translator`
- `Know Your Rights`

---

## 📁 Project Structure

```text
EchoSense/
│
├── ui.py               # Main Streamlit app (all three modules integrated)
├── EchoSense.ipynb     # Notebook with phase-wise code & experiments
├── Data.pdf            # Legal / accessibility rights document for the chatbot
├── asl_model.pth       # Trained ASL ResNet18 model (not in repo by default)
└── README.md           # Documentation
```

> ⚠️ Note: `asl_model.pth` and some model weights are **not committed** and must be generated or downloaded separately.

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Varshith-Y/EchoSense.git
cd EchoSense
```

### 2. Create & Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

If you create a `requirements.txt`, it should include packages like:

```txt
streamlit
openai
whisper
pyaudio
torch
torchvision
opencv-python
pdfplumber
langchain
chromadb
tiktoken
autogen-agentchat
kagglehub
numpy
Pillow
```

Then install:

```bash
pip install -r requirements.txt
```

Or install manually if you prefer.

### 4. Set Your OpenAI API Key

For security, **do not hardcode** your API key.  
Instead, set it as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"      # macOS / Linux
setx OPENAI_API_KEY "your_api_key_here"        # Windows (new shell required)
```

Make sure `ui.py` and `EchoSense.ipynb` read the key from `OPENAI_API_KEY`.

---

## ▶️ Running the App

From the project root, run:

```bash
streamlit run ui.py
```

You’ll see:

1. A **welcome page** (“EchoSense” title + description)
2. A sidebar with:
   - `Speech to Braille`
   - `ASL Translator`
   - `Know Your Rights`

### Requirements

- **Microphone** (for Speech to Braille)
- **Webcam** (for ASL Translator)
- `Data.pdf` present in the same folder for the “Know Your Rights” chatbot

---

## 📦 Model & Data

### Whisper (Speech to Text)
- Uses the `small` Whisper model: `whisper.load_model("small")`
- Operates locally (no external API for transcription)

### ASL Model
- Trained in `EchoSense.ipynb` using:
  - Dataset: `grassknoted/asl-alphabet` (Kaggle)
  - Architecture: `torchvision.models.resnet18` with custom final layer
- Saved as `asl_model.pth`
- Loaded in `ui.py` by `load_asl_model()`

### PDF Chatbot
- Uses `Data.pdf` as the source document
- Vectorized using:
  - `OpenAIEmbeddings`
  - `Chroma` (local vector store)
- Queried via LangChain’s `RetrievalQA` with OpenAI LLM

---

## 🚀 Future Enhancements

Potential extensions:

- 📱 Deployable mobile or tablet app
- 🔡 Support for **full words and phrases** in ASL, not just alphabet
- 🔁 Continuous conversation memory for all modules
- 🧩 Integration with **refreshable Braille displays**
- 🌐 Multi-language support for speech and text

---
