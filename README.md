# 📘 EchoSense
### *AI Accessibility Assistant for Real-World Disability Support*

EchoSense is an assistive-technology prototype built to enhance communication accessibility for individuals with disabilities. It combines speech processing, deep learning, and a simple user interface to convert audio input into readable text.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model & Data](#model--data)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

EchoSense is designed to support individuals with accessibility needs by translating spoken audio into structured text.  
The project aims to provide a lightweight, scalable, and user-friendly tool that reduces communication barriers in real-time.

This repository includes the prototype model, GUI, and notebook used for experimentation and testing.

---

## ⭐ Features

✔ **Speech-to-Text Conversion**  
Converts spoken audio into clear, readable output.

✔ **Model-Driven Processing**  
Uses a trained deep learning model for prediction.

✔ **Simple User Interface**  
A streamlined GUI (`ui.py`) to run the app without any technical complexity.

✔ **Model Development Notebook**  
`EchoSense.ipynb` contains data exploration, preprocessing, training experiments, and visualisations.

✔ **Extendable Architecture**  
Designed for future expansions such as ASL-to-text or Braille output.

---

## 📁 Project Structure

EchoSense/
│
├── ui.py # Main UI application
├── EchoSense.ipynb # Model development and experimentation notebook
├── Data.pdf # Dataset notes / reference information
└── README.md # Documentation


Model weights must be downloaded separately (see below).

---

## ⚙️ How It Works

1. **Input Layer**  
   User provides audio input (e.g., microphone or uploaded file).

2. **Processing Layer**  
   - Audio is cleaned and transformed  
   - Passed into the pretrained model  
   - Text prediction is generated  

3. **Output Layer**  
   - Text is displayed within the GUI  
   - Future versions will support haptic or Braille-style output  

**Tech Stack Includes**  
- Python  
- Torch (or similar ML framework)  
- Librosa  
- Jupyter Notebook  
- Custom GUI  

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Varshith-Y/EchoSense.git
cd EchoSense

2. Install Dependencies

If you have requirements.txt, run:
pip install -r requirements.txt

Otherwise, typical dependencies may include:
pip install numpy pandas torch librosa scikit-learn matplotlib

3. Download Model Weights

Download from Google Drive:

🔗 Model Weights
https://drive.google.com/file/d/1-zcuc59wegeAEiVPOWyiDnptB-WkyswJ/view?usp=sharing

Place the downloaded file into:
/models
Create this directory if it does not exist.

▶️ Usage
Run the Application
python ui.py

Steps:

Launch the interface

Provide audio input

Receive generated text output

For Model Experimentation

Open:
EchoSense.ipynb
in Jupyter Notebook.

📦 Model & Data

The pretrained model is hosted externally (see link above).

EchoSense.ipynb documents:

Dataset preparation

Feature extraction

Training loops & hyperparameters

Evaluation scores

Model visualisations

Data.pdf provides dataset notes and reference guidelines.

🚀 Future Enhancements

Planned expansions include:

ASL-to-Text using computer vision

Braille device integration

Real-time speaker identification

On-device inference for mobile/edge hardware

Multi-modal interaction support (audio + gesture + text)

