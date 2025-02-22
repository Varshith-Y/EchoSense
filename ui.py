import os
import tempfile
import time
import cv2
import numpy as np
import wave
import pyaudio
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import openai
import whisper
from autogen import ConversableAgent
from torchvision import models, transforms


# Custom CSS for Aesthetic Design with Background Images

st.markdown("""
<style>
/* Main app container with a gradient and background image */
[data-testid="stAppViewContainer"] {
    background: 
      linear-gradient(to bottom right, rgba(29,38,113,0.85), rgba(195,55,100,0.85)),
      url("https://source.unsplash.com/1600x900/?nature,water");
    background-blend-mode: overlay;
    background-size: cover;
}

/* Set main text color and font */
[data-testid="stAppViewContainer"] * {
    color: #ffffff;
    font-family: "Helvetica Neue", sans-serif;
}

/* Sidebar with gradient and background image */
[data-testid="stSidebar"] {
    background: 
      linear-gradient(to bottom, rgba(46,46,46,0.9), rgba(55,59,68,0.9)),
      url("https://source.unsplash.com/800x1200/?abstract");
    background-blend-mode: overlay;
    background-size: cover;
}

/* Style primary buttons */
button[kind="primary"] {
    background-color: #ff7f50 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Style secondary buttons */
button[kind="secondary"] {
    background-color: #4CAF50 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Add extra padding at the top */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# Set Your OpenAI API Key 

OPENAI_API_KEY = 'Enter Your API Key Here'
openai.api_key = OPENAI_API_KEY
if not OPENAI_API_KEY:
    st.error("Error: OPENAI_API_KEY not found.")


# Know Your Rights Module (using LangChain)

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vector_db = None
        self.qa_chain = None  
        self.load_and_vectorize_pdf()
        self.build_qa_chain()  

    def extract_text(self):
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_text(text)

    def create_vector_db(self, text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_db = Chroma.from_texts(text_chunks, embeddings)

    def load_and_vectorize_pdf(self):
        text = self.extract_text()
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        text_chunks = self.split_text(text)
        self.create_vector_db(text_chunks)

    def build_qa_chain(self):
        retriever = self.vector_db.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=retriever
        )

    def ask(self, query):
        full_query = f"""
You are an AI assistant responding to a user query based on a provided document.
Please provide a detailed answer with references from the document.
Always include a legal disclaimer at the end: "Disclaimer: This response is based on the provided document and is for informational purposes only. Please consult a qualified lawyer for legal advice."

User Query: {query}
"""
        return self.qa_chain.run(full_query)

def pdf_chatbot_app():
    st.title("Know Your Rights")
    st.write("Ask questions about your accessibility rights.")

    if "pdf_chatbot" not in st.session_state:
        try:
            st.session_state.pdf_chatbot = PDFChatbot("Data.pdf")
        except Exception as e:
            st.error(f"Error processing: {e}")

    if "pdf_chatbot" in st.session_state:
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Processing your question..."):
                try:
                    answer = st.session_state.pdf_chatbot.ask(question)
                    st.write("**Answer:**", answer)
                except Exception as e:
                    st.error(f"Error during query: {e}")

# Braille Dictionary & Helper

BRAILLE_DICT = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
    'z': '⠵',
    ' ': ' ',
    ',': '⠂', '.': '⠲', '?': '⠦', '!': '⠖', ';': '⠆',
    ':': '⠒', '-': '⠤', '(': '⠣', ')': '⠜', '"': '⠐⠦',
    "'": '⠄',
    '0': '⠼⠚', '1': '⠼⠁', '2': '⠼⠃', '3': '⠼⠉', '4': '⠼⠙',
    '5': '⠼⠑', '6': '⠼⠋', '7': '⠼⠛', '8': '⠼⠓', '9': '⠼⠊'
}

def text_to_braille(text: str) -> str:
    """Convert the given text into a simplified Grade-1 Braille string."""
    return ''.join(BRAILLE_DICT.get(char, '?') for char in text.lower())

# Speech-to-Braille Functionality
def record_audio(filename="temp.wav", record_secs=5, rate=16000):
    """Record audio from the default microphone for the specified duration."""
    chunk = 1024
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=rate,
                     input=True,
                     frames_per_buffer=chunk)
    frames = []
    for _ in range(int(rate / chunk * record_secs)):
        frames.append(stream.read(chunk))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

def speech_to_braille_app():
    st.title("Speech-to-Braille Converter (Record Audio Only)")
    audio_file_path = None

    if st.button("Start Recording"):
        st.markdown("🔴 **Recording in progress...** Please speak into the microphone.")
        audio_file_path = record_audio(filename="temp.wav", record_secs=5)
    
    if audio_file_path is not None:
        whisper_model = whisper.load_model("small")
        result = whisper_model.transcribe(audio_file_path)
        user_text = result["text"].strip()

        st.subheader("You:")
        st.write(user_text)

        ai_response = agent.generate_reply(messages=[{"role": "user", "content": user_text}])
        ai_text = ai_response if ai_response else "I couldn't generate a response."

        st.subheader("AI Response (English)")
        st.write(ai_text)

        braille_output = text_to_braille(ai_text)
        st.subheader("Braille Output")
        st.write(braille_output)

        st.download_button(
            "Download Braille Output",
            data=braille_output,
            file_name="braille_output.txt",
            mime="text/plain"
        )

        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

# ASL Translator Functionality
def load_asl_model():
    class_names = [
        'A','B','C','D','E','F','G','H','I','J','K',
        'L','M','N','O','P','Q','R','S','T','U','V',
        'W','X','Y','Z','del','nothing','space'
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    checkpoint = torch.load("asl_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model, class_names, device

def ai_agent(response_text):
    """
    AI agent to generate responses based on the recognized word.
    If the word is short (≤3 characters), it will ask about the user's day.
    """
    prompt = f"""
    You are a friendly and engaging AI assistant. 
    Your job is to respond to user inputs with natural, context-aware conversations. 
    Follow these rules:
    
    1️⃣ If the input is a short word (1-3 characters), respond in a casual, engaging way by asking about their day or well-being.
    2️⃣ If the input is a full sentence or longer word, generate a thoughtful, context-aware response.
    3️⃣ Be friendly, conversational, and engaging.
    
    User Input: "{response_text}"
    AI Response:
    """
    if len(response_text) <= 3:
        return f"{response_text.capitalize()}! That’s a short word. How's your day going?"
    try:
        response = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        if isinstance(response, str):
            return response
        elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            return response[0].get("content", "No response generated.")
        else:
            return "Unexpected response format from AI."
    except Exception as e:
        return f"Error occurred while generating response: {e}"

def recognize_asl_live():
    st.title("Live Hand Sign Recognition")
    model, class_names, device = load_asl_model()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
        return
    recognized_letters = []
    last_predicted_label = None
    last_predicted_time = time.time()
    start_time = time.time()  
    video_placeholder = st.empty()  
    letter_placeholder = st.empty()  
    word_placeholder = st.empty()    
    ai_response_placeholder = st.empty()  
    stop_button = st.button("Stop ASL Recognition")
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > 100:  
            st.warning("Time limit reached. Stopping recognition.")
            break
        if stop_button:
            st.warning("Recognition stopped manually.")
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        roi_x, roi_y, roi_w, roi_h = 300, 40, 224, 224
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_t = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_t)
            _, preds = torch.max(outputs, 1)
        predicted_label = class_names[preds.item()]
        current_time = time.time()
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_label}", (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        letter_placeholder.write(f"**Current Letter:** {predicted_label}")
        if predicted_label == last_predicted_label:
            if current_time - last_predicted_time >= 2:
                if predicted_label == "nothing":
                    continue
                elif predicted_label == "space":
                    recognized_letters.append(" ")
                else:
                    recognized_letters.append(predicted_label)
                last_predicted_time = current_time
        else:
            last_predicted_label = predicted_label
            last_predicted_time = current_time
        word_placeholder.write(f"**Recognized Letters So Far:** {' '.join(recognized_letters)}")
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        if len(recognized_letters) >= 5:
            final_word = ''.join(recognized_letters)
            recognized_letters = []  
            ai_response = ai_agent(final_word)
            word_placeholder.success(f"**Final Recognized Word:** {final_word}")
            ai_response_placeholder.info(f"**AI Response:** {ai_response}")
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def hand_sign_to_text_app():
    st.title("ASL Translator Conversion")
    if st.button("Start Live ASL Recognition"):
        recognize_asl_live()

# Initialize the AI Agent for Other Modules
agent = ConversableAgent(
    name="BrailleChatbot",
    llm_config={
        "config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]
    },
    code_execution_config=False,  
    human_input_mode="NEVER",
)

# Integrated UI with Welcome & Navigation
def welcome_page():
    st.markdown(
        """
        <style>
        @keyframes slideDown {
            0% { opacity: 0; transform: translateY(-50px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .title-animation {
            animation: slideDown 1.5s ease-out forwards;
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            margin-top: 30vh;
        }
        .welcome-text {
            text-align: center;
            font-size: 20px;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="title-animation">EchoSense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-text">'
        'EchoSense is an innovative accessibility tool designed to bridge communication gaps. '
        'It converts speech into braille, translates sign language into text, and lets you know your accessibilty rights; all with an AI agent.'
        '</div>', unsafe_allow_html=True
    )
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button("Get Started with EchoSense"):
            st.session_state.page = "main_app"

def main_app():
    st.sidebar.title("🔹 Navigation")
    page = st.sidebar.radio("Go to:", ["Speech to Braille", "ASL Translator", "Know Your Rights"])
    with st.sidebar.expander("ℹ️ Help", expanded=False):
        st.markdown("### Available Functions")
        st.write("- **Speech to Braille**: Record audio, transcribe, then convert the AI response to Braille.")
        st.write("- **ASL Translator**: Recognize American Sign Language via webcam and receive AI response.")
        st.write("- **Know Your Rights**: Ask questions about your accessibility rights")
    
    if page == "Speech to Braille":
        speech_to_braille_app()
    elif page == "ASL Translator":
        hand_sign_to_text_app()
    elif page == "Know Your Rights":
        pdf_chatbot_app()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if st.session_state.page == "welcome":
        welcome_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
