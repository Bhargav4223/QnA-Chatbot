import os
import tempfile
import uuid
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, Docx2txtLoader,UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
from langchain.schema import Document
import time
import ocrmypdf
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
from langdetect import detect
from googletrans import Translator
# Initialize logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "default_api_key")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = FastAPI()

# Store chat sessions
chat_sessions = {}

class URLInput(BaseModel):
    url: str

class ChatInput(BaseModel):
    chat_id: str
    question: str
    temperature: float

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatHistory(BaseModel):
    messages: List[ChatMessage]

def extract_content_from_url(url):
    try:
        logging.info(f"Extracting content from URL: {url}")
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--headless")  # If you're running in a container
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        # Initialize the WebDriver with ChromeDriverManager
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options
        )
        driver.get(url)

        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, 'html.parser')

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator='\n', strip=True)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        doc = Document(
            page_content='\n\n'.join(paragraphs),
            metadata={"source": url}
        )

        logging.info("Content extracted successfully from URL.")
        return [doc]
    except Exception as e:
        logging.error(f"Error extracting content from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting content from URL: {str(e)}")



def extract_youtube_transcript(url):
    try:
        video_id = None
        if "youtube.com" in url:
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "/embed/" in url:
                video_id = url.split("/embed/")[1].split("?")[0]
        elif "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]

        if not video_id:
            raise ValueError("Invalid YouTube URL format")

        # Get all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Check if transcripts are available
        if transcript_list is None:
            raise ValueError("No transcripts available for this video")

        # Try to get the English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English is not available, use the first available transcript
            transcript = next(iter(transcript_list))

        # Extract the text from the transcript
        text = " ".join([entry['text'] for entry in transcript.fetch()])

        # If the transcript is not in English, translate it
        if transcript.language_code != 'en':
            translator = Translator()
            translation = translator.translate(text, dest='en')
            text = translation.text

        return text

    except Exception as e:
        logging.error(f"Error extracting YouTube transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting YouTube transcript: {str(e)}")

def extract_audio_content(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    audio.export("temp.wav", format="wav")

    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    os.remove("temp.wav") 
    return text

def extract_video_content(file_path):
    video = VideoFileClip(file_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")
    text = extract_audio_content("temp_audio.wav")
    os.remove("temp_audio.wav")
    return text

def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            ocr_pdf_path = temp_file.name
        ocrmypdf.ocr(file_path, ocr_pdf_path, force_ocr=True)
        loader = UnstructuredPDFLoader(ocr_pdf_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension in ['.mp3', '.wav', '.ogg']:
        text = extract_audio_content(file_path)
        return [Document(page_content=text, metadata={"source": file_path})]
    elif file_extension in ['.mp4', '.avi', '.mov']:
        text = extract_video_content(file_path)
        return [Document(page_content=text, metadata={"source": file_path})]
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    logging.info("Setting up vector store.")
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    logging.info("Vector store setup completed.")
    return vectorstore

def create_chain(vectorstore, temperature=0):
    logging.info("Creating conversational retrieval chain.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm = llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    logging.info("Conversational retrieval chain created.")
    return chain, llm

@app.post("/process_url")
async def process_url(url_input: URLInput):
    logging.info(f"Received URL to process: {url_input.url}")
    documents = extract_content_from_url(url_input.url)
    vectorstore = setup_vectorstore(documents)
    conversation_chain, initial_llm = create_chain(vectorstore)
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = (conversation_chain, initial_llm)
   
    #chat_sessions[chat_id] = conversation_chain

    logging.info(f"URL processed and stored with chat_id: {chat_id}")
    return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}

@app.post("/process_youtube")
async def process_youtube(url_input: URLInput):
    logging.info(f"Received YouTube URL to process: {url_input.url}")
    try:
        transcript = extract_youtube_transcript(url_input.url)
        documents = [Document(page_content=transcript, metadata={"source": url_input.url})]
        vectorstore = setup_vectorstore(documents)
        conversation_chain, initial_llm = create_chain(vectorstore)
        chat_id = str(uuid.uuid4())
        chat_sessions[chat_id] = (conversation_chain, initial_llm)

        logging.info(f"YouTube video processed and stored with chat_id: {chat_id}")
        return {"chat_id": chat_id, "message": "YouTube video content processed and stored successfully."}
    except Exception as e:
        logging.error(f"Error processing YouTube video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing YouTube video: {str(e)}")

@app.post("/process_document")
async def process_document_endpoint(file: UploadFile = File(...)):
    logging.info(f"Received document file to process: {file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        documents = load_document(temp_file_path)
        vectorstore = setup_vectorstore(documents)
        conversation_chain, initial_llm = create_chain(vectorstore)
        chat_id = str(uuid.uuid4())
        chat_sessions[chat_id] = (conversation_chain, initial_llm)

        #chat_sessions[chat_id] = conversation_chain

        logging.info(f"Document processed and stored with chat_id: {chat_id}")
        return {"chat_id": chat_id, "message": "Document content processed and stored successfully."}
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        os.unlink(temp_file_path)

@app.post("/chat")
async def chat(chat_input: ChatInput):
    logging.info(f"Received chat input: chat_id={chat_input.chat_id}, question={chat_input.question}, temperature={chat_input.temperature}")
    if chat_input.chat_id not in chat_sessions:
        logging.warning(f"Chat session not found: {chat_input.chat_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")

    conversation_chain, current_llm = chat_sessions[chat_input.chat_id]
    
    if current_llm.temperature != chat_input.temperature:
        new_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=chat_input.temperature
        )
        conversation_chain.combine_docs_chain.llm_chain.llm = new_llm
        chat_sessions[chat_input.chat_id] = (conversation_chain, new_llm)

    try:
        response = conversation_chain({"question": chat_input.question})
        logging.info("Response generated successfully.")
        return {"response": response["answer"]}
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")