# Q&A Chatbot for Custom Files, URLs, and YouTube Videos using Gemini-1.5-Flash & FAISS-GPU

This repository contains a Q&A chatbot web application that accepts files in various formats, URLs, and YouTube video URLs, and answers questions based on the provided content. The app is built using **Streamlit** for the frontend, **FastAPI** for the backend, **Gemini-1.5-Flash** as the LLM model, and **FAISS-GPU** as the vector database for document retrieval.

## Features

- **Multi-format input support:** 
  - Upload files: PDF, TXT, CSV, XLSX, XLS, DOCX, MP3, WAV, OGG, MP4, AVI, MOV
  - Input web URLs for content extraction
  - Input YouTube video URLs for transcript extraction and analysis
- **OCR Support:** Automatic OCR processing for PDF files
- **Audio and Video Processing:** Extract and analyze content from audio and video files
- **Conversational Q&A:** Ask questions related to the uploaded file, web content, or YouTube video, and the chatbot will provide relevant answers
- **Gemini-1.5-Flash LLM:** Utilizes the Google Generative AI model for high-quality conversational AI
- **FAISS-GPU Vector DB:** For fast and accurate document embedding and retrieval
- **Streamlit Frontend:** Clean, interactive UI for seamless interactions
- **FastAPI Backend:** Efficient and scalable backend API
- **Temperature Control:** Adjust the model's temperature setting for varied response creativity

---

## Table of Contents

- [Setup](#setup)
  - [Running with Docker Compose](#running-with-docker-compose)
  - [Running Locally](#running-locally)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contributing](#contributing)

---

## Setup

### Running with Docker Compose

The easiest way to run the application is using Docker Compose. This method sets up both the frontend and backend services.

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system

#### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Build and run the Docker containers:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501` to access the frontend.

### Running Locally

If you prefer to run the application without Docker, follow these steps:

#### Prerequisites
- Python 3.8 or higher installed on your system
- pip (Python package manager)

#### Backend Setup
1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install backend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the backend directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the backend server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup
1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install frontend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`.

---

## Usage

Once the application is running, follow these steps to interact with it:

1. **Select Input Type:**
   Choose between File Upload, URL input, or YouTube URL input in the sidebar.

2. **Input Content:**
   - **File Upload:** Select a file from the supported formats (PDF, TXT, CSV, XLSX, XLS, DOCX, MP3, WAV, OGG, MP4, AVI, MOV).
   - **URL Input:** Enter a web URL to extract content from a webpage.
   - **YouTube URL:** Enter a YouTube video URL to extract and analyze the video transcript.

3. **Adjust Model Settings:**
   Use the temperature slider in the sidebar to control the creativity of the model's responses (0.0 for more deterministic, 1.0 for more creative).

4. **Ask Questions:**
   Once the content is processed, type your questions in the chat input field at the bottom of the page. The chatbot will process the content and provide answers based on the input.

5. **View Chat History:**
   The app maintains a chat history, allowing you to scroll through previous interactions with the chatbot.

---

## Project Structure

Here's an overview of the key files and directories in this project:

- `backend/`
  - `main.py`: The FastAPI backend application script.
  - `requirements.txt`: Backend Python dependencies.
  - `Dockerfile`: Docker configuration for the backend.
- `frontend/`
  - `app.py`: The Streamlit frontend application script.
  - `requirements.txt`: Frontend Python dependencies.
  - `Dockerfile`: Docker configuration for the frontend.
- `docker-compose.yml`: Docker Compose file for running both frontend and backend services.
- `.env`: Environment variables file (create this manually).

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as per the terms of the license.

---

## Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request. Any contributions or suggestions to improve the application are welcome.

---

By following the above instructions, you will be able to run the Q&A chatbot either using Docker Compose or by setting up the frontend and backend separately. This application makes it easy to get answers from uploaded files, web pages, and YouTube videos using the power of LLMs and vector databases!
