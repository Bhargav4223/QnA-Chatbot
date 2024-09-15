# Q&A Chatbot for Custom Files using Gemini-1.5-Flash & FAISS-GPU

This repository contains a Q&A chatbot web application that accepts files in various formats (PDF, CSV, TXT, XLSX, DOCX) and answers questions based on the contents of the file. The app is built using the **Streamlit** framework for the frontend, **Gemini-1.5-Flash** as the LLM model, and **FAISS-GPU** as the vector database for document retrieval.

## Features

- **Multi-format file support:** Upload PDF, TXT, CSV, XLSX, or DOCX files.
- **Conversational Q&A:** Ask questions related to the uploaded file, and the chatbot will provide relevant answers.
- **Gemini-1.5-Flash LLM:** Utilizes the Google Generative AI model for high-quality conversational AI.
- **FAISS-GPU Vector DB:** For fast and accurate document embedding and retrieval.
- **Streamlit Frontend:** Clean, interactive UI for seamless interactions.

---

## Table of Contents

- [Setup](#setup)
  - [Running with Streamlit](#running-with-streamlit)
  - [Running with Docker](#running-with-docker)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Setup

### 1. Running with Streamlit

To run this application using the basic Streamlit setup, follow these steps:

#### Prerequisites
- Python 3.8 or higher installed on your system
- Install [pip](https://pip.pypa.io/en/stable/installation/) if not installed

#### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **We need to need install mandatory dependencies below related to ocrymypdf if you are on Windows else ocr functionality won't work**
   ```bash
    GhostScript
    Tesseract
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

6. **Open the app in your browser:**  
   The app will be running at `http://localhost:8501`.

---

### 2. Running with Docker

You can also run the application within a Docker container. Follow these steps:

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system

#### Running with Docker (Without Docker Compose)
1. **Build the Docker image:**
   ```bash
   docker build -t my-streamlit-app .
   ```

2. **Run the Docker container:**
   ```bash
    docker run -p 8501:8501 my-streamlit-app
   ```

3. **Open the app in your browser:**  
   Navigate to `http://localhost:8501`.

#### Running with Docker Compose
Alternatively, you can use Docker Compose to spin up the container:

1. **Run the Docker Compose command:**
   ```bash
   docker-compose up --build
   ```

2. **Open the app in your browser:**  
   Navigate to `http://localhost:8501`.

#### Stopping the Container
To stop the Docker container, run:
```bash
docker stop my-streamlit-app
```

To remove the container:
```bash
docker rm my-streamlit-app
```

---

## Usage

Once the application is running, follow these steps to interact with it:

1. **Upload a File:**  
   Upload a file by selecting from the supported formats (PDF, TXT, CSV, XLSX, DOCX).
   
2. **Ask a Question:**  
   After uploading the file, type your question in the chat input field at the bottom of the page. The chatbot will process the file and provide an answer based on the content of the uploaded file.

3. **View Chat History:**  
   The app maintains a chat history, allowing you to scroll through previous interactions with the chatbot.

---

## Project Structure

Here's an overview of the key files and directories in this project:

- `app.py`: The main Streamlit application script.
- `requirements.txt`: A list of required Python libraries.
- `Dockerfile`: Docker configuration for building the image.
- `docker-compose.yml`: Docker Compose file for running the application with ease.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as per the terms of the license.

---

## Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request. Any contributions or suggestions to improve the application are welcome.

---

By following the above instructions, you will be able to run the Q&A chatbot either using Streamlit or Docker, depending on your preference. This application makes it easy to get answers from any uploaded file in various formats using the power of LLMs and vector databases!
