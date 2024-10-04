import os
import tempfile
import requests
import streamlit as st

def main():
    st.set_page_config(page_title="Chatbot For Custom Documents & URL With FastAPI", page_icon="🤖")
    st.title("Document Chatbot")

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.header("Input Options")
        input_type = st.radio("Choose input type:", ("File", "URL", "YouTube"))

        if input_type == "File":
            uploaded_file = st.file_uploader(label="Upload your document", type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "mp3", "wav", "ogg", "mp4", "avi", "mov"])
            if uploaded_file:
                process_file(uploaded_file)

        elif input_type == "URL":
            url = st.text_input("Enter a URL:")
            if url:
                process_url(url)

        elif input_type == "YouTube":
            youtube_url = st.text_input("Enter a YouTube URL:")
            if youtube_url:
                process_youtube(youtube_url)

        st.header("Model Settings")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # Chat section
    if st.session_state.chat_id:
        st.subheader("Chat")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask your question:")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    response = requests.post(
                        f"http://backend:8000/chat",
                        json={"chat_id": st.session_state.chat_id, "question": question, "temperature": temperature}
                    )
                    response.raise_for_status()
                    answer = response.json()["response"]
                    message_placeholder.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    message_placeholder.error(f"Error generating response: {str(e)}")

def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        response = requests.post(
            "http://backend:8000/process_document",
            files={"file": open(temp_file_path, "rb")}
        )
        response.raise_for_status()
        st.session_state.chat_id = response.json()["chat_id"]
        st.success("Document content processed and stored successfully!")
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

    os.unlink(temp_file_path)

def process_url(url):
    try:
        response = requests.post(
            "http://backend:8000/process_url",
            json={"url": url}
        )
        response.raise_for_status()
        st.session_state.chat_id = response.json()["chat_id"]
        st.success("URL content processed and stored successfully!")
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")

def process_youtube(youtube_url):
    try:
        response = requests.post(
            "http://backend:8000/process_youtube",
            json={"url": youtube_url}
        )
        response.raise_for_status()
        st.session_state.chat_id = response.json()["chat_id"]
        st.success("YouTube video content processed and stored successfully!")
    except Exception as e:
        st.error(f"Error processing YouTube video: {str(e)}")

if __name__ == "__main__":
    main()