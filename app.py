import json
import os

import numpy as np
import pymysql
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from streamlit.logger import get_logger

from chains import load_embedding_model, load_llm

# Database connection parameters
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "SentenceTransformer")
llm_name = os.getenv("LLM", "llama2")

# Check if the required environment variables are set
if not all([db_host, db_user, db_password, db_name, ollama_base_url]):
    st.write("The application requires some information before running.")
    with st.form("connection_form"):
        db_host = st.text_input("Enter DB_HOST")
        db_user = st.text_input("Enter DB_USER")
        db_password = st.text_input("Enter DB_PASSWORD", type="password")
        db_name = st.text_input("Enter DB_NAME")
        ollama_base_url = st.text_input("Enter OLLAMA_BASE_URL")
        st.markdown(
            "Only enter the OPENAI_API_KEY to use OpenAI instead of Ollama. Leave blank to use Ollama."
        )
        openai_apikey = st.text_input("Enter OPENAI_API_KEY", type="password")
        submit_button = st.form_submit_button("Submit")
    if submit_button:
        if not all([db_host, db_user, db_password, db_name]):
            st.write("Enter the database information.")
        if not (ollama_base_url or openai_apikey):
            st.write("Enter the Ollama URL or OpenAI API Key.")
        if openai_apikey:
            llm_name = "gpt-3.5"
            os.environ["OPENAI_API_KEY"] = openai_apikey

logger = get_logger(__name__)

# Load embeddings and LLM
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def get_db_connection():
    connection = pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=int(os.getenv("DB_PORT", 3306)),
    )
    return connection


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Check if the token is an instance of AIMessage and extract the content
        if isinstance(token, AIMessage):
            token = token.content  # Extract content from the AIMessage
        self.text += token
        self.container.markdown(self.text)


def store_chunks_in_db(chunks, embedding_func):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Create table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chunk TEXT,
            embedding TEXT  # Change to TEXT to store the serialized embedding
        )
    """
    )

    for chunk in chunks:
        embedding = embedding_func([chunk])[0]  # Generate embedding for the chunk
        embedding_str = json.dumps(embedding)  # Serialize the embedding directly
        cursor.execute(
            "INSERT INTO pdf_chunks (chunk, embedding) VALUES (%s, %s)",
            (chunk, embedding_str),
        )

    connection.commit()
    cursor.close()
    connection.close()


def retrieve_chunks(query_embedding):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT chunk, embedding FROM pdf_chunks")
    chunks = cursor.fetchall()

    # Find the closest matching chunk using cosine similarity
    closest_chunk = None
    max_similarity = -1

    for chunk, embedding_str in chunks:
        # Deserialize the stored embedding from JSON
        embedding = json.loads(embedding_str)
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_chunk = chunk

    cursor.close()
    connection.close()

    return closest_chunk


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    st.header("📄 Chat with your PDF file")

    # Upload your PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Langchain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store the chunks in MariaDB
        store_chunks_in_db(chunks, embeddings.embed_documents)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            query_embedding = embeddings.embed_query(query)
            closest_chunk = retrieve_chunks(query_embedding)

            if closest_chunk:
                stream_handler = StreamHandler(st.empty())
                # Wrap the chunk in HumanMessage
                response = llm([HumanMessage(content=closest_chunk)])
                stream_handler.on_llm_new_token(response)
            else:
                st.write("No matching chunk found.")


if __name__ == "__main__":
    main()
