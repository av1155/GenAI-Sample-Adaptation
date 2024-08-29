import json
from typing import Any, List

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import BedrockChat, ChatOllama, ChatOpenAI
from langchain.embeddings import (BedrockEmbeddings, OllamaEmbeddings,
                                  SentenceTransformerEmbeddings)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

from utils import (BaseLogger, cosine_similarity, extract_title_and_question,
                   get_db_connection)


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/tmp"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "llama3":
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=8192,  # Update to 8192 for llama3's context length
        )
    elif llm_name == "phi3:mini":
        logger.info("LLM: Using Phi3 Mini")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            top_k=10,  # You can adjust this based on your preference
            top_p=0.3,  # You can adjust this based on your preference
            num_ctx=4096,  # Phi3 Mini's context length
        )
    elif llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def store_chunks_in_db(chunks, embedding_func):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Create table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chunk TEXT,
            embedding BLOB
        )
    """
    )

    for chunk in chunks:
        embedding = embedding_func([chunk])[0]  # Generate embedding for the chunk
        cursor.execute(
            "INSERT INTO pdf_chunks (chunk, embedding) VALUES (%s, %s)",
            (chunk, embedding),
        )

    connection.commit()
    cursor.close()
    connection.close()


def retrieve_chunks(query_embedding, embedding_func):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT chunk, embedding FROM pdf_chunks")
    chunks = cursor.fetchall()

    # Find the closest matching chunk using cosine similarity
    closest_chunk = None
    max_similarity = -1

    for chunk, embedding in chunks:
        # Deserialize the stored embedding from JSON
        embedding = json.loads(embedding)
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_chunk = chunk

    cursor.close()
    connection.close()

    return closest_chunk


def configure_qa_rag_chain(llm, embeddings):
    # RAG response
    general_system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links from Stackoverflow.
    You should prefer information from accepted or more upvoted answers.
    Make sure to rely on information from the answers and not on questions to provide accurate responses.
    When you find a particular answer in the context useful, make sure to cite it in the answer using the link.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Each answer you generate should contain a section at the end of links to 
    Stackoverflow questions and answers you found useful, which are described under Source value.
    You can only use links to StackOverflow questions that are present in the context and always
    add links to the end of the answer in the style of citations.
    Generate concise answers with a references sources section of links to 
    relevant StackOverflow questions only at the end of the answer.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    return qa_chain


def generate_ticket(llm_chain, input_question):
    # Example function to generate a ticket based on an input question
    response = llm_chain(input_question)
    new_title, new_question = extract_title_and_question(response["answer"])
    return (new_title, new_question)
