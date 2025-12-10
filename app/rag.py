import os
from uuid import uuid4

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

QDRANT_URL = os.environ.get("QDRANT_URL")

client = QdrantClient(url=QDRANT_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def create_vector_store(file_id: str, file_path: str):
    # Загрузка документов и чанкование
    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    # Размер чатка сделал большим, чтобы загрузить весь файл. Можно модифицировать
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1080 * 44,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(docs)

    # Создание коллекции по имени документа
    client.create_collection(
        collection_name=file_id,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name=file_id, embedding=embeddings
    )

    # Загрузка чанков
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)


def generate_answer(file_id: str, question: str) -> str:
    vector_store = QdrantVectorStore(
        client=client, collection_name=file_id, embedding=embeddings
    )

    @tool(
        response_format="content_and_artifact",
        description="Retrieve Context for the answer"
    )
    def retrieve(query: str):
        retrieved_docs = vector_store.similarity_search(query)
        serialized = "\n\n".join(
            f"Контекст: {doc.page_content}" for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    system_prompt = (
        "Ты - ассистент, отвечающий на запросы пользователя, используя документы ниже"
    )

    rag_agent = create_agent(
        model="gpt-4.1", tools=[retrieve], system_prompt=system_prompt
    )

    response = rag_agent.invoke({"messages": [{"role": "user", "content": question}]})

    return response["messages"][-1].content
