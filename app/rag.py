import os
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

QDRANT_URL = os.environ.get("QDRANT_URL")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(url=QDRANT_URL)


def create_vector_store(file_id: str, file_path: str):
    # Загрузка документов и чанкование
    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1080,
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
