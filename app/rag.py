import os
from uuid import uuid4

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    Modifier,
    SparseIndexParams,
)

from scripts.clean_text import clean_docx_text

QDRANT_URL = os.environ.get("QDRANT_URL")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP"))
VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

client = QdrantClient(url=QDRANT_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def create_vector_store(file_id: str, file_path: str):
    text = clean_docx_text(file_path)
    docs = [Document(page_content=text)]

    # loader = TextLoader(file_path)
    # docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    splits = text_splitter.split_documents(docs)

    client.create_collection(
        collection_name=file_id,
        vectors_config={VECTOR_NAME: VectorParams(size=1536, distance=Distance.COSINE)},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                modifier=Modifier.IDF, index=SparseIndexParams(on_disk=False)
            )
        },
    )

    vector_store = get_vector_store(collection_name=file_id)

    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)


def generate_answer(file_id: str, question: str) -> str:
    vector_store = get_vector_store(collection_name=file_id)

    @tool(
        response_format="content_and_artifact",
        description="Retrieve Context for the answer",
    )
    def retrieve(query: str):
        retrieved_docs = vector_store.similarity_search(query)
        serialized = "\n\n".join(
            f"Контекст: {doc.page_content}" for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    system_prompt = (
        "Ты - корпоративный информационный агент, отвечающий на вопросы сотрудников. "
        "Твоя задача - предоставлять точную и релевантную информацию, основываясь исключительно на данных из предоставленных документах. "
        "Если информация отсутствует в документах, сообщи об этом. Не выдумывай ответы и не добавляй лишнюю информацию. "
        "НЕ используй Markdown форматирование в ответах и отвечай одним параграфом. "
    )

    rag_agent = create_agent(
        model="gpt-4o-mini", tools=[retrieve], system_prompt=system_prompt
    )

    response = rag_agent.invoke({"messages": [{"role": "user", "content": question}]})

    response = response["messages"][-1].content.replace("\n", " ")

    return response
