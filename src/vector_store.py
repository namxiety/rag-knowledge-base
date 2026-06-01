from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 使用中文优化的嵌入模型（完全本地运行，无需API）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def create_vector_store(chunks, persist_dir="./chroma_db"):
    """创建并持久化Chroma向量数据库"""
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

def load_vector_store(persist_dir="./chroma_db"):
    """加载已有向量数据库"""
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
