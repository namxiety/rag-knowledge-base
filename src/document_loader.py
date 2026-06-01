import os
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents(data_dir: str):
    """加载目录下所有 .txt, .md, .pdf 文件"""
    documents = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith('.txt'):
            loader = TextLoader(path, encoding='utf-8')
        elif file.endswith('.md'):
            loader = UnstructuredMarkdownLoader(path)
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            continue
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file  # 记录来源文件
        documents.extend(docs)
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """将文档切分为固定大小的文本块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"切分完成：{len(documents)} 个文档 → {len(chunks)} 个文本块")
    return chunks
