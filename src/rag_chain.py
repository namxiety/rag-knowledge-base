from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化DeepSeek客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def build_prompt(question, retrieved_docs):
    """构建带检索上下文的提示词"""
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = "\n".join([f"- {doc.metadata.get('source', 'unknown')}" for doc in retrieved_docs])
    
    prompt = f"""你是一个基于个人知识库的问答助手。请根据以下【参考内容】回答问题。如果参考内容中没有相关信息，请明确回答“根据当前知识库无法回答该问题”。回答要准确、简洁，并尽量引用参考内容中的原句。

【参考内容】
{context}

【问题】
{question}

【回答】
"""
    return prompt, sources

def rag_query(vectordb, question, top_k=3):
    """执行RAG查询：检索 → 生成"""
    # 检索最相关的 top_k 个文档块
    retrieved_docs = vectordb.similarity_search(question, k=top_k)
    
    # 构建提示
    prompt, sources = build_prompt(question, retrieved_docs)
    
    # 调用DeepSeek生成答案
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的知识库问答助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # 低温度提高事实性
        max_tokens=1000
    )
    answer = response.choices[0].message.content
    return answer, sources, retrieved_docs

def plain_llm_query(question):
    """不使用检索，直接问大模型（用于对比）"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个通用AI助手。"},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return response.choices[0].message.content
