import streamlit as st
from vector_store import load_vector_store
from rag_chain import rag_query, plain_llm_query

st.set_page_config(page_title="个人知识库问答", layout="wide")
st.title("📚 个人知识库问答 (RAG)")

# 加载向量库（需提前构建好）
@st.cache_resource
def init_vectordb():
    return load_vector_store("./chroma_db")

vectordb = init_vectordb()

# 侧边栏：模式选择
mode = st.sidebar.radio("问答模式", ["RAG模式（基于知识库）", "普通LLM模式（无检索）"])

# 主界面
question = st.text_input("请输入你的问题：", placeholder="例如：梯度下降算法中的学习率如何选择？")

if question:
    with st.spinner("思考中..."):
        if mode == "RAG模式（基于知识库）":
            answer, sources, docs = rag_query(vectordb, question)
            st.success("✅ RAG回答")
            st.markdown(answer)
            with st.expander("🔍 查看检索到的相关文档片段"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**片段 {i+1}** (来源: {doc.metadata.get('source', '未知')})")
                    st.text(doc.page_content[:500] + "...")
            st.caption(f"📄 引用来源：\n{sources}")
        else:
            answer = plain_llm_query(question)
            st.info("🤖 普通LLM回答（无知识库）")
            st.markdown(answer)
