import os
import pandas as pd
import re
import streamlit as st
import tempfile
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_keywords(file):
    """讀取關鍵字，可支援 TXT 或 Excel"""
    if file is not None:
        if file.name.endswith(".txt"):
            keywords = [line.strip() for line in file.getvalue().decode("utf-8").splitlines() if line.strip()]
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
            keywords = set()
            for col in df.columns:
                keywords.update(df[col].dropna().astype(str).tolist())
            keywords = list(keywords)
        else:
            st.error("❌ 不支援的檔案格式，請使用 .txt 或 .xlsx")
            return []
        return keywords
    return []

def load_and_split_documents(uploaded_files):
    """處理上傳的 PDF 文件"""
    all_documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # 創建臨時檔案存儲 PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # 使用 PyPDFLoader 讀取臨時 PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # 記錄文件名稱
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
            
            # 切割文本
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_documents.extend(text_splitter.split_documents(documents))
            
            # 刪除臨時檔案
            os.remove(tmp_file_path)
        except Exception as e:
            st.error(f"❌ 無法載入 {uploaded_file.name}: {e}")
    return all_documents

def search_keywords_exact(documents, keywords):
    """確保關鍵字匹配完全一致（大小寫無關，但內容完全匹配）"""
    results = []
    for doc in documents:
        page_number = doc.metadata.get('page', 'Unknown')
        source_file = doc.metadata.get('source', 'Unknown File')
        content = doc.page_content
        
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', content, re.IGNORECASE):
                snippet = doc.page_content[:200]
                results.append({'file': source_file, 'keyword': keyword, 'page': page_number, 'content': snippet})
    return results

def display_results(results):
    """顯示搜尋結果"""
    if results:
        for result in results:
            st.subheader(f"📄 來源檔案: {result['file']}")
            st.write(f"🔍 關鍵字: {result['keyword']}")
            st.write(f"📄 頁數: {result['page']}")
            st.text(result['content'])
            st.markdown("---")

def save_results_to_excel(results):
    """將搜尋結果存為 Excel"""
    df = pd.DataFrame(results, columns=["來源檔案", "關鍵字", "頁數", "內容"])
    df.to_excel("keyword_search_results.xlsx", index=False)
    st.success("📄 搜尋結果已儲存為 keyword_search_results.xlsx")

# Streamlit UI
st.title("📄 PDF 關鍵字搜尋工具")

uploaded_files = st.file_uploader("請上傳 PDF 檔案", type=["pdf"], accept_multiple_files=True)
keywords_file = st.file_uploader("請上傳關鍵字檔案 (TXT 或 Excel)", type=["txt", "xlsx"])

if uploaded_files and keywords_file:
    keywords = load_keywords(keywords_file)
    if not keywords:
        st.warning("⚠️ 未能讀取任何關鍵字，請檢查上傳的檔案格式。")
    else:
        documents = load_and_split_documents(uploaded_files)
        if not documents:
            st.warning("未能解析任何 PDF 內容，請確認上傳的檔案格式。")
        else:
            st.write("🔍 正在搜尋關鍵字...")
            search_results = search_keywords_exact(documents, keywords)
            display_results(search_results)
            save_results_to_excel(search_results)
