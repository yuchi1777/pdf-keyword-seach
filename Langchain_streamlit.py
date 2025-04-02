import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import pandas as pd

# Streamlit UI
st.title("📄 PDF 關鍵字搜尋工具")

# 上傳 PDF 檔案
uploaded_files = st.file_uploader("請上傳 PDF 檔案", type=["pdf"], accept_multiple_files=True)

# 上傳關鍵字檔案（TXT 或 Excel）
keywords_file = st.file_uploader("請上傳關鍵字檔案 (TXT 或 Excel)", type=["txt", "xlsx"])

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

def extract_text_from_pdf(uploaded_files):
    """使用 PyMuPDF 解析 PDF 並提取文字"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # 顯示處理中狀態
            with st.spinner(f"正在處理 {uploaded_file.name}..."):
                # 創建臨時檔案存儲 PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # 讀取 PDF 內容
                doc = fitz.open(tmp_file_path)
                for page_num, page in enumerate(doc):
                    raw_text = page.get_text("text") or ""
                    # 修正斷行：carboxy-\nmethylcellulose → carboxymethylcellulose
                    text = raw_text.replace("-\n", "").replace("\n", " ")
                    documents.append({
                        "file": uploaded_file.name,
                        "page": page_num + 1,
                        "content": text
                    })
                
                doc.close()
                os.remove(tmp_file_path)  # 刪除臨時檔案
        
        except Exception as e:
            st.error(f"❌ 無法載入 {uploaded_file.name}: {e}")
    
    return documents

def search_keywords_exact(documents, keywords):
    """完全比對關鍵字（大小寫無關）"""
    results = []
    for doc in documents:
        file_name = doc["file"]
        page_number = doc["page"]
        content = doc["content"]
        
        for keyword in keywords:
            if keyword.lower() in content.lower():
                results.append({
                    "file": file_name,
                    "keyword": keyword,
                    "page": page_number
                })
    
    return results

def display_results(results):
    """顯示搜尋結果"""
    if results:
        st.success("✅ 查詢完成！")
        for result in results:
            st.write(f"📄 檔案: {result['file']} | 🔍 關鍵字: {result['keyword']} | 📄 頁數: {result['page']}")
    else:
        st.warning("❌ 未找到任何關鍵字！")

# 執行主程式
if uploaded_files and keywords_file:
    st.write("🚀 檔案上傳完成，開始處理中...")
    
    keywords = load_keywords(keywords_file)
    if not keywords:
        st.warning("⚠️ 未能讀取任何關鍵字，請檢查上傳的檔案格式。")
    else:
        documents = extract_text_from_pdf(uploaded_files)
        if not documents:
            st.warning("未能解析任何 PDF 內容，請確認上傳的檔案格式。")
        else:
            st.write("🔍 正在搜尋關鍵字...")
            search_results = search_keywords_exact(documents, keywords)
            display_results(search_results)
