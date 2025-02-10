import os
import pandas as pd
import re
from collections import defaultdict
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from fuzzywuzzy import fuzz, process
from concurrent.futures import ThreadPoolExecutor

# Path settings
chroma_path = "/Users/yuchi/vectorstore"
document_folder_path = "/Users/yuchi/python/監測期刊"
keywords_path = "/Users/yuchi/python/AZ keywords list(20241113).txt"  # 可以換成 keywords.xlsx

# Load keywords from file
def load_keywords(file_path):
    """讀取關鍵字，可支援 TXT 或 Excel"""
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f.readlines() if line.strip()]
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        keywords = set()
        for col in df.columns:
            keywords.update(df[col].dropna().astype(str).tolist())
        keywords = list(keywords)
    else:
        raise ValueError("Unsupported file format. Please use .txt or .xlsx")
    return keywords

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
hf_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100, pad_token_id=50256, truncation=True, device=-1)
local_llm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={"temperature": 0.1, "do_sample": False, "max_new_tokens": 100})

def load_and_split_document(pdf):
    """載入並切分單個 PDF"""
    try:
        loader = PyPDFLoader(pdf)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # 優化 chunk size
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"❌ Error loading {pdf}: {e}")
        return []

def load_and_split_documents_from_folder(folder_path):
    """多執行緒批量載入資料夾內所有 PDF"""
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    all_documents = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # 使用 4 個執行緒加速處理
        results = executor.map(load_and_split_document, pdf_files)
    
    for docs in results:
        all_documents.extend(docs)
    
    return all_documents

def search_keywords_fuzzy(documents, keywords, threshold=85):
    """更快的模糊匹配搜尋"""
    results = defaultdict(list)
    
    for doc in documents:
        page_number = doc.metadata.get('page', 'Unknown')
        content = doc.page_content.lower()
        
        for keyword in keywords:
            match, score = process.extractOne(keyword.lower(), content.split())
            if score >= threshold:
                snippet = doc.page_content[:200]
                results[keyword].append({'page': page_number, 'content': snippet})
    
    return results

def print_search_results(results):
    """Print the search results in a formatted way."""
    print("\n=== Keyword Search Results ===")
    for keyword, matches in results.items():
        if matches:
            print(f"\n{keyword} (Found in {len(matches)} locations):")
            for idx, match in enumerate(matches[:3], 1):
                print(f"\nMatch {idx} (Page {match['page']}):")
                print(f"{match['content']}...")
                print("-" * 80)

def save_results_to_excel(results):
    """將搜尋結果存為 Excel"""
    output_data = []
    for keyword, matches in results.items():
        for match in matches:
            output_data.append([keyword, match['page'], match['content']])
    
    df = pd.DataFrame(output_data, columns=["關鍵字", "頁數", "內容"])
    df.to_excel("keyword_search_results.xlsx", index=False)
    print("📄 搜尋結果已儲存為 keyword_search_results.xlsx")

# Main execution
if __name__ == "__main__":
    print("Loading and splitting documents...")
    documents = load_and_split_documents_from_folder(document_folder_path)
    if not documents:
        print("No valid documents found. Exiting...")
        exit()

    print("\nLoading keywords from file...")
    keywords = load_keywords(keywords_path)
    
    print("\nSearching for keywords with fuzzy matching...")
    search_results = search_keywords_fuzzy(documents, keywords)
    
    print_search_results(search_results)
    save_results_to_excel(search_results)
