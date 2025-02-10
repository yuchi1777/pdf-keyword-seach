import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from collections import defaultdict

# Path settings
chroma_path = "/Users/yuchi/vectorstore"
document_path = "/Users/yuchi/python/台灣醫界_vol67(11).pdf"

# Keywords list
keywords = [
    "Acalabrutinib", "Anastrozole", "Anifrolumab", "Atenolol", "Benralizumab",
    "Bicalutamide", "Budesonide", "Dapagliflozin", "Durvalumab", "Eculizumab",
    "Esomeprazole", "Fulvestrant", "Gefitinib", "Goserelin", "Metoprolol",
    "Olaparib", "Osimertinib", "Palivizumab", "Ravulizumab", "Rosuvastatin",
    "Saxagliptin", "Selumetinib", "Sodium", "Tamoxifen", "Tezepelumab",
    "Ticagrelor", "Tremelimumab", "ANDEXANET", "Vaxzevria", "Cilgavimab",
    "BAMBUTEROL", "CANDE", "CAPIVASERTIB", "CEFTAROLINE", "CEFTAZIDIME",
    "EPLONTERSEN", "EXENATIDE", "FELODIPINE", "FORMOTEROL", "FLUMIST",
    "LESINURAD", "LIDOCAINE", "LINACLOTIDE", "LISINOPRIL", "Nirsevimab",
    "MEROPENEM", "OMEPRAZOLE", "PRAMLINTIDE", "PROPOFOL", "PROPRANOLOL",
    "QUETIAPINE", "ROFLUMILAST", "ROPIVACAINE", "ROXADUSTAT", "TERBUTALINE",
    "TRASTUZUMAB", "FLUOROURACIL", "Asfotase", "Calquence", "Arimidex",
    "Saphnelo", "Tenormin", "Fasenra", "Casodex", "Pulmicort", "Symbicort",
    "BREZTRI", "Forxiga", "Xigduo", "Imfinzi", "Soliris", "Nexium", "Faslodex",
    "Iressa", "Bevespi", "Zoladex", "Betaloc", "Lynparza", "Tagrisso", "Synagis",
    "Ultomiris", "Crestor", "Onglyza", "Qtern", "Koselugo", "Lokelma", "Nolvadex",
    "Tezspire", "Brilinta", "Imjudo", "Andexxa", "Tenoretic", "Niften",
    "COVID-19 Vaccine", "EVUSHELD", "Bambec", "ZolaCos", "Airsupra", "Atacand",
    "TRUQAP", "Zinforo", "Zavicefta", "Nexium", "Bydureon", "Plendil", "Logimax",
    "Oxis", "Xylocaine", "Zestril", "Zestoretic", "Beyfortus", "Meronem",
    "Selocomb", "Seloken", "Losec", "SymlinPen", "Diprivan", "Inderal",
    "Seroquel", "Daxas", "Naropin", "Evrenzo", "Kombiglyze", "Bricanyl",
    "Enhertu", "Strensiq"
]

# Initialize embeddings and LLM
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
hf_pipeline = pipeline(
    "text-generation", 
    model="gpt2", 
    max_new_tokens=50,
    pad_token_id=50256,
    truncation=True
)
local_llm = HuggingFacePipeline(
    pipeline=hf_pipeline,
    model_kwargs={"temperature": 0.7}
)

def load_and_split_documents(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def search_keywords_in_documents(documents, keywords):
    """Search for all keywords in the documents and return matches with context."""
    results = defaultdict(list)
    
    for doc in documents:
        page_number = doc.metadata.get('page', 'Unknown')
        for keyword in keywords:
            if keyword.lower() in doc.page_content.lower():
                # Store the matching content with page number
                results[keyword].append({
                    'page': page_number,
                    'content': doc.page_content
                })
    
    return results

def print_search_results(results):
    """Print the search results in a formatted way."""
    print("\n=== Keyword Search Results ===")
    for keyword, matches in results.items():
        if matches:
            print(f"\n{keyword} (Found in {len(matches)} locations):")
            for idx, match in enumerate(matches, 1):
                print(f"\nMatch {idx} (Page {match['page']}):")
                # Print a snippet of the content around the keyword
                content = match['content']
                print(f"{content[:200]}...")  # Print first 200 characters
                print("-" * 80)

# Main execution
if __name__ == "__main__":
    print("Loading and splitting documents...")
    documents = load_and_split_documents(document_path)

    print("\nSearching for keywords...")
    search_results = search_keywords_in_documents(documents, keywords)
    
    # Print results
    print_search_results(search_results)
    
    # Save results to a file
    with open('keyword_search_results.txt', 'w', encoding='utf-8') as f:
        f.write("Keyword Search Results\n")
        f.write("=====================\n\n")
        for keyword, matches in search_results.items():
            if matches:
                f.write(f"\n{keyword} (Found in {len(matches)} locations):\n")
                for idx, match in enumerate(matches, 1):
                    f.write(f"\nMatch {idx} (Page {match['page']}):\n")
                    f.write(f"{match['content'][:200]}...\n")
                    f.write("-" * 80 + "\n")

    print("\nResults have been saved to 'keyword_search_results.txt'")