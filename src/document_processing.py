import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_folder(folder_path):
    """Loads all .txt files from a given folder."""
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def sliding_window_chunking_lc(docs, chunk_size=200, stride=50):
    """Splits documents into chunks using a sliding window approach."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size - stride,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

def process_documents(folder_path):
    """The main function to load and chunk documents."""
    raw_docs = load_documents_from_folder(folder_path)
    chunks = sliding_window_chunking_lc(raw_docs)
    return chunks