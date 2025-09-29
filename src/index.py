from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from embedding import Embedder

class FAISSIndex:
    def __init__(self, embedding_model=None):
        self.embedder = embedding_model or Embedder()
        self.vectorstore = None

    def build_index(self, texts, metadatas=None):
        """
        texts: List of strings or LangChain Documents
        metadatas: Optional list of metadata dictionaries
        """
        if isinstance(texts[0], str):
            docs = [Document(page_content=text, metadata=metadatas[i] if metadatas else {}) for i, text in enumerate(texts)]
        else:
            docs = texts

        self.vectorstore = FAISS.from_documents(docs, self.embedder)

    def save(self, path="faiss_index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path="faiss_index"):
        self.vectorstore = FAISS.load_local(path, self.embedder)

    def search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)
