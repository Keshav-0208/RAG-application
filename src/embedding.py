from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

class Embedder:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}  # set to "cuda" if using GPU
        )

    def embed(self, texts):
        # Accepts list of strings, returns list of vectors
        return self.embedder.embed_documents(texts)
