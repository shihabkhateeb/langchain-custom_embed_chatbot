from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
os.environ["OPENAI_API_KEY"] = "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"

# Load Data
loader = UnstructuredFileLoader("data_to_ingest.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)