from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
loader=PyPDFLoader("D:/vikram_genai/phase2/dense_rag/documentloader/genAI pdf.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks,model,persist_directory="D:/vikram_genai/phase2/dense_rag/vector_stores/chroma_db")
print("vector store created ")