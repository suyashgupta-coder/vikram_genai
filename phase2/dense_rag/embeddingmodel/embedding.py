from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings

model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

text=[
    "Hello i am suyash gupta",
    "i am working in techsara",
    "i am an ai developer"
]
vector = model.embed_documents(text)
print(vector)