
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

#1 Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

#2 Create a chat model
llm = ChatGroq(model="openai/gpt-oss-120b")

#3 connect to the vector store
vectorstores = Chroma(
    persist_directory="D:/vikram_genai/phase2/dense_rag/vector_stores/chroma_db",
    embedding_function=embedding_model
)

#4 create a retrival
retriever = vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.7
    }
)


# 5. Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the given context.

If the answer is not available in the context, say:
"I don't know."

Context:
{context}

Question:
{question}
""")


print("RAG System is ready")
print("Type 0 to exit")


while True:
    query =input("\n YOU:")
    if query =="0":
        break
    docs = retriever.invoke(query)
    context = "\n \n ".join(doc.page_content for doc in docs)
    print("\n Retrieved Context: \n ",context)
    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })
    response = llm.invoke(final_prompt)
    print("AI:",response.content)