from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("D:/vikram_genai/phase2/dense_rag/documentloader/genAI pdf.pdf")
docs = loader.load()
print(docs[7])
#print(len(docs))