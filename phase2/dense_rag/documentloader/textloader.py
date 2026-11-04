from langchain_community.document_loaders import TextLoader

loader=TextLoader("D:/vikram_genai/phase2/dense_rag/documentloader/notes.txt")
docs = loader.load()
print(docs[0].page_content)
