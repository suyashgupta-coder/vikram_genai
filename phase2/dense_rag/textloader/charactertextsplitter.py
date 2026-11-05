from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader=TextLoader("D:/vikram_genai/phase2/dense_rag/documentloader/notes.txt")
docs = loader.load()
splitter = CharacterTextSplitter(separator="",chunk_size=20,chunk_overlap=1)
chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()