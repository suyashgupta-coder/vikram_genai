from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough


model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

parser = StrOutputParser()

prompt1 =ChatPromptTemplate.from_messages([
    ('system',"you are a code generator"),
    ('human'"{topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant who explain code in simple way"),
    ("human","{topic}")
])

seq= prompt1|model|parser
seq2=RunnableParallel(
    {
        "code":RunnablePassthrough(),
        "explanation":prompt2|model|parser
    }
)
chain = seq|seq2

result=chain.invoke({
    "topic": "bubble sort in c++"
})
print("Result:",result["code"])
print("Result:",result["explanation"])


import logging

logging.basicConfig(level=logging.INFO)

logging.info("Application started")
logging.warning("Something needs attention")
logging.error("An error occurred")