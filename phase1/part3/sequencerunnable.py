from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#1 prompt 

prompt = ChatPromptTemplate.from_template(
    "explain {topic} in simple term"
)

#2 model
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

#3 parser

parser = StrOutputParser()

chain = prompt |model | parser

result=chain.invoke({
    "topic": "Genrative AI"
})
print("Result:",result)
