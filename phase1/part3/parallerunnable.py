from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

parser = StrOutputParser()

short_prompt=ChatPromptTemplate.from_template(
    "explain {topic} in 1-2 line"
)

detailed_prompt=ChatPromptTemplate.from_template(
    "explain {topic} in detail but it should be in simple term"
)

chain = RunnableParallel({
    "short": short_prompt|model|parser,
    "detail":detailed_prompt|model|parser

})

result= chain.invoke({
    "topic":"agentic ai"
})

print("Short Explanation",result['short'])
print("Detailed Explanation",result['detail'])
