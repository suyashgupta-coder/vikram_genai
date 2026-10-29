# AI bot where you can give him a para about some movie and it is going to give you few things:
#1- take raw para about a movie 
#2- Extract important structured information 
#3- generate a clean summary
# - return it in a json format 

from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")


class Movie(BaseModel):
    title:str
    release_year:Optional[str] = None
    director: Optional[str] = None
    genre: List[str] = []
    main_cast: List[str] = []
    language: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    runtime: Optional[str] = None
    plot_overview: str
    budget: Optional[str] = None
    box_office_collection: Optional[str] = None 

parser =PydanticOutputParser(
    pydantic_object=Movie
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are MovieSage, an AI assistant specialized in analyzing movies.

Your task is to take raw, unstructured information about a movie and
transform it into structured JSON data.

Follow these rules:

1. Extract information only from the provided text.
2. Do not invent information.
3. If information is missing, use null or an empty list where appropriate.
4. Generate a clean and concise plot overview.
5. Return the result according to these formatting instructions:

{format_instructions}
"""
    ),
    (
        "human",
        """
Analyze the following raw movie information:

{paragraph}
"""
    )
])
para = input("Give me your paragraph:")
final_prompt = prompt.invoke({
    "paragraph":para,
    "format_instructions":parser.get_format_instructions()
    })
res = model.invoke(final_prompt)
movie = parser.parse(res.content[0]["text"])
print(movie.model_dump_json(indent=2))