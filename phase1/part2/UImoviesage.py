# MovieSage: paste a raw paragraph about a movie and get
# 1- structured information extracted from it
# 2- a clean plot overview
# 3- the result as JSON
# Run with: streamlit run app.py

from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()


class Movie(BaseModel):
    title: str
    release_year: Optional[str] = None
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


parser = PydanticOutputParser(pydantic_object=Movie)

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

SAMPLE_PARAGRAPH = (
    "Released in July 2010, Inception is a mind-bending science fiction thriller "
    "written and directed by Christopher Nolan. Leonardo DiCaprio stars as Dom Cobb, "
    "a skilled thief who steals secrets by entering people's dreams. Instead of stealing "
    "an idea, Cobb is offered a chance to get his old life back if he can plant one in "
    "the mind of Saito's business rival, Robert Fischer, played by Cillian Murphy. To pull "
    "off this \"inception,\" Cobb assembles a team that includes his partner Arthur "
    "(Joseph Gordon-Levitt), the young architect Ariadne (Elliot Page), and the forger "
    "Eames (Tom Hardy), with Ken Watanabe as Saito. The mission takes them through layered "
    "dream worlds, from rain-soaked city streets to a snowy mountain fortress, while Cobb "
    "is haunted by the memory of his late wife Mal (Marion Cotillard). The film explores "
    "themes of grief, guilt, memory and what is real. With its rotating hallway fight and "
    "Hans Zimmer's powerful score, Inception earned more than $830 million worldwide and "
    "won four Academy Awards, including Best Cinematography and Best Visual Effects."
)


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def response_text(content):
    # Gemini may return a plain string or a list of content blocks
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def analyze(paragraph: str) -> Movie:
    final_prompt = prompt.invoke({
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions()
    })
    res = get_model().invoke(final_prompt)
    return parser.parse(response_text(res.content))


def load_sample():
    st.session_state.paragraph = SAMPLE_PARAGRAPH


def show_value(value):
    return value if value else "—"


# ---------------- UI ----------------

st.set_page_config(page_title="MovieSage", page_icon="🎬", layout="centered")

st.title("🎬 MovieSage")
st.caption("Paste a paragraph about any movie. MovieSage extracts the key details and writes a clean overview.")

st.text_area(
    "Movie paragraph",
    key="paragraph",
    height=220,
    placeholder="Released in 2010, Inception is a science fiction thriller directed by...",
)

col_analyze, col_sample = st.columns([1, 1])
analyze_clicked = col_analyze.button("Analyze movie", type="primary", width="stretch")
col_sample.button("Use sample paragraph", on_click=load_sample, width="stretch")

if analyze_clicked:
    paragraph = st.session_state.paragraph.strip()
    if not paragraph:
        st.warning("Please paste a paragraph about a movie first.")
    else:
        with st.spinner("Analyzing the movie..."):
            try:
                st.session_state.movie = analyze(paragraph)
            except Exception as e:
                st.session_state.pop("movie", None)
                st.error(f"Could not analyze the paragraph: {e}")

movie = st.session_state.get("movie")

if movie:
    st.divider()

    title = movie.title
    if movie.release_year:
        title += f" ({movie.release_year})"
    st.header(title)

    if movie.genre:
        st.markdown(" ".join(f":blue-badge[{g}]" for g in movie.genre))

    tab_overview, tab_details, tab_json = st.tabs(["Overview", "Details", "JSON"])

    with tab_overview:
        st.subheader("Plot overview")
        st.write(movie.plot_overview)

        st.subheader("Main cast")
        if movie.main_cast:
            st.markdown("\n".join(f"- {actor}" for actor in movie.main_cast))
        else:
            st.write("—")

    with tab_details:
        c1, c2, c3 = st.columns(3)
        c1.metric("Director", show_value(movie.director))
        c2.metric("Runtime", show_value(movie.runtime))
        c3.metric("Rating", show_value(movie.rating))

        c4, c5, c6 = st.columns(3)
        c4.metric("Language", show_value(movie.language))
        c5.metric("Country", show_value(movie.country))
        c6.metric("Release year", show_value(movie.release_year))

        c7, c8 = st.columns(2)
        c7.metric("Budget", show_value(movie.budget))
        c8.metric("Box office", show_value(movie.box_office_collection))

    with tab_json:
        json_text = movie.model_dump_json(indent=2)
        st.code(json_text, language="json")
        st.download_button(
            "Download JSON",
            data=json_text,
            file_name=f"{movie.title.lower().replace(' ', '_')}.json",
            mime="application/json",
        )
