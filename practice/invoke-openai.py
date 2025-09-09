import os
from getpass import getpass

from langchain_openai import ChatOpenAI

from practice.prompt import first_prompt, article

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)

openai_model = "gpt-4o-mini"

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model=openai_model)

# For unique creative responses
creative_llm = ChatOpenAI(temperature=0.9, model=openai_model)

chain_one = (
        {"article": lambda x: x["article"]}
        | first_prompt
        | creative_llm
        | {"article_title": lambda x: x.content}
)

article_title_msg = chain_one.invoke({"article": article})
# print(article_title_msg)
