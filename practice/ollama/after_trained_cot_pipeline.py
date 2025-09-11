import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"

model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

system_prompt = """
Be a helpful assistant and answer the user's question.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{query}"),
])

pipeline = prompt_template | llm

query = (
    "James has 7 apples, he eats 4 and is given an additional 19 apples, "
    "James gives 15 apples to Josh, and Josh gives James 2 apples, how "
    "many apples does James have?"
)
result = pipeline.invoke({"query": query}).content
print(result)
