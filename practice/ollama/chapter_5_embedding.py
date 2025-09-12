import os

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# model_name = "llama3.2:1b-instruct-fp16"
model_name = "llama3.2:3b-instruct-fp16"

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"


# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

embedding = OllamaEmbeddings(model=model_name)

vecstore_a = DocArrayInMemorySearch.from_texts(
    ["half the info is here", "Josh's birthday is June the 12th"],
    embedding=embedding
)
vecstore_b = DocArrayInMemorySearch.from_texts(
    ["the other half of the info is here", "Josh was born in 2002"],
    embedding=embedding
)

prompt_str = """ Using the following context, answer the question as completely as possible:

Context A: 
{context_a}

Context B: 
{context_b}

Question:
{question}

Answer (better include year, month and day if available, only give the full date no explain)"""

prompt = ChatPromptTemplate.from_template(prompt_str)

retriever_a = vecstore_a.as_retriever()
retriever_b = vecstore_b.as_retriever()

retrieval = RunnableParallel(
    {
        "context_a": retriever_a, "context_b": retriever_b, "question": RunnablePassthrough()
    }
)

output_parser = StrOutputParser()

chain = retrieval | prompt | llm | output_parser

result = chain.invoke("What was the date when Josh was born")
print(result)
