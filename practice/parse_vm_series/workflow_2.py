from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

model_name = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(temperature=0.0, model=model_name)

embedding_model = 'nomic-embed-text'
embeddings = OllamaEmbeddings(model=embedding_model)

output_parser = StrOutputParser()

# 1. Load CSVseries
series_relationship = 'gcp_series_relationship.csv'
compute_specs = 'gcp-compute-specs.csv'

print("Start to load all retrievers")
# File 1
loader_1 = CSVLoader(series_relationship)
docs_1 = loader_1.load()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_kwargs={'k': 10})

print("Finish loading retriever 1")

# File 2
loader_2 = CSVLoader(compute_specs)
docs_2 = loader_2.load()
vectorstore_2 = FAISS.from_documents(docs_2, embeddings)
retriever_2 = vectorstore_2.as_retriever(search_kwargs={'k': 20})

print("Finish loading retriever 2")

# 3. Prompt Template
prompt_template = """
You are an expert in hardware series information.
The user will ask about a specific series.
Based on the retrieved context below, answer the question.

You are given two pieces of content:

Context 1:
{context_1}

Context 2:
{context_2}

Please note:
- The field `series_name` in Context 1 and the field `model_category` in Context 2 refer to the same concept.

Based on the above, process, compare, or answer questions regarding these two pieces of content.

User question: {question}

Only return the answer, no explain.
"""
prompt = PromptTemplate.from_template(prompt_template)

# 4. Chain Workflow

retrieval = RunnableParallel({
    "context_1": retriever_1,
    "context_2": retriever_2,
    "question": RunnablePassthrough()
})

chain = retrieval | prompt | llm | output_parser

# 5. Run Query
# result = chain.invoke("What is the series_category of P6-b200?")
# result = chain.invoke("List all series name of series family Hpc")
result = chain.invoke("what is the vcpu amount of model c2d-highmem-2")
# result = chain.invoke("What is the previous and next generation of P6-b200?")
# result = chain.invoke("List all unique series_family in the provided context")
print(result)
