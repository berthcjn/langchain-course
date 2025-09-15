from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough, RunnableMap
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

# File 1
loader = CSVLoader(series_relationship)
docs = loader.load()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# 3. Prompt Template
prompt_template = """
You are an expert in hardware series information.
The user will ask about a specific series.
Based on the retrieved context below, answer the question.

Context:
{context}

When comparing, matching, or extracting information, strictly filter by `series_name`. 
Only consider values that are exactly equal to the given `series_name`. 
Do not match partial strings or similar names.

User question: {question}

If the answer has many series, should order by the series name.

Only return the answer, no explain.
"""
prompt = PromptTemplate.from_template(prompt_template)

# 4. Chain Workflow

retrieval = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
})

chain = retrieval | prompt | llm | output_parser

# 5. Run Query
# result = chain.invoke("What is the series_category of P6-b200?")
result = chain.invoke("What is the previous and next generation of m2?")
print(result)
