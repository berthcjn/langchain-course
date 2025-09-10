from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

prompt = """
Answer the user's query based on the context below.                 
If you cannot answer the question using the
provided information answer with "I don't know".

Context: {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt),
    HumanMessagePromptTemplate.from_template("{query}"),
])

# print(prompt_template)

model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

pipeline = prompt_template | llm

context = """Aurelio AI is an AI company developing tooling for AI
engineers. Their focus is on language AI with the team having strong
expertise in building AI agents and a strong background in
information retrieval.

The company is behind several open source frameworks, most notably
Semantic Router and Semantic Chunkers. They also have an AI
Platform providing engineers with tooling to help them build with
AI. Finally, the team also provides development services to other
organizations to help them bring their AI tech to market.

Aurelio AI became LangChain Experts in September 2024 after a long
track record of delivering AI solutions built with the LangChain
ecosystem."""

# query = "what does Aurelio AI do?"
query = "what do you know from this context?"

output_1 = pipeline.invoke({"query": query, "context": context})

print(output_1.content)
