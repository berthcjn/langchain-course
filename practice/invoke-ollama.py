from langchain.chains.llm import LLMChain
from langchain_ollama.chat_models import ChatOllama

from practice.prompt import first_prompt, article, second_prompt, third_prompt

model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

# initialize another LLM with temperature 0.9, this makes the LLM more creative
creative_llm = ChatOllama(temperature=0.9, model=model_name)

chain_one = LLMChain(
    llm=creative_llm,  # for more creativity we use the LLM with temperature 0.9
    prompt=first_prompt,
    output_key="article_title"  # specifies the output key for what our LLM generates
)

output_one = chain_one.invoke({"article": article})
article_title = output_one["article_title"]
print("Article Title:", article_title)

chain_two = LLMChain(
    llm=llm,  # we use the more deterministic LLM here
    prompt=second_prompt,
    output_key="summary"
)

output_two = chain_two.invoke({"article": article, "article_title": article_title})
summary = output_two["summary"]
print(summary)

chain_three = LLMChain(
    llm=creative_llm,
    prompt=third_prompt,
    output_key="article_para"
)
output_three = chain_three.invoke({"article": article})
new_para = output_three["article_para"]
print(new_para)
