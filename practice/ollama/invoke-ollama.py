from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_ollama.chat_models import ChatOllama

from practice.prompt import first_prompt, article, second_prompt, third_prompt, fourth_prompt, image_prompt

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
print('-' * 100)

chain_two = LLMChain(
    llm=llm,  # we use the more deterministic LLM here
    prompt=second_prompt,
    output_key="summary"
)

output_two = chain_two.invoke({"article": article, "article_title": article_title})
summary = output_two["summary"]
print(summary)
print('-' * 100)

chain_three = LLMChain(
    llm=creative_llm,
    prompt=third_prompt,
    output_key="article_para"
)
output_three = chain_three.invoke({"article": article})
article_para = output_three["article_para"]
print(article_para)
print('-' * 100)

chain_four = LLMChain(
    llm=llm,  # we need precision here so we use the more deterministic LLM
    prompt=fourth_prompt,
    output_key="new_suggestion_article"
)
output_four = chain_four.invoke({"article": article, "article_para": article_para})
print(output_four["new_suggestion_article"])
print('-' * 100)

chain_five = LLMChain(llm=llm, prompt=image_prompt, output_key="article_image")

article_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four, chain_five],  # our linked chains
    input_variables=["article"],  # the single input variable (used by our first chain)
    output_variables=["article_title", "summary", "article_para", "new_suggestion_article", "article_image"],
    # all of the outputs we want to return
    verbose=True  # to show AI intermediate steps
)

result = article_chain.invoke({"article": article})
print(result)
