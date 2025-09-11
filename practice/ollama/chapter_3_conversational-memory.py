from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field

model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

# ------------------ Old: ConversationBufferMemory --------------------------

# memory = ConversationBufferMemory(return_messages=True)
#
# memory.save_context(
#     {"input": "Hi, my name is Josh"},  # user message
#     {"output": "Hey Josh, what's up? I'm an AI model called Zeta."}  # AI response
# )
# memory.save_context(
#     {"input": "I'm researching the different types of conversational memory."},  # user message
#     {"output": "That's interesting, what are some examples?"}  # AI response
# )
# memory.save_context(
#     {"input": "I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory."},  # user message
#     {"output": "That's interesting, what's the difference?"}  # AI response
# )
# memory.save_context(
#     {"input": "Buffer memory just stores the entire conversation, right?"},  # user message
#     {"output": "That makes sense, what about ConversationBufferWindowMemory?"}  # AI response
# )
# memory.save_context(
#     {"input": "Buffer window memory stores the last k messages, dropping the rest."},  # user message
#     {"output": "Very cool!"}  # AI response
# )
#
# memory.load_memory_variables({})
#
# chain = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
#
# result = chain.invoke({"input": "what is my name again?"})
# print(result['response'])
#
# # ------------------ New: ConversationBufferMemory vs RunnableWithMessageHistory --------------------------
#
system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

pipeline = prompt_template | llm


#
# chat_map = {}
#
#
# def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = InMemoryChatMessageHistory()
#     return chat_map[session_id]
#
#
# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )
#
# res_1 = pipeline_with_history.invoke(
#     {"query": "Hi, my name is Josh"},
#     config={"session_id": "id_123"}
# )
# print(res_1.content)
#
# res_2 = pipeline_with_history.invoke(
#     {"query": "What is my name again?"},
#     config={"session_id": "id_123"}
# )
# print(res_2.content)

# # ------------------ Old: ConversationBufferWindowMemory --------------------------

# memory = ConversationBufferWindowMemory(k=4, return_messages=True)
#
# memory.chat_memory.add_user_message("Hi, my name is Josh")
# memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
# memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
# memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
# memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
# memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
# memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
# memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
# memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
# memory.chat_memory.add_ai_message("Very cool!")
#
# memory.load_memory_variables({})
#
# chain = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
#
# result = chain.invoke({"input": "what is my name again?"})
# print(result['response'])


# ------------------ New: ConversationBufferWindowMemory vs RunnableWithMessageHistory --------------------------

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        # print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}


def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    # print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    # remove anything beyond the last
    return chat_map[session_id]


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

res_1 = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)
print('-' * 100)
print(res_1.content)

chat_map["id_k4"].clear()  # clear the history

# manually insert history
chat_map["id_k4"].add_user_message("Hi, my name is Josh")
chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
chat_map["id_k4"].add_ai_message("Very cool!")

# print(chat_map["id_k4"].messages)

res_2 = pipeline_with_history.invoke(
    {"query": "what is my name again?"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)
print('-' * 100)
print(res_2.content)

res_3 = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)
print('-' * 100)
print(res_3.content)

res_4 = pipeline_with_history.invoke(
    {"query": "what is my name again?"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)
print('-' * 100)
print(res_4.content)
