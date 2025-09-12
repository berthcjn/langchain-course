import os

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, \
    ConversationSummaryBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field

model_name = "llama3.2:1b-instruct-fp16"

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"

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

# class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
#     messages: list[BaseMessage] = Field(default_factory=list)
#     k: int = Field(default_factory=int)
#
#     def __init__(self, k: int):
#         super().__init__(k=k)
#         # print(f"Initializing BufferWindowMessageHistory with k={k}")
#
#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         """Add messages to the history, removing any messages beyond
#         the last `k` messages.
#         """
#         self.messages.extend(messages)
#         self.messages = self.messages[-self.k:]
#
#     def clear(self) -> None:
#         """Clear the history."""
#         self.messages = []
#
#
# chat_map = {}
#
#
# def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
#     # print(f"get_chat_history called with session_id={session_id} and k={k}")
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = BufferWindowMessageHistory(k=k)
#     # remove anything beyond the last
#     return chat_map[session_id]
#
#
# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(
#             id="session_id",
#             annotation=str,
#             name="Session ID",
#             description="The session ID to use for the chat history",
#             default="id_default",
#         ),
#         ConfigurableFieldSpec(
#             id="k",
#             annotation=int,
#             name="k",
#             description="The number of messages to keep in the history",
#             default=4,
#         )
#     ]
# )
#
# res_1 = pipeline_with_history.invoke(
#     {"query": "Hi, my name is Josh"},
#     config={"configurable": {"session_id": "id_k4", "k": 4}}
# )
# print('-' * 100)
# print(res_1.content)
#
# chat_map["id_k4"].clear()  # clear the history
#
# # manually insert history
# chat_map["id_k4"].add_user_message("Hi, my name is Josh")
# chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
# chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
# chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
# chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
# chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
# chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
# chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
# chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
# chat_map["id_k4"].add_ai_message("Very cool!")
#
# # print(chat_map["id_k4"].messages)
#
# res_2 = pipeline_with_history.invoke(
#     {"query": "what is my name again?"},
#     config={"configurable": {"session_id": "id_k4", "k": 4}}
# )
# print('-' * 100)
# print(res_2.content)
#
# res_3 = pipeline_with_history.invoke(
#     {"query": "Hi, my name is Josh"},
#     config={"configurable": {"session_id": "id_k4", "k": 4}}
# )
# print('-' * 100)
# print(res_3.content)
#
# res_4 = pipeline_with_history.invoke(
#     {"query": "what is my name again?"},
#     config={"configurable": {"session_id": "id_k4", "k": 4}}
# )
# print('-' * 100)
# print(res_4.content)

# # ------------------ Old: ConversationSummaryMemory --------------------------

# memory = ConversationSummaryMemory(llm=llm)
#
# chain = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
#
# chain.invoke({"input": "hello there my name is Josh"})
# chain.invoke({"input": "I am researching the different types of conversational memory."})
# chain.invoke({"input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory."})
# chain.invoke({"input": "Buffer memory just stores the entire conversation"})
# chain.invoke({"input": "Buffer window memory stores the last k messages, dropping the rest."})
#
# res = chain.invoke({"input": "What is my name again?"})
# print(res['response'])

# ------------------ New: ConversationSummaryMemory vs RunnableWithMessageHistory --------------------------

# class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
#     messages: list[BaseMessage] = Field(default_factory=list)
#     llm: ChatOllama = Field(default_factory=ChatOllama)
#
#     def __init__(self, llm: ChatOllama):
#         super().__init__(llm=llm)
#
#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         """Add messages to the history, removing any messages beyond
#         the last `k` messages.
#         """
#         self.messages.extend(messages)
#         # construct the summary chat messages
#         summary_prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(
#                 "Given the existing conversation summary and the new messages, "
#                 "generate a new summary of the conversation. Ensuring to maintain "
#                 "as much relevant information as possible BUT keep the summary "
#                 "concise and no more than a short paragraph in length."
#             ),
#             HumanMessagePromptTemplate.from_template(
#                 "Existing conversation summary:\n{existing_summary}\n\n"
#                 "New messages:\n{messages}"
#             )
#         ])
#         # format the messages and invoke the LLM
#         new_summary = self.llm.invoke(
#             summary_prompt.format_messages(existing_summary=self.messages, messages=messages)
#         )
#         # replace the existing history with a single system summary message
#         self.messages = [SystemMessage(content=new_summary.content)]
#
#     def clear(self) -> None:
#         """Clear the history."""
#         self.messages = []
#
#
# chat_map = {}
#
#
# def get_chat_history(session_id: str, llm: ChatOllama) -> ConversationSummaryMessageHistory:
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = ConversationSummaryMessageHistory(llm=llm)
#     # return the chat history
#     return chat_map[session_id]
#
#
# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(
#             id="session_id",
#             annotation=str,
#             name="Session ID",
#             description="The session ID to use for the chat history",
#             default="id_default",
#         ),
#         ConfigurableFieldSpec(
#             id="llm",
#             annotation=ChatOllama,
#             name="LLM",
#             description="The LLM to use for the conversation summary",
#             default=llm,
#         )
#     ]
# )
#
# pipeline_with_history.invoke(
#     {"query": "Hi, my name is Josh"},
#     config={"session_id": "id_123", "llm": llm}
# )
#
# print('time1', chat_map["id_123"].messages)
#
# pipeline_with_history.invoke(
#     {"query": "I'm researching the different types of conversational memory."},
#     config={"session_id": "id_123", "llm": llm}
# )
#
# print('time2', chat_map["id_123"].messages)
#
# for msg in [
#     "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
#     "Buffer memory just stores the entire conversation",
#     "Buffer window memory stores the last k messages, dropping the rest."
# ]:
#     pipeline_with_history.invoke(
#         {"query": msg},
#         config={"session_id": "id_123", "llm": llm}
#     )
#
# print('time3', chat_map["id_123"].messages)
#
# res = pipeline_with_history.invoke(
#     {"query": "What is my name again?"},
#     config={"session_id": "id_123", "llm": llm}
# )
# print(res.content)


# # ------------------ Old: ConversationSummaryBufferMemory --------------------------

# memory = ConversationSummaryBufferMemory(
#     llm=llm,
#     max_token_limit=100,
#     return_messages=True
# )
#
# chain = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
#
# res = chain.invoke({"input": "Hi, my name is Josh"})
# print(res['response'])
#
#
# for i, msg in enumerate([
#     "I'm researching the different types of conversational memory.",
#     "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
#     "Buffer memory just stores the entire conversation",
#     "Buffer window memory stores the last k messages, dropping the rest."
# ]):
#     chain.invoke({"input": msg})

# ------------------ New: ConversationSummaryBufferMemory vs RunnableWithMessageHistory --------------------------

class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOllama = Field(default_factory=ChatOllama)
    k: int = Field(default_factory=int)

    def __init__(self, llm: ChatOllama, k: int):
        super().__init__(llm=llm, k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages and summarizing the messages that we
        drop.
        """
        existing_summary = None
        old_messages = None
        # see if we already have a summary message
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            print(">> Found existing summary")
            existing_summary: str | None = self.messages.pop(0)
        # add the new messages to the history
        self.messages.extend(messages)
        # check if we have too many messages
        if len(self.messages) > self.k:
            print(
                f">> Found {len(self.messages)} messages, dropping "
                f"latest {len(self.messages) - self.k} messages.")
            # pull out the oldest messages...
            old_messages = self.messages[:self.k]
            # ...and keep only the most recent messages
            self.messages = self.messages[-self.k:]
        if old_messages is None:
            print(">> No old messages to update summary with")
            # if we have no old_messages, we have nothing to update in summary
            return
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible BUT keep the summary "
                "concise and no more than a short paragraph in length."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])
        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )
        print(f">> New summary: {new_summary.content}")
        # prepend the new summary to the history
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}


def get_chat_history(
        session_id: str,
        llm: ChatOllama,
        k: int
) -> ConversationSummaryBufferMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    # return the chat history
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
            id="llm",
            annotation=ChatOllama,
            name="LLM",
            description="The LLM to use for the conversation summary",
            default=llm,
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

pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"session_id": "id_123", "llm": llm, "k": 4}
)
chat_map["id_123"].messages

for i, msg in enumerate([
    "I'm researching the different types of conversational memory.",
    "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
    "Buffer memory just stores the entire conversation",
    "Buffer window memory stores the last k messages, dropping the rest."
]):
    print(f"---\nMessage {i + 1}\n---\n")
    pipeline_with_history.invoke(
        {"query": msg},
        config={"session_id": "id_123", "llm": llm, "k": 4}
    )
