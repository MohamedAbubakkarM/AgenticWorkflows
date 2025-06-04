import os
from typing import Type, Dict, Any
import json
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

curr_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(curr_dir, "db")
persist_dir = os.path.join(db_dir, "chat_history")


llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key="my_secret_key")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="my_secret_key")


db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k":3})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)


qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. Use three sentences maximum and keep the answer "
    "concise."
    "If you don't know the answer, just say that you "
    "don't know."
    "\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


def store_message_in_chroma(user_input, ai_response):
    """Stores conversation history in ChromaDB"""
    db.add_documents([
        Document(page_content=user_input, metadata={"role": "user"}),
        Document(page_content=ai_response, metadata={"role": "assistant"})
    ])

def get_chat_history_from_chroma():
    """Retrieves stored chat history from ChromaDB"""
    results = db.similarity_search("", k=10)  # Retrieve the last 10 messages
    history = []
    
    for doc in results:
        role = doc.metadata.get("role", "unknown")
        message = doc.page_content
        if role == "user":
            history.append(HumanMessage(content=message))
        elif role == "assistant":
            history.append(AIMessage(content=message))

    return history

#Agent from here
agent_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.
    You are a agent who knows how to use tools efficiently for given {input}.
    The available tools are : {tool_names}
    {tools}
    If you know which tool to use, provide:
    Thought: Explain reasoning.
    Action: The tool to use.
    Action Input: The input for the tool.

    If no tool is required, provide:
    Final Answer: Your final response.

    {agent_scratchpad}
    """
)


class SearchInput(BaseModel):
    query: str = "should be a search query"


class SearchTool(BaseTool):
    name: str = Field(default="Simple_Search", description="Name of the search tool")
    description: str = Field(default="Useful when you need an answer for the current events.", description="Tool description")
    args_schema: Type[BaseModel] = SearchInput
    def _run(self, query:str) -> str:
        """Use the tool"""
        from tavily import TavilyClient
        client = TavilyClient(api_key="my_secret_key")
        results = client.search(query=query)
        return f"Search results for {query}:\n\n\n{results}\n"
    

tools = [
    SearchTool()
]


agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt=agent_prompt)

agent_exec = AgentExecutor(
    agent=agent,
    tools=tools,
    #verbose=True,
    handle_parsing_errors=True
)

def conversation():
    print("Start your query with my agent:\n")
    chat_history = []
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        if query.lower() == 'history':
            previous_chats = get_chat_history_from_chroma()
            for msg in previous_chats:
                role = "You" if isinstance(msg, HumanMessage) else "Agent"
                print(f"{role}: {msg.content}")
            continue

        
        result = rag_chain.invoke({"input": query, "chat_history": get_chat_history_from_chroma()})
        print("\nRAG response: ", result['answer'])
        response = agent_exec.invoke({"input": query, "chat_history": chat_history})
        ai_message = response['output'] if isinstance(response, dict) else response
        print(f"\nAgent Response: {ai_message}")

        store_message_in_chroma(query, ai_response=ai_message)

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=ai_message))



conversation()