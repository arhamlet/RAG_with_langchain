import requests

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool




load_dotenv()

embeddings  = OpenAIEmbeddings(
    model = "text-embedding-3-large",
)

texts = [
    "Apple makes very good computers.",
    "I  believe apple is innovative.",
    "I love apples.",
    "I am a fan of macbooks.",
    "I enjoy oranges.",
    "I like Lenovo thinkpads.",
    "I think pear tastes very good."
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)
# print(vector_store.similarity_search("Apples are my favorite food.", k=7))
print(vector_store.similarity_search("Linux is a great operating system.", k=7))

retriever = vector_store.as_retriever(search_kwargs = {"k": 3})
retriever_tool = create_retriever_tool(retriever, name="knowledge_base_search", description = "Search the given database for information")

agent = create_agent(
    model = "gpt-4.1-mini",
    tools=[retriever_tool],
    system_prompt="you are a helpful ai assistant. Whenever something is asked about computers or fruits, first use the Knowledge base search tool to retrieve the context and answer succinctly. Maybe you have to use the tool multiple times before answering.",

)
response = agent.invoke(
    {
        "messages":[

            { "role": "user",
            "content": "What fruits does the person  like and which computers does he like?"
              }
        ]
    }
)
print(response)
print(response['messages'][-1].content)
