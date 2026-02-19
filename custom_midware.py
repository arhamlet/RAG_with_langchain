import requests
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt, AgentMiddleware, AgentState



load_dotenv()


@dataclass
class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("before agent triggered")

    def before_model(self, state: AgentState, runtime):
        print("before model")

    def after_model(self, state: AgentState, runtime):
        print("after model")

    def after_agent(self, state: AgentState, runtime):
        print("after agent time:", time.time() - self.start_time)


agent = create_agent(
    model = 'gpt-4o-mini',
    middleware = [HooksDemo()],
)

response = agent.invoke(
    {"messages":
         [
             SystemMessage("You are a helpful assistant."),
             HumanMessage("What is PCA?"),
         ]}
)


print(response['messages'][-1].content)
