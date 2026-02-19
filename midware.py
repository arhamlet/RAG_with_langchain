import requests

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from dataclasses import dataclass


load_dotenv()


@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful and very concise assistant"
    match user_role:
        case "expert":
            return f'{base_prompt} provide detailed technical responses'
        case "beginner":
            return f'{base_prompt} keep explanation simple and basic'
        case "child":
            return f'{base_prompt} explain everything as if you were literally talking to a 5 years old'
        case _:
            return base_prompt

agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware=[user_role_prompt],
    context_schema=Context,

)

response = agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "explain PCA."
            }
        ]
    }, context=Context(user_role="child")
)


print(response)