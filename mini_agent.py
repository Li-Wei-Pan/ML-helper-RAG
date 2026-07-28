#Takes a question
# Has one tool — calculator
# Runs a loop — LLM decides action, tool executes, observation fed back
# Returns a final answer
import os
import re
from pydantic import Field, BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
OPENAI_KEY = os.getenv('OPEN_AI_KEY')
embedding_model = OpenAIEmbeddings(OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_KEY,
    temperature=0
)


class Tool(BaseModel):
    name: str
    description: str
    action: str
    observation: str

    class ConfigDict:
        arbitrary_types_allowed = True
    
SYSTEM_PROMPT = ' f"Extract only the mathematical expression from this question as a Python-evaluable string. Return ONLY the expression, nothing else: {question}"'


def calculator(expression):
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)  # safe — only digits and operators allowed
        return str(result)
    except:
        return "Error: Could not evaluate expression"

def math_agent(user_query):
    extraction = model.invoke( [{'role': 'user', "content": f"Extract only the mathematical expression from this question as a Python-evaluable string. Return ONLY the expression, nothing else: {user_query}"}])
    result = calculator(extraction)
    if result:
        return result
    return f'Failed to compute'

