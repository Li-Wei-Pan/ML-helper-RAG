from pydantic import BaseModel
from typing import Callable, Any
from utils import embed_single, build_context_with_budget
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
import re
import asyncio

load_dotenv()
OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
emb_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_KEY,
    temperature=0
)

SYSTEM_PROMPT = """You are an AI agent with access to the following tools:

1. rag_search(query) — Search the ML knowledge base for information about 
   machine learning concepts, algorithms, and techniques.
   Use this when the question is about ML topics.

2. calculator(expression) — Evaluate mathematical expressions.
   Use this for any numerical calculations.
   Only use basic operators: + - * / ** ()

To use a tool, respond in this EXACT format:
THOUGHT: your reasoning about what to do next
ACTION: tool_name
INPUT: tool input

When you have enough information to answer, respond in this EXACT format:
THOUGHT: I have enough information to answer
FINAL: your final answer here

You must always start with a THOUGHT.
Never make up information — only use what tools return.
"""

# initialize async engine
async_engine = create_async_engine(CONNECTION_STRING.replace('postgresql://', 'postgresql+asyncpg://'))
db = PGVector(embeddings=emb_model, collection_name='ML knowledge', connection = async_engine, create_extension=False)


class tool(BaseModel):
    name: str
    description: str
    function :Callable
    
    class Config:
        arbitrary_types_allowed = True

async def rag_search( query: str)-> str:
    string = ""
    emb_query = await embed_single(emb_model, query)
    results = await db.asimilarity_search_with_score_by_vector(emb_query, 3)
    filtered_results = [i for i in results if  i[1] <= 0.25]
    if not filtered_results:
        return f"NO chunks found"
        

    for doc, score in filtered_results:
        string += doc.page_content+'\n\n'
    return string

def calculator(expression):
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)  # safe — only digits and operators allowed
        return str(result)
    except:
        return "Error: Could not evaluate expression"

async def run_agent(question, max_steps):
    # Step 1 — initialize message history
    # messages should be a list with two entries:
    # - system message containing SYSTEM_PROMPT
    # - user message containing the question
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, 
                {'role': 'user', 'content': question}]
    history = []
    # Step 2 — define available tools as a dict
    tools = {'rag_search': rag_search, 'calculator': calculator}
    # key = tool name string
    # value = the actual function
    # hint: {"rag_search": rag_search, ...}
    
    # Step 3 — start the ReAct loop (max_steps iterations)
    for step in range(max_steps):
        
        # Step 3a — call the LLM with current messages
        response = await model.ainvoke(messages)
        # use await model.ainvoke(messages)
        # extract .content from response
        content = response.content

        # Step 3b — check if LLM returned a FINAL answer
        # if "FINAL:" is in content → extract and return it
        # hint: content.split("FINAL:")[-1].strip()
        if 'FINAL' in content:
            return content.split('FINAL:')[-1].strip()

        # Step 3c — check if LLM returned an ACTION
        # if "ACTION:" not in content → LLM broke format, return content directly
        if "ACTION" not in content:
            return content
        
        # Step 3d — parse ACTION and INPUT from content
        lines = content.strip().split("\n")
        action = None
        tool_input = None
        for line in lines:
            if line.startswith('ACTION'):
                action = line.replace("ACTION:", "").strip()
            if  line.startswith('INPUT'):
                tool_input = line.replace("INPUT:", "").strip()
            
    
        if action not in tools:
            observation = f"Error: tool {action} not found"
        else:
            tool_fn = tools[action]
            if asyncio.iscoroutinefunction(tool_fn):
                observation = await tool_fn(tool_input)
            else:
                observation = tool_fn(tool_input)
            print(f"Step {step+1} | ACTION: {action} | INPUT: {tool_input}")
            print(f"OBSERVATION: {str(observation)[:200]}")

            messages.append({'role': 'assistant', 'content': content})
            messages.append({'role': 'user', 'content': f'observation: {observation}'})
        
    return 'Agent reached maximum steps without finding an answer'
