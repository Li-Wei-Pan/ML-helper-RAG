from pydantic import BaseModel
from typing import Callable, Any
from utils import embed_single, build_context_with_budget
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import re
import asyncio
import datetime
import sqlalchemy
import json
from utils import cross_reranking, evalute_with_reranking, get_cross_encoder

load_dotenv()
OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
emb_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_KEY,
    temperature=0
)

client = OpenAI(api_key=OPENAI_KEY)

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
# for api.py only
# async_engine = create_engine(CONNECTION_STRING.replace('postgresql://', 'postgresql+asyncpg://'))

sync_engine = create_engine(CONNECTION_STRING)

#async_db = PGVector(embeddings=emb_model, collection_name='ML knowledge', connection = async_engine, create_extension=False)


db = PGVector(
    embeddings=emb_model,
    collection_name='ML knowledge',
    connection=sync_engine  # test
)

class tool(BaseModel):
    name: str
    description: str
    function :Callable
    
    class ConfigDict:
        arbitrary_types_allowed = True

async def rag_search( query: str)-> str:
    string = ""
    emb_query = await embed_single(emb_model, query)
    results = db.similarity_search_with_score_by_vector(emb_query, 3)
    filtered_results = [i for i in results if  i[1] <= 0.25]
    if not filtered_results:
        return f"NO chunks found"
        

    for doc, score in filtered_results:
        string += doc.page_content+'\n\n'
    return string.strip()

def calculator(expression):
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)  # safe — only digits and operators allowed
        return str(result)
    except:
        return "Error: Could not evaluate expression"



memory_store = []
def save_to_memory(question: str, answer: str, session_id : str):

    memory_store.append({'question': question, 'answer': answer, 'session_id': session_id, 'timestamp': datetime.datetime.utcnow().isoformat()})
    return memory_store

def retrieve_memory( n: int  =3):
    if not memory_store:
        return ''
    
    retrieved_memory = memory_store[-n:]
    formatted = ""
    for i in retrieved_memory:
        formatted += f"Q: {i['question']} \n A: {i['answer']}\n\n"
    return formatted.strip()

ML_PROMPT = """You are an ML specialist agent with access to one tool:

    1. rag_search(query) — Search the ML knowledge base for information about 
    machine learning concepts, algorithms, and techniques.

    To use the tool, respond in this EXACT format:
    THOUGHT: your reasoning
    ACTION: rag_search
    INPUT: your search query

    When done:
    THOUGHT: I have enough information
    FINAL: your answer
    """

async def machinelearing_agent(question:str):
    # run_agent with only rag_search tool
    # detect "No relevant content" in result
    # return clear error if retrieval failed
    
    result,  content = await run_agent(
        question,
        tools={"rag_search": rag_search},
        max_steps=3,
        system_prompt= ML_PROMPT
    )
    if "No relevant content" in result:
        return f"Error: {result}"
    return result



async def math_agent(question):
    # specialized agent — only uses calculator
    # detect "Error" in result
    # return clear error if calculation failed
    extraction = await model.ainvoke([{
        "role": "user",
        "content": f"Extract only the mathematical expression from this question as a Python-evaluable string. Return ONLY the expression, nothing else: {question}"
    }])
    expression = extraction.content.strip()
    cal_result =  calculator(expression)   
    if cal_result and "Error" not in cal_result:
        return cal_result
    
    return f"Failed to do math operations" 

async def orchestrator(question, max_attempts):
    ORCHESTRATOR_PROMPT = """You are an orchestrator that routes questions 
    to the right specialist agent.

    Available agents:
    1. ml_agent — answers questions about machine learning concepts,
                algorithms, and techniques
    2. math_agent — handles mathematical calculations and numerical problems

    Respond in EXACTLY this format:
    AGENT: agent_name
    REASON: one sentence explaining why
    If a specialist returns an error or no results:
    - Try a different agent if the question could be answered another way
    - If no agent can help, respond with AGENT: none and explain why
    """
    execution_history = f'User Question: {question}\n'
    for attempt in range(max_attempts):
    # Step 1 — LLM decides which agent to call
        routing_decision = await model.ainvoke([
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content": execution_history}
    ])
        routing_content = routing_decision.content
    # Step 2 — route to correct specialist
        agent_match = re.search(r"AGENT: \s*(.*)", routing_content, re.IGNORECASE)
    # Step 3 — return result
        if not agent_match:
            return f"Orchestrator failed to format routing decision"
    # Step 4 — handle unknown agent or none
        selected_agent = agent_match.group(1).strip().lower()
        if selected_agent == "ml_agent":
            result = await machinelearing_agent(question)
        elif selected_agent == 'math_agent':
            result = await math_agent(question)

        elif selected_agent == 'none':
            return f'No suitable agent found. Orchestrator reassoning: {routing_decision}'
        else:
            result = "Error: Invalid agent selected."
        if 'Error' in result:
            execution_history += f"Attempt{attempt +1} - called {selected_agent}: {result}\n"
            continue

    # Step 5 — return specialist result
        return result
    return "Orchestration failed: Maximum attempts reached without a valid answer."

async def run_agent(question, max_steps=5, session_id = None,  tools = None, system_prompt =None):
    memory_context = retrieve_memory(n = 3)
    print(f'memory context: {memory_context[:100] if memory_context else "EMPTY" }')
    system_content = SYSTEM_PROMPT
    if memory_context:
        system_content += f"\n\n Relevant past conversations: \n {memory_context}"

    # Step 1 — initialize message history
    messages = [{'role': 'system', 'content': system_content}, 
                {'role': 'user', 'content': question}]
    # Step 2 — define available tools as a dict
    
    if tools is None:
        tools = {'rag_search': rag_search, 'calculator': calculator}

    base_prompt = system_prompt if system_prompt else SYSTEM_PROMPT    
    # Step 3 — start the ReAct loop (max_steps iterations)
    #var to track rag_search()
    retrieved_context = ""
    for step in range(max_steps):

        # Step 3a — call the LLM with current messages
        response = await model.ainvoke(messages)
        content = response.content

        # Step 3b — check if LLM returned a FINAL answer
        if 'FINAL' in content:
            final_answer = content.split('FINAL')[-1].strip()
    
            save_to_memory(question, final_answer, session_id)
            return final_answer,  retrieved_context

        # Step 3c — check if LLM returned an ACTION

        if "ACTION" not in content:
            return content, ""
        
        # Step 3d — parse ACTION and INPUT from content
        lines = content.strip().split("\n")
        action = None
        tool_input = None

        for line in lines:
            if line.startswith('ACTION'):
                action = line.replace("ACTION:", "").strip()
            if  line.startswith('INPUT'):
                tool_input = line.replace("INPUT:", "").strip()



        if not action:
            observation = f'No action found'
            print(f"Step {step+1} | ACTION: None | INPUT: {tool_input}")
            print(f"OBSERVATION: {observation}")
        
        elif not tool_input:
            observation = f"Error: no input provided for tool {action}"
            print(f"Step {step+1} | ACTION: {action} | INPUT: None")
            print(f"OBSERVATION: {observation}")

        elif action not in tools:
            observation = f"Error: tool {action} not found"
            print(f"Step {step+1} | ACTION: {action} | INPUT: {tool_input}")
            print(f"OBSERVATION: {observation}")
        

        else:
            tool_fn = tools[action]
            if asyncio.iscoroutinefunction(tool_fn):
                observation = await tool_fn(tool_input)
            else:
                observation = tool_fn(tool_input)
            print(f"Step {step+1} | ACTION: {action} | INPUT: {tool_input}")
            print(f"OBSERVATION: {str(observation)[:200]}")
            if action == "rag_search":
                retrieved_context = observation 

        messages.append({'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': f'OBSERVATION: {observation}'})
    save_to_memory(question, 'Agent reached maximum steps', session_id)
    return 'Agent reached maximum steps without finding an answer', retrieved_context

async def llm_judge(user_query, retrieved_chunks, agent_answer):
    judge_prompt = f"""You are an objective evaluation judge. Your task is to evaluate an AI agent's answer based ONLY on the provided retrieved context. You must ignore all outside knowledge.

    Retrieved Context:
    {retrieved_chunks}

    User Question:
    {user_query}

    Agent Answer:
    {agent_answer}

    Grade each category on a scale of 1 to 5:
    1. Faithfulness (1-5): Every claim in the answer must be supported by the retrieved context.
    2. Relevance (1-5): The answer directly addresses the user question.
    3. Hallucination (1-5): Scan for entities or claims in the answer that are absent from the context (5 = no hallucination, 1 = severe hallucination).
    """
    structured_model = model.with_structured_output(JudgeOutput)
    eval_result = structured_model.invoke(input = judge_prompt)

    return  eval_result
    
# # --- EXECUTION TEST ---

# async def main():
#     question = "What is cross validation?"
#     answer, context = await  run_agent(question)
#     print("\n--- AGENT RESPONSE ---")
#     print("ANSWER:", answer)
#     print("CONTEXT:", context[:150])

#     print("\n--- RUNNING LLM JUDGE ---")
#     judge_result = await llm_judge(question, context, answer)
#     print("JUDGE SCORES:", judge_result)

# if __name__ == "__main__":
#     asyncio.run(main())
