import asyncio
import os
import re
import json
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_postgres import PGVector
from utils import embed_single, build_context_with_budget
from agent import rag_search, calculator, _db, get_db, emb_model, model


load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('MCP_API_KEY')}"}

OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
MCP_SERVERS = {
    'http://localhost:8001',
    'http://localhost:8002'
}
async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    server = TOOL_SERVER_MAP.get(tool_name)
    if server is None:
        return f'Error Tool: {tool_name} not found'
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{server}/execute",
            json={"tool_name": tool_name, "arguments": arguments},
            timeout=30.0,
            headers= headers
        )
        if response.status_code == 404:
            return f"Error: tool '{tool_name}' not found"
        if response.status_code != 200:
            return f"Error: MCP server returned {response.status_code}"
        
        return response.json()["result"]


async def run_mcp_agent(question: str, max_steps: int =5) -> str:
    messages = [
        {'role': 'system', 'content': "You are a helpful ML assistant. Use the available tools to answer questions accurately."},
        {'role': 'user', 'content': question}    
    ] 
    discovered_tools = await discover_and_build_tools()
    
    for step in range(max_steps):
        response = await model.ainvoke(messages, tools = discovered_tools)

        if not response.tool_calls:
            return response.content # no final tool call then return result
        messages.append({
            "role": "assistant", 
            "content": response.content or "",
            "tool_calls": response.tool_calls
        })
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            print(f"Step { step +1}| TOOL: {tool_name}| ARGS: {tool_args}")

            result = await call_mcp_tool(tool_name, tool_args)

            print(f"RESULT: {str(result)[:200]}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result)
            })
    
    return "Agent reached maximum steps"


def convert_openai_schema(server_tool: dict)-> dict:
    properties = {}
    for param_name, param_type in server_tool["parameters"].items():
        properties[param_name] = {"type": param_type}

    return {
        "type": "function",
        "function": {
            "name": server_tool["name"],
            "description": server_tool["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(server_tool["parameters"].keys())
            }
        }
    }

TOOL_SERVER_MAP = {}
async def discover_and_build_tools()-> list:
    result = []
    seen_names = set()
    TOOL_SERVER_MAP.clear()

    for server in MCP_SERVERS:
        try:
            async with httpx.AsyncClient() as client:
                response =await client.get(f"{server}/tools",headers = headers, timeout = 30)
                if response.status_code != 200:
                    print(f"warning: {server} returned {response.status_code}")
                    continue

                extracted_tools = response.json()['tools']
                added = 0
                for tool in extracted_tools:
                    schema = convert_openai_schema(tool)
                    tool_name = schema['function']['name']
                    if tool_name not in seen_names:
                        result.append(schema)
                        seen_names.add(tool_name)
                        TOOL_SERVER_MAP[tool_name] = server
                        added += 1

                print(f"""{added} of tools found in this server: {server}""")
                pass
        except Exception as e:
            print(f'warning: could not reach {server}:{e}')
    return result
