from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import asyncio
from agent import rag_search, calculator  # import tools
from dotenv import load_dotenv
import os
import secrets

load_dotenv()
security_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)): 
    MCP_KEY = os.getenv("MCP_API_KEY")
    if not MCP_KEY or not secrets.compare_digest(credentials.credentials, MCP_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
    
app = FastAPI(dependencies= [Depends(verify_token)])


@app.get('/tools')
def list_tools():
    return {
        "tools": [
            {
                "name": "rag_search",
                "description": """Search the ML knowledge base for information about machine learning 
                concepts, algorithms, and techniques. Use when the question is about 
                ML topics.""",
                "parameters": {"query": "string"}
            },
            {
                "name": "calculator", 
                "description": "Evaluate mathematical expressions. Use for any numerical calculations. Only use basic operators: + - * / ** (). Do NOT use for ML concept questions.",
                "parameters": {"expression": "string"}
            }
        ]
    }

class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict

AVAILABLE_TOOLS = {
    "rag_search": rag_search,
    "calculator": calculator
}


@app.post('/execute')
async def execute_tool(request:ToolRequest):
    if request.tool_name not in AVAILABLE_TOOLS:
        raise HTTPException(status_code= 404, detail = f"{request} tool not found")
    fn = AVAILABLE_TOOLS[request.tool_name]
    if asyncio.iscoroutinefunction(fn):
        result = await fn(**request.arguments)
    else:
        result = fn(**request.arguments)
    return {"result": str(result)}
