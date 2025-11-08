#!/usr/bin/env python3
"""
GLM-4.6 FastAPI server with authentication
"""
import os
import secrets
import time
import uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Authentication
security = HTTPBearer()
AUTH_TOKEN = os.getenv("GLM_AUTH_TOKEN")
if not AUTH_TOKEN:
    raise ValueError("GLM_AUTH_TOKEN not found in environment variables. Please check your .env file.")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the authentication token"""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# OpenAI-compatible Request/Response models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message (system, user, assistant)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="glm-4.6", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=8192, ge=1, le=16384, description="Maximum output tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# Legacy endpoint models (for backward compatibility)
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=8192, ge=1, le=16384, description="Maximum output tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=1, le=100, description="Top-k sampling")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    finish_reason: str

# Global model instance
llm_model = None

def download_model():
    """Download GLM-4.6 model (uses cache if available)"""
    model_name = "zai-org/GLM-4.6"
    cache_dir = "./models"

    print(f"Loading {model_name} (will use cache if available)...")
    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir
    )
    print(f"Model ready at: {local_path}")
    return local_path

def load_model():
    """Load model with vLLM"""
    global llm_model
    if llm_model is None:
        print("Loading GLM-4.6 model with vLLM...")
        model_path = download_model()
        llm_model = LLM(
            model=model_path,
            tensor_parallel_size=8,  # Use all 8 GPUs
            trust_remote_code=True
        )
        print("Model loaded successfully!")
    return llm_model

# FastAPI app
app = FastAPI(
    title="GLM-4.6 API Server",
    description="High-performance GLM-4.6 inference server with vLLM",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "GLM-4.6"}

@app.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "glm-4.6",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "zai-org",
                "permission": [],
                "root": "glm-4.6",
                "parent": None
            }
        ]
    }

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages to a single prompt string"""
    prompt_parts = []
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"Human: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")

    # Add assistant prefix for completion
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)

def count_tokens(text: str) -> int:
    """Simple token counting (approximate)"""
    return len(text.split())

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    token: str = Depends(verify_token)
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        llm = load_model()

        # Convert messages to prompt
        prompt = format_chat_prompt(request.messages)

        sampling_params = SamplingParams(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 8192,
            stop=request.stop
        )

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]

        # Count tokens
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(output.outputs[0].text)

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=output.outputs[0].text.strip()
                    ),
                    finish_reason=output.outputs[0].finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    token: str = Depends(verify_token)
):
    """Legacy generate endpoint (for backward compatibility)"""
    try:
        llm = load_model()

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop
        )

        outputs = llm.generate([request.prompt], sampling_params)
        output = outputs[0]

        return GenerateResponse(
            text=output.outputs[0].text,
            prompt=request.prompt,
            finish_reason=output.outputs[0].finish_reason
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    print(f"Starting GLM-4.6 API server...")
    print(f"Auth token: {AUTH_TOKEN}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
