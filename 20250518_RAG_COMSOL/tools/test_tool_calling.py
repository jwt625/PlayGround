#!/usr/bin/env python3
"""
Tool Calling Test Script for vLLM Server
Demonstrates how to implement tool calling with a local LLM server.
"""

import json
import requests
import time
from typing import Dict, List, Any

# Configuration
VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "qwen3-30b-a3b"

def call_llm(messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto") -> Dict:
    """Make a request to the vLLM server."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    
    response = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    return response.json()

def get_weather(location: str) -> Dict:
    """Mock weather function."""
    # In a real implementation, this would call a weather API
    mock_weather = {
        "New York": {"temperature": "72°F", "condition": "Partly cloudy", "humidity": "65%"},
        "London": {"temperature": "15°C", "condition": "Rainy", "humidity": "80%"},
        "Tokyo": {"temperature": "25°C", "condition": "Sunny", "humidity": "55%"}
    }
    
    return mock_weather.get(location, {
        "temperature": "Unknown", 
        "condition": "Data not available", 
        "humidity": "Unknown"
    })

def calculate(expression: str) -> Dict:
    """Safe calculator function."""
    try:
        # Only allow basic mathematical operations for safety
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return {"result": result, "expression": expression}
        else:
            return {"error": "Invalid characters in expression"}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}

def send_email(to: str, subject: str, body: str = "") -> Dict:
    """Mock email sending function."""
    # In a real implementation, this would send an actual email
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "body": body,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def web_search(query: str) -> Dict:
    """Mock web search function."""
    # In a real implementation, this would call a search API
    mock_results = {
        "Python programming": [
            "Python is a high-level programming language known for its simplicity and readability.",
            "It's widely used in web development, data science, AI, and automation.",
            "Python was created by Guido van Rossum and first released in 1991."
        ]
    }
    
    return {
        "query": query,
        "results": mock_results.get(query, ["No specific results found for this query."])
    }

# Tool definitions for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. New York, London, Tokyo"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Email address of the recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line of the email"
                    },
                    "body": {
                        "type": "string",
                        "description": "Body content of the email"
                    }
                },
                "required": ["to", "subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Function mapping
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "send_email": send_email,
    "web_search": web_search
}

def execute_tool_call(tool_call: Dict) -> Dict:
    """Execute a tool call and return the result."""
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    if function_name in FUNCTION_MAP:
        try:
            result = FUNCTION_MAP[function_name](**arguments)
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": json.dumps(result)
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": json.dumps({"error": f"Tool execution failed: {str(e)}"})
            }
    else:
        return {
            "tool_call_id": tool_call["id"],
            "role": "tool",
            "content": json.dumps({"error": f"Unknown function: {function_name}"})
        }

def chat_with_tools(user_message: str) -> None:
    """Complete chat interaction with tool calling."""
    print(f"User: {user_message}")
    print("-" * 50)
    
    # Initial conversation
    messages = [{"role": "user", "content": user_message}]
    
    # Get LLM response
    response = call_llm(messages, tools=TOOLS)
    
    if "choices" not in response or not response["choices"]:
        print("Error: No response from LLM")
        return
    
    choice = response["choices"][0]
    assistant_message = choice["message"]
    
    # Add assistant message to conversation
    messages.append(assistant_message)
    
    # Check if the assistant wants to call tools
    if assistant_message.get("tool_calls"):
        print("Assistant is calling tools...")
        
        # Execute each tool call
        for tool_call in assistant_message["tool_calls"]:
            print(f"Calling: {tool_call['function']['name']} with {tool_call['function']['arguments']}")
            tool_result = execute_tool_call(tool_call)
            messages.append(tool_result)
            print(f"Tool result: {tool_result['content']}")
        
        # Get final response with tool results
        final_response = call_llm(messages)
        if "choices" in final_response and final_response["choices"]:
            final_message = final_response["choices"][0]["message"]["content"]
            print(f"\nAssistant: {final_message}")
        else:
            print("Error: No final response from LLM")
    else:
        # No tools called, just show the response
        print(f"Assistant: {assistant_message.get('content', 'No content in response')}")

def main():
    """Run interactive tool calling tests."""
    print("=== vLLM Tool Calling Test ===")
    print("Testing various tool calling scenarios...\n")
    
    # Test scenarios
    test_cases = [
        "What's the weather like in New York and calculate 25 * 4?",
        "Send an email to alice@example.com with subject 'Project Update' and search for information about Python programming",
        "Calculate the result of (100 + 50) / 3 and tell me the weather in Tokyo",
        "Just have a normal conversation - what's your favorite color?"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        chat_with_tools(test_case)
        print("\n")

if __name__ == "__main__":
    main() 