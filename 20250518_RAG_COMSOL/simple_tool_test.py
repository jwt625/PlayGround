#!/usr/bin/env python3
"""
Simple Tool Calling Test for vLLM Server
Quick test script for experimenting with tool calling.
"""

import json
import requests

def test_tool_calling():
    """Simple test of tool calling functionality."""
    
    # Define a simple calculator tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "The mathematical operation to perform"
                        }
                    },
                    "required": ["operation"]
                }
            }
        }
    ]
    
    # Test message
    messages = [
        {
            "role": "user",
            "content": "Calculate 42 * 17 + 8 for me please"
        }
    ]
    
    # Make request to vLLM
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-30b-a3b",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
    )
    
    result = response.json()
    
    print("=== Tool Calling Test Result ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # Check if tools were called
    if "choices" in result and result["choices"]:
        message = result["choices"][0]["message"]
        if "tool_calls" in message and message["tool_calls"]:
            print("\n✅ Tool calling is working!")
            print("Tools called:")
            for tool_call in message["tool_calls"]:
                print(f"  - {tool_call['function']['name']}: {tool_call['function']['arguments']}")
        else:
            print("\n❌ No tools were called")
    else:
        print("\n❌ Invalid response format")

if __name__ == "__main__":
    test_tool_calling() 