# Tool Calling Guide for vLLM Server

## Overview

Your vLLM server is successfully running with tool calling capabilities enabled. The server is configured with:
- **Model**: Qwen3-30B-A3B
- **Port**: 8000
- **Tool Parser**: Hermes
- **Auto Tool Choice**: Enabled

## How Tool Calling Works

### 1. Architecture
Tool calling in LLMs follows this pattern:
```
User Request → LLM → Tool Call Decision → Tool Execution → Tool Results → LLM → Final Response
```

### 2. Key Components

#### Tool Definitions
Tools are defined using JSON schema format:
```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
      "type": "object",
      "properties": {
        "param_name": {
          "type": "string",
          "description": "Parameter description"
        }
      },
      "required": ["param_name"]
    }
  }
}
```

#### Tool Execution Flow
1. **Request with Tools**: Send user message + tool definitions to LLM
2. **Tool Call Response**: LLM responds with tool calls (if needed)
3. **Execute Tools**: Your application executes the requested tools
4. **Return Results**: Send tool results back to LLM
5. **Final Response**: LLM provides final answer using tool results

### 3. API Format

#### Basic Request
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "Your message"}],
    "tools": [...tool_definitions...],
    "tool_choice": "auto"
  }'
```

#### Tool Choice Options
- `"auto"`: LLM decides when to use tools
- `"none"`: Never use tools
- `{"type": "function", "function": {"name": "specific_tool"}}`: Force specific tool

## Testing Tool Calling

### 1. Manual Testing (curl)
```bash
# Test with weather and calculator tools
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "What is 15 * 23?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "Math expression"}
          },
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### 2. Python Testing
Use the provided test scripts:

#### Quick Test
```bash
python3 simple_tool_test.py
```

#### Comprehensive Test
```bash
python3 test_tool_calling.py
```

## Implementation Patterns

### 1. Tool Function Mapping
```python
FUNCTION_MAP = {
    "get_weather": get_weather_function,
    "calculate": calculate_function,
    "send_email": send_email_function
}

def execute_tool_call(tool_call):
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    if function_name in FUNCTION_MAP:
        result = FUNCTION_MAP[function_name](**arguments)
        return {
            "tool_call_id": tool_call["id"],
            "role": "tool",
            "content": json.dumps(result)
        }
```

### 2. Error Handling
```python
try:
    result = FUNCTION_MAP[function_name](**arguments)
except Exception as e:
    result = {"error": f"Tool execution failed: {str(e)}"}
```

### 3. Conversation Flow
```python
# 1. Initial request
messages = [{"role": "user", "content": user_input}]
response = call_llm(messages, tools=TOOLS)

# 2. Handle tool calls
if response["choices"][0]["message"].get("tool_calls"):
    # Execute tools and add results to conversation
    for tool_call in tool_calls:
        tool_result = execute_tool_call(tool_call)
        messages.append(tool_result)
    
    # 3. Get final response
    final_response = call_llm(messages)
```

## Best Practices

### 1. Tool Design
- **Clear descriptions**: Make tool purposes obvious
- **Specific parameters**: Use detailed parameter descriptions
- **Error handling**: Always handle tool execution failures
- **Security**: Validate inputs, especially for dangerous operations

### 2. Performance
- **Parallel execution**: Execute multiple tools concurrently when possible
- **Caching**: Cache tool results for repeated calls
- **Timeouts**: Set reasonable timeouts for tool execution

### 3. User Experience
- **Progress indicators**: Show when tools are being executed
- **Fallback responses**: Handle cases where tools fail
- **Clear formatting**: Present tool results in user-friendly format

## Common Tool Types

### 1. Information Retrieval
- Web search
- Database queries
- API calls
- File reading

### 2. Computation
- Mathematical calculations
- Data processing
- Statistical analysis
- Unit conversions

### 3. Actions
- Sending emails
- File operations
- API calls
- System commands

### 4. External Services
- Weather APIs
- Translation services
- Image generation
- Payment processing

## Troubleshooting

### Common Issues
1. **Tools not called**: Check tool descriptions and user intent alignment
2. **Invalid arguments**: Verify parameter schemas and validation
3. **Execution errors**: Implement proper error handling and logging
4. **Performance issues**: Consider tool execution timeouts and caching

### Debugging Tips
- Log all tool calls and results
- Test tools independently before integration
- Use simple test cases first
- Monitor server logs for errors

## Server Configuration

Your current vLLM server configuration:
```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-log-requests \
  --uvicorn-log-level warning \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.8 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Key tool-related flags:
- `--enable-auto-tool-choice`: Enables automatic tool selection
- `--tool-call-parser hermes`: Uses Hermes format for tool calling

## Next Steps

1. **Extend tools**: Add more domain-specific tools for your use case
2. **Integration**: Integrate tool calling into your applications
3. **Monitoring**: Set up logging and monitoring for tool usage
4. **Security**: Implement proper authentication and authorization
5. **Optimization**: Profile and optimize tool execution performance 