# DevLog-004: Model Evaluation Setup and Testing Plan

**Date**: 2025-10-08  
**Status**: Ready for Handoff  
**Location**: `/home/ubuntu/GitHub/PlayGround/20251007_GLM_4p5/glm-4.5-air-setup`

## Objective

Set up and run a comprehensive evaluation comparing two LLM endpoints:
1. **GLM 4.5** - Running locally on vLLM server
2. **Llama 4 Maverick** - Running on Lambda internal API

## Network Topology Discovery

### Current Instance
- **IP**: 192.222.54.152
- **Location**: `/home/ubuntu/GitHub/PlayGround/20251007_GLM_4p5/glm-4.5-air-setup`
- **Access**: GLM 4.5 endpoint (local)

### GLM 4.5 Endpoint
- **URL**: http://192.222.54.152:8000/v1
- **Status**: ✓ Working from current instance
- **Auth**: Bearer token authentication
- **API Type**: vLLM completions endpoint

### Lambda (Llama 4 Maverick) Endpoint
- **URL**: https://internal-inference-api.bugnest.net/v1
- **IP**: 192.222.55.178
- **Status**: ✗ NOT accessible from current instance (192.222.54.152)
- **Issue**: No route to host - network isolation between subnets
- **Auth**: Bearer token authentication
- **API Type**: OpenAI-compatible chat completions endpoint
- **Model Name**: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`

### Network Issue Details

```bash
# DNS Resolution: ✓ Works
$ nslookup internal-inference-api.bugnest.net
Name:	internal-inference-api.bugnest.net
Address: 192.222.55.178

# Ping: ✗ Fails
$ ping -c 3 internal-inference-api.bugnest.net
From 192.222.54.152 icmp_seq=1 Destination Host Unreachable
100% packet loss

# Curl: ✗ Fails
curl: (7) Failed to connect to 192.222.55.178 port 443: No route to host
```

**Root Cause**: Network routing/firewall between subnet 192.222.54.x and 192.222.55.x is blocked.

**Confirmed**: Lambda endpoint works from external machines (tested by user).

## Files Created

### 1. Environment Configuration: `.env`
```bash
# GLM 4.5 vLLM Server Configuration
GLM_API_BASE=http://192.222.54.152:8000/v1
GLM_API_KEY=glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw

# Llama 4 Maverick Configuration
LAMBDA_API_BASE=https://internal-inference-api.bugnest.net/v1
LAMBDA_API_KEY=d77395f0d45d6dd9a470196c78a69ba0dd184363e4f7825938f3aed5545bf69a
LAMBDA_MODEL=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
```

### 2. Endpoint Test Script: `test_endpoints.py`
- Tests both GLM and Lambda endpoints
- Validates connectivity and API responses
- Uses python-dotenv for configuration

**Run with**:
```bash
cd /home/ubuntu/GitHub/PlayGround/20251007_GLM_4p5/glm-4.5-air-setup
uv run python test_endpoints.py
```

**Current Results**:
- GLM 4.5: ✓ PASSED
- Lambda: ✗ FAILED (network issue)

### 3. Evaluation Script: `eval_models.py`
Comprehensive evaluation script that:
- Runs 10 test cases across different categories:
  - General Knowledge (2 tests)
  - Math (1 test)
  - Science (1 test)
  - Programming (1 test)
  - History (1 test)
  - Reasoning (1 test)
  - Creative Writing (1 test)
  - Language (1 test)
  - Problem Solving (1 test)
- Queries both models with identical prompts
- Measures response time for each query
- Saves results to timestamped JSON file: `eval_results_YYYYMMDD_HHMMSS.json`
- Generates summary statistics

**Run with**:
```bash
cd /home/ubuntu/GitHub/PlayGround/20251007_GLM_4p5/glm-4.5-air-setup
uv run python eval_models.py
```

### 4. Test Commands Reference: `test_commands.sh`
Bash script with curl commands for both endpoints (for manual testing).

## API Endpoint Details

### GLM 4.5 API Format

**Endpoint**: `POST http://192.222.54.152:8000/v1/completions`

**Headers**:
```
Authorization: Bearer glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw
Content-Type: application/json
```

**Request Body**:
```json
{
  "prompt": "[gMASK]<sop><|system|>\nYou are a helpful AI assistant.<|user|>\nWhat is the capital of France?<|assistant|>\n",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Important**: GLM requires special prompt formatting with `[gMASK]<sop>` prefix and role tags.

**Example Response**:
```json
{
  "choices": [{
    "text": "<think>...</think>The capital of France is Paris."
  }]
}
```

### Lambda (Llama 4 Maverick) API Format

**Endpoint**: `POST https://internal-inference-api.bugnest.net/v1/chat/completions`

**Headers**:
```
Authorization: Bearer d77395f0d45d6dd9a470196c78a69ba0dd184363e4f7825938f3aed5545bf69a
Content-Type: application/json
```

**Request Body**:
```json
{
  "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Important**: Must use exact model name `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` (not `llama4-maverick`).

**Example Response**:
```json
{
  "choices": [{
    "message": {
      "content": "The capital of France is Paris."
    }
  }]
}
```

## Test Commands

### Test GLM Endpoint (from current instance)
```bash
curl -X POST "http://192.222.54.152:8000/v1/completions" \
  -H "Authorization: Bearer glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "[gMASK]<sop><|system|>\nYou are a helpful AI assistant.<|user|>\nWhat is the capital of France?<|assistant|>\n",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Test Lambda Endpoint (from machine with network access)
```bash
curl -X POST "https://internal-inference-api.bugnest.net/v1/chat/completions" \
  -H "Authorization: Bearer d77395f0d45d6dd9a470196c78a69ba0dd184363e4f7825938f3aed5545bf69a" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Can you tell me a joke?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Dependencies

Already installed in the project:
- `requests` - HTTP client
- `python-dotenv` - Environment variable management

Install if needed:
```bash
uv pip install python-dotenv
```

## Handoff Instructions

### Option 1: Run from External Machine (Recommended)

Since the Lambda endpoint is accessible from external machines but not from the current instance:

1. **Copy the evaluation files to your external machine**:
   ```bash
   # Files to copy:
   - .env
   - test_endpoints.py
   - eval_models.py
   ```

2. **Install dependencies**:
   ```bash
   pip install requests python-dotenv
   ```

3. **Test both endpoints**:
   ```bash
   python test_endpoints.py
   ```
   
   Expected: Both endpoints should pass.

4. **Run the evaluation**:
   ```bash
   python eval_models.py
   ```
   
   This will:
   - Run 10 test cases on both models
   - Save results to `eval_results_YYYYMMDD_HHMMSS.json`
   - Print summary statistics

### Option 2: Split Evaluation

If you need to run from different machines:

1. **On current instance (192.222.54.152)**: Run GLM tests only
2. **On external machine**: Run Lambda tests only
3. **Merge results manually** or modify script to combine JSON outputs

### Option 3: Fix Network Routing

Contact network admin to enable routing between:
- Source: 192.222.54.152 (current instance)
- Destination: 192.222.55.178 (Lambda API)
- Port: 443 (HTTPS)

## Expected Output

### Test Results File Structure
```json
{
  "timestamp": "20251008_063000",
  "datetime": "2025-10-08T06:30:00",
  "models": {
    "glm": {
      "name": "GLM 4.5",
      "api_base": "http://192.222.54.152:8000/v1"
    },
    "lambda": {
      "name": "Llama 4 Maverick",
      "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
      "api_base": "https://internal-inference-api.bugnest.net/v1"
    }
  },
  "test_cases": [
    {
      "test_number": 1,
      "category": "General Knowledge",
      "question": "What is the capital of France?",
      "system_prompt": "You are a helpful AI assistant.",
      "glm_response": {
        "success": true,
        "answer": "The capital of France is Paris.",
        "response_time": 1.23,
        "error": null,
        "raw_response": {...}
      },
      "lambda_response": {
        "success": true,
        "answer": "The capital of France is Paris.",
        "response_time": 0.87,
        "error": null,
        "raw_response": {...}
      }
    },
    ...
  ]
}
```

### Summary Statistics
- Success rate for each model
- Average response time
- Min/Max response times
- Per-category performance

## Known Issues

1. **Network Isolation**: Current instance cannot reach Lambda API
   - Workaround: Run from external machine

2. **Model Name Sensitivity**: Lambda API requires exact model name
   - Must use: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
   - NOT: `llama4-maverick`

3. **Prompt Format Differences**: 
   - GLM uses special formatting with `[gMASK]<sop>` and role tags
   - Lambda uses standard OpenAI chat format

## Next Steps

1. Run evaluation from a machine with access to both endpoints
2. Analyze results in the generated JSON file
3. Compare performance metrics:
   - Response quality (manual review)
   - Response time
   - Success rate
   - Category-specific performance

## Questions for Network Admin

If network routing fix is desired:
1. Can we enable routing from 192.222.54.152 to 192.222.55.178:443?
2. Is there a firewall rule blocking inter-subnet communication?
3. Should we use a different instance that has access to both endpoints?

---

**Status**: Ready for execution from external machine with network access to both endpoints.

