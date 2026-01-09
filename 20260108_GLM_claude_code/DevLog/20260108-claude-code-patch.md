# Claude Code Model Name Case Preservation Patch

## Issue Summary

Claude Code automatically lowercases all custom model names before sending them to the API. When setting `ANTHROPIC_DEFAULT_SONNET_MODEL=zai-org/GLM-4.6-FP8`, the application converts it to `zai-org/glm-4.6-fp8`, causing API requests to fail when the endpoint requires exact case matching.

## Root Cause

The issue originates in the `vw()` function in Claude Code's bundled JavaScript code. This function processes model names with the following logic:

1. Converts input to lowercase for alias matching: `let Q=A.toLowerCase().trim()`
2. Checks if the lowercased name matches known aliases (sonnet, opus, haiku, opusplan)
3. For known aliases, returns the appropriate model from environment variables or defaults
4. **For unknown/custom models, returns the lowercased version `Q` instead of the original input**

## Patch Location

**File:** `/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js`

**Line:** 891

**Function:** `vw(A)`

## Patch Details

### Original Code
```javascript
function vw(A){
  let Q=A.toLowerCase().trim(),
  B=Q.endsWith("[1m]"),
  G=B?Q.replace(/\[1m]$/i,"").trim():Q;
  if(cg1(G))
    switch(G){
      case"opusplan":return AM()+(B?"[1m]":"");
      case"sonnet":return AM()+(B?"[1m]":"");
      case"haiku":return F7A()+(B?"[1m]":"");
      case"opus":return TUA();
      default:
    }
  return Q  // Returns lowercased version
}
```

### Patched Code
```javascript
function vw(A){
  let Q=A.toLowerCase().trim(),
  B=Q.endsWith("[1m]"),
  G=B?Q.replace(/\[1m]$/i,"").trim():Q;
  if(cg1(G))
    switch(G){
      case"opusplan":return AM()+(B?"[1m]":"");
      case"sonnet":return AM()+(B?"[1m]":"");
      case"haiku":return F7A()+(B?"[1m]":"");
      case"opus":return TUA();
      default:
    }
  return A.trim()  // Returns original case, trimmed
}
```

### Change Summary
Changed the return statement from `return Q` to `return A.trim()` to preserve the original case of custom model names while still removing whitespace.

## Implementation

```bash
# Backup original file
cp /opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js \
   /opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js.backup

# Apply patch
sed -i '' 's/case"opus":return TUA();default:}return Q}/case"opus":return TUA();default:}return A.trim()}/' \
   /opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js
```

## Impact

After applying this patch, custom model names specified in `ANTHROPIC_DEFAULT_SONNET_MODEL` will be sent to the API with their original case preserved, allowing integration with case-sensitive model endpoints.

## Limitations

This patch will be overwritten when Claude Code is updated. The patch must be reapplied after each update to the `@anthropic-ai/claude-code` package.

## How Claude Code Calls the Inference API

### API Client Creation

Claude Code uses a factory function `rw()` to create the appropriate API client based on environment variables and configuration.

**Location:** `/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js` (search for `async function rw(`)

### Key Environment Variables

1. **`ANTHROPIC_BASE_URL`**: Custom API endpoint URL
   - When set, overrides the default Anthropic API endpoint
   - Example: `"https://internal-inference.bugnest.net/"`
   - Applied to the client configuration as `baseUrl` option

2. **`ANTHROPIC_API_KEY`**: API authentication key
   - Used for authentication with the API endpoint
   - Can be overridden by user settings

3. **`ANTHROPIC_MODEL`**: Default model override
   - Overrides the default model selection

4. **`ANTHROPIC_SMALL_FAST_MODEL`**: Small/fast model override

### Client Configuration Flow

```javascript
// Simplified flow from the code
async function rw({apiKey, maxRetries, model, fetchOverride}) {
    let config = {
        defaultHeaders: {
            'x-app': 'cli',
            'User-Agent': xc(),
            // ... other headers
        },
        maxRetries: maxRetries,
        timeout: parseInt(process.env.API_TIMEOUT_MS || '600000', 10),
        dangerouslyAllowBrowser: true,
        // ... other options
    };
    
    // Apply ANTHROPIC_BASE_URL if set
    if (process.env.ANTHROPIC_BASE_URL) {
        config.baseUrl = process.env.ANTHROPIC_BASE_URL;
    }
    
    // Create appropriate client (Bedrock, Vertex, Foundry, or standard Anthropic)
    if (process.env.CLAUDE_CODE_USE_BEDROCK) {
        return new BedrockClient(config);
    } else if (process.env.CLAUDE_CODE_USE_VERTEX) {
        return new VertexClient(config);
    } else if (process.env.CLAUDE_CODE_USE_FOUNDRY) {
        return new FoundryClient(config);
    } else {
        // Standard Anthropic client
        config.apiKey = apiKey || getApiKey();
        return new AnthropicClient(config);
    }
}
```

### API Request Execution

The actual API calls are made through the client's `beta.messages.stream()` or `beta.messages.create()` methods:

**Streaming requests:**
```javascript
let stream = client.beta.messages.stream({
    model: modelName,
    messages: normalizedMessages,
    system: systemPrompt,
    tools: toolSchemas,
    max_tokens: maxTokens,
    betas: betaFeatures,
    metadata: { user_id: userId },
    // ... other parameters
}, { signal: abortSignal });
```

**Non-streaming requests:**
```javascript
let response = await client.beta.messages.create({
    model: modelName,
    messages: normalizedMessages,
    system: systemPrompt,
    tools: toolSchemas,
    max_tokens: maxTokens,
    betas: betaFeatures,
    metadata: { user_id: userId },
    // ... other parameters
});
```

### Request Headers

Default headers include:
- `x-app: cli` - Identifies the application as CLI
- `User-Agent: <version info>` - Claude Code version information
- Custom headers from `ANTHROPIC_CUSTOM_HEADERS` environment variable (newline-separated `key: value` pairs)

### Usage Tracking

Claude Code tracks API usage through:
1. **Analytics events**: `tengu_api_query`, `tengu_api_success`, `tengu_api_error`
2. **Token counting**: Input tokens, output tokens, cache read/creation tokens
3. **Cost calculation**: Based on model pricing and token usage
4. **Rate limit monitoring**: Via response headers (`anthropic-ratelimit-*`)

### Custom API Endpoint Requirements

For a custom endpoint to work with Claude Code, it must:
1. Accept the same request format as Anthropic's Messages API
2. Return responses in the same format (including streaming events)
3. Support the `/v1/messages` endpoint
4. Handle the `model` parameter with exact case matching (after applying the patch)
5. Optionally support rate limit headers for usage tracking

### Configuration via Settings File

The `.claude/settings.json` file can override environment variables:

```json
{
    "ANTHROPIC_BASE_URL": "https://internal-inference.bugnest.net/",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "zai-org/GLM-4.6-FP8"
}
```

These settings are loaded and merged with environment variables, with settings file taking precedence.
