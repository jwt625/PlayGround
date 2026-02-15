# DevLog 000: OpenClaw Signal Bot Setup

Date: 2026-02-14/15
Project: OpenClaw Personal AI Assistant
Channel: Signal

## Overview

This document captures the setup process, bugs encountered, and solutions for getting OpenClaw working with Signal as a messaging channel using a custom inference endpoint.

---

## Environment

- OS: macOS 15.5 (arm64)
- Node.js: v23.7.0 (requirement: >=22)
- Package Manager: pnpm
- signal-cli: 0.13.24 (via Homebrew)
- OpenClaw: 2026.2.15

---

## Initial Setup

1. Clone repository and install dependencies:
   ```bash
   git clone https://github.com/openclaw/openclaw.git
   pnpm install
   pnpm ui:build
   pnpm build
   ```

2. Run onboarding wizard:
   ```bash
   pnpm openclaw onboard
   ```

---

## Issue 1: signal-cli QR Code Linking Fails

### Symptom
Running `signal-cli link -n "OpenClaw"` outputs only a text link (`sgnl://linkdevice?uuid=...&pub_key=...`), not a visual QR code. When converted to QR code manually, Signal app returns "QR code invalid" error.

### Attempted Solutions
- Installed qrencode: `brew install qrencode`
- Converted link to QR: `signal-cli link -n "OpenClaw" | head -n 1 | qrencode -t ansiutf8`
- Multiple attempts with fresh links

### Root Cause
- Signal linking sessions expire extremely fast (30-60 seconds)
- signal-cli QR linking is known to be unreliable
- Race condition between generating QR and scanning

### Solution
Abandoned QR linking. Used SMS registration path instead:
```bash
signal-cli -a +1XXXXXXXXXX register
signal-cli -a +1XXXXXXXXXX verify <SMS_CODE>
```

### Lesson Learned
For signal-cli integration, prefer dedicated number registration (Path B) over device linking (Path A). Registration is more reliable and gives a clean, dedicated bot identity.

---

## Issue 2: Pairing System Requires Manual Approval

### Symptom
After sending a message to the bot, no response received.

### Root Cause
OpenClaw config had `dmPolicy: "pairing"` which requires explicit approval before the bot responds to DMs.

### Solution
1. Check pending pairings:
   ```bash
   pnpm openclaw pairing list signal
   ```
2. Approve the pairing code:
   ```bash
   pnpm openclaw pairing approve signal <CODE>
   ```

### Configuration Reference
```json
"channels": {
  "signal": {
    "dmPolicy": "pairing",
    "groupPolicy": "allowlist"
  }
}
```

### Lesson Learned
The pairing system is a security feature. First-time contacts must be explicitly approved. Pairing codes expire after 1 hour.

---

## Issue 3: Model Inference Stuck at "conjuring..."

### Symptom
After pairing approval, bot receives messages but hangs at "conjuring..." indefinitely (19+ seconds with no response).

### Diagnosis
Checked logs:
```bash
tail -50 ~/.openclaw/logs/gateway.err.log
```

Found error:
```
FailoverError: Model context window too small (4096 tokens). Minimum is 16000.
blocked model (context window too small): <custom-model-id> ctx=4096 (min=16000)
```

### Root Cause
OpenClaw requires a minimum context window of 16,000 tokens. The model was configured with only 4,096 tokens in the config file.

### Solution
Updated `~/.openclaw/openclaw.json`:
```bash
# Backup first
cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.backup

# Update context window values
sed -i '' 's/"contextWindow": 4096/"contextWindow": 131072/' ~/.openclaw/openclaw.json
sed -i '' 's/"maxTokens": 4096/"maxTokens": 131072/' ~/.openclaw/openclaw.json
```

Gateway auto-reloads on config change, or restart manually.

### Lesson Learned
When configuring custom model providers, set contextWindow to at least 16000 (OpenClaw minimum). For models like GLM-4, use actual model capacity (e.g., 131072 for 128k context).

---

## Issue 4: Signal Profile Warning

### Symptom
Logs show warning:
```
WARN ManagerImpl - No profile name set. When sending a message it's recommended to set a profile name with the updateProfile command.
```

### Solution (Optional)
```bash
signal-cli -a +1XXXXXXXXXX updateProfile --name "OpenClaw Bot"
```

---

## Useful Commands Reference

| Command | Purpose |
|---------|---------|
| `pnpm openclaw status` | Overall system status |
| `pnpm openclaw status --deep` | Detailed status with probes |
| `pnpm openclaw channels status --probe` | Check channel connectivity |
| `pnpm openclaw pairing list signal` | List pending pairing requests |
| `pnpm openclaw pairing approve signal <CODE>` | Approve a pairing |
| `pnpm openclaw logs --follow` | Tail logs in real-time |
| `pnpm openclaw tui` | Interactive terminal UI |
| `tail ~/.openclaw/logs/gateway.err.log` | View error logs directly |
| `tail ~/.openclaw/logs/gateway.log` | View general logs directly |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `~/.openclaw/openclaw.json` | Main configuration |
| `~/.openclaw/agents/` | Agent configurations |
| `~/.openclaw/logs/` | Log files |
| `~/.openclaw/workspace/` | Agent workspace |

---

## Custom Model Provider Configuration

Example configuration for custom OpenAI-compatible endpoint:

```json
"models": {
  "mode": "merge",
  "providers": {
    "custom-provider-name": {
      "baseUrl": "https://your-endpoint.example.com/v1",
      "apiKey": "your-api-key",
      "api": "openai-completions",
      "models": [
        {
          "id": "model-id",
          "name": "Model Display Name",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 131072
        }
      ]
    }
  }
}
```

Key fields:
- `api`: Use "openai-completions" for OpenAI-compatible APIs
- `contextWindow`: Must be >= 16000 for OpenClaw
- `maxTokens`: Maximum output tokens

---

## Summary of Critical Learnings

1. Use SMS registration for signal-cli, not QR linking
2. Pairing system requires manual approval for first contact
3. Context window must be >= 16000 tokens
4. Check `~/.openclaw/logs/gateway.err.log` for inference errors
5. Gateway auto-reloads on config changes
6. The `pnpm openclaw logs` command does not support `--tail` option; use `--limit` instead

