# Open WebUI Setup for Kimi-K2-Thinking

## Basic Testing Results

### 1. Model Availability Test
```bash
curl -X GET $LLM_URL/v1/models \
  -H "Authorization: Bearer $LLM_TOKEN"
```
**Result:** Model `kimi-k2-thinking` is available

### 2. Inference Test
```bash
curl -X POST $LLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_TOKEN" \
  -d '{
    "model": "kimi-k2-thinking",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```
**Result:** Model responds with reasoning process visible in `reasoning_content` field

---

## Open WebUI Deployment

### Quick Start
```bash
./start_openwebui.sh
```

### Container Status
```bash
docker ps | grep open-webui
```

### Access the UI
**URL:** http://localhost:8080

### Configuration Files
- **Environment:** `.env.openwebui` (contains API URL and key)
- **Launch Script:** `start_openwebui.sh` (automated deployment)

### Configuration
- **API Base URL:** Stored in `.env.openwebui`
- **API Key:** Stored in `.env.openwebui`
- **Model Name:** kimi-k2-thinking
- **Port:** 8080 (mapped to container's 8080)
- **Data Volume:** open-webui (persistent storage)

---

## First-Time Setup Steps

1. **Open your browser** and navigate to: http://localhost:8080

2. **Create an admin account** (first user becomes admin)
   - Enter your email and password
   - This is a local account, not connected to any external service

3. **Configure the model connection:**
   - Go to **Settings** (gear icon) → **Connections**
   - The OpenAI API connection should already be configured via environment variables
   - Verify the API Base URL matches your `.env.openwebui` configuration

4. **Select the model:**
   - In the chat interface, click the model selector dropdown
   - Choose `kimi-k2-thinking`

5. **Start chatting!**
   - The model will show its reasoning process
   - Conversations are saved automatically

---

## 🔧 Management Commands

### View Logs
```bash
docker logs open-webui
docker logs -f open-webui  # Follow logs in real-time
```

### Restart Container
```bash
docker restart open-webui
```

### Stop Container
```bash
docker stop open-webui
```

### Start Container (if stopped)
```bash
docker start open-webui
```

### Remove Container (keeps data volume)
```bash
docker rm -f open-webui
```

### Recreate Container (if needed)
```bash
./start_openwebui.sh
```

Or manually:
```bash
docker run -d \
  --name open-webui \
  -p 8080:8080 \
  --env-file .env.openwebui \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

---

## Features

- **ChatGPT-like Interface:** Modern, intuitive UI
- **Conversation History:** All chats are saved
- **Model Switching:** Easy to switch between models (if you add more)
- **Markdown Support:** Rich text formatting in responses
- **Code Highlighting:** Syntax highlighting for code blocks
- **File Uploads:** Can upload documents (if configured)
- **User Management:** Multi-user support with permissions
- **Dark/Light Mode:** Theme switching

---

## Troubleshooting

### Container won't start
```bash
docker logs open-webui
```

### Can't access UI
- Check if container is running: `docker ps | grep open-webui`
- Check if port 8080 is available: `lsof -i :8080`
- Try accessing: http://127.0.0.1:8080

### Model not responding
- Verify the inference service is running: `curl http://192.222.55.145:8000/v1/models -H "Authorization: Bearer $LLM_TOKEN"`
- Check API key is correct in `.env.openwebui`
- Check Open WebUI logs for connection errors

### Reset everything
```bash
docker rm -f open-webui
docker volume rm open-webui
# Then recreate the container
```

---

## Model Information

- **Model ID:** kimi-k2-thinking
- **Max Context:** 12,288 tokens
- **Special Feature:** Reasoning model with visible thinking process
- **Performance:** ~10.5 tokens/sec generation speed
- **Concurrent Requests:** 1-4 supported

---

## Security Notes

- The API key is stored in the container environment
- Open WebUI data is persisted in a Docker volume
- For production use, consider:
  - Setting up HTTPS/TLS
  - Restricting CORS settings
  - Using a reverse proxy (nginx/traefik)
  - Implementing proper authentication

