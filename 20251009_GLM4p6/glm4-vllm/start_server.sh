cd /home/ubuntu/wentao/20251009_GLM4p6/glm4-vllm && \
export PATH="$HOME/.local/bin:$PATH" && \
nohup uv run python main.py > glm4_server.log 2>&1 &
echo $! > glm4_server.pid