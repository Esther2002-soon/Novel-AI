cat > /mnt/aicv/esther/novel-ai/start.sh << 'EOF'
#!/bin/bash
# 如果 session 已存在就 attach，不存在就新建
if tmux has-session -t novel-ai 2>/dev/null; then
    tmux attach -t novel-ai
else
    tmux new-session -d -s novel-ai -n logs
    tmux send-keys -t novel-ai:logs "docker logs -f ollama-novel" Enter
    tmux new-window -t novel-ai -n webui
    tmux send-keys -t novel-ai:webui "docker logs -f openwebui-novel" Enter
    tmux new-window -t novel-ai -n mem0
    tmux send-keys -t novel-ai:mem0 "docker logs -f mem0-novel" Enter
    tmux new-window -t novel-ai -n shell
    tmux attach -t novel-ai
fi
EOF
chmod +x /mnt/aicv/esther/novel-ai/start.sh