#!/bin/bash
docker exec openwebui-novel sqlite3 /app/backend/data/webui.db "VACUUM;"
echo "$(date): VACUUM done"
