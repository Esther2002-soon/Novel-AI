# Novel-AI：有長期記憶的小說寫作 AI 助理

> 一個完全 local 部署的中文小說寫作 AI，用 RAG 架構讓 LLM 記住你的小說設定、角色、世界觀。不依賴任何外部 API，資料完全不外流。

---

## 為什麼做這個

寫小說最大的痛點是 context limit——每次開新 chat，AI 就忘記所有設定，角色個性、世界觀、前情要一直重餵。

這個系統解決這個問題：用 Qdrant vector DB 存記憶片段，每次對話自動語意搜尋相關設定，注入 system prompt，讓 LLM 在任何 chat room 都「記得」你的小說。

---

## 系統架構

```
手機/電腦（透過 Tailscale）
        ↓
Open WebUI（port 3000）     ← chat 介面
        ↓
Pipelines（port 9099）      ← mem0 filter pipeline
  ├── inlet：搜記憶 → inject system prompt
  └── outlet：非同步存對話記憶
        ↓
FastAPI mem0 service（port 8080）
  ├── REST API（/add /search /get /delete）
  └── /ui 記憶管理介面（手刻 HTML/JS）
        ↓
  ┌─────┴──────┐
  ▼            ▼
Qdrant       Ollama（port 11434）
(port 6333)  ├── qwen2.5:32b（寫作 LLM）
             └── nomic-embed-text（embedding）
```

---

## RAG 記憶系統詳解

這是整個系統最核心的部分，分成「存記憶」跟「取記憶」兩個流程。

### 存記憶流程

```
[使用者訊息]
     ↓
過濾：role=user、長度 ≥ 8 字
     ↓
nomic-embed-text 向量化
輸出 768 維 float 向量
     ↓
cosine similarity dedup check
（與現有記憶比對，score > 0.92 → UPDATE，否則 ADD）
     ↓
PointStruct 存入 Qdrant
payload: {user_id, memory, created_at}
```

### 取記憶流程（Top-5 Retrieval）

每次使用者發訊息，pipeline inlet 觸發以下流程：

**Step 1：Query Embedding**

把使用者的輸入文字丟進 `nomic-embed-text`，得到一個 768 維的 float 向量：

```python
resp = ol.embeddings(model="nomic-embed-text", prompt=user_message)
query_vector = resp["embedding"]  # shape: [768]
```

**Step 2：Qdrant Cosine Similarity Search**

把 query vector 送進 Qdrant，Qdrant 計算它跟 collection 裡所有記憶向量的 cosine similarity：

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- 結果介於 -1 到 1，越接近 1 代表語意越相近
- Qdrant 對所有記憶點做這個計算，依照 score 降序排列
- 加上 `user_id` filter，只搜這個 project 的記憶：

```python
hits = qdrant.query_points(
    collection_name="mem0_raw",
    query=query_vector,
    limit=5,                          # 取 top-5
    query_filter=models.Filter(
        must=[models.FieldCondition(
            key="user_id",
            match=models.MatchValue(value=user_id)
        )]
    )
)
```

**Step 3：為什麼是 Top-5？**

- 太少（top-1, top-2）：容易漏掉相關但不是最相似的設定
- 太多（top-10+）：inject 太多 token 進 system prompt，增加 LLM 的 context 負擔，可能反而干擾生成
- Top-5 是在「recall 夠用」跟「不塞爆 context」之間的平衡點

**Step 4：注入 System Prompt**

把 top-5 記憶組成 memory injection，塞進 system prompt：

```python
memory_text = "\n".join([f"- {m['memory']}" for m in memories[:5]])
memory_injection = f"""
【重要：以下是你必須遵守的已知設定，不可違背或自行編造】
{memory_text}
【設定結束】
"""
```

用強調語氣（「不可違背或自行編造」）是 Prompt Engineering 的設計——告訴 LLM 這些設定的優先級高於它自己的推斷。

**Step 5：Cosine Dedup（存記憶時）**

存新記憶之前先做 similarity check，避免語意重複的記憶一直堆積：

```python
hits = qdrant.query_points(query=new_vector, limit=1, ...)
if hits.points and hits.points[0].score > 0.92:
    # 太相似，UPDATE 現有記憶
    qdrant.upsert(id=existing_id, vector=new_vector, ...)
else:
    # 夠新，ADD 新的記憶點
    qdrant.upsert(id=new_uuid, vector=new_vector, ...)
```

threshold 0.92 的選擇：
- 0.92 以上幾乎是同一句話的改寫
- 0.85-0.92 是語意相近但細節不同（應該保留兩條）
- 0.92 以下全部新增

---

## 核心功能

### @tag 多專案隔離
在訊息裡打 `@projectname`，記憶自動切換到對應的 namespace。同一帳號可以管理多個寫作專案，記憶完全不混。

```
@fantasy_novel 主角叫做 Arthur，是一個失憶的騎士
@romance_story 女主角在咖啡廳第一次見到男主角
```

同一個專案開再多個 chat room 都共享同一份記憶。沒有 @tag 的對話完全不存記憶。

### 記憶管理介面
`/ui` 是手刻的 web 介面（dark theme，HTTP Basic Auth 保護）：
- 查看所有 project
- 查看 project 內所有記憶條目
- 手動新增記憶
- 刪除單條 / 整個 project

### 非阻塞存記憶
`outlet` 用 `asyncio.create_task()` fire-and-forget，存記憶在背景跑，不 block LLM response 回傳。

---

## Tech Stack

| 元件 | 技術 | 說明 |
|------|------|------|
| LLM | Ollama + qwen2.5:32b Q4 | 19GB，跑在 2 張 GPU |
| Embedding | nomic-embed-text | 274MB，向量維度 768 |
| Vector DB | Qdrant | cosine similarity search |
| Memory API | FastAPI + uvicorn | 自訂 REST API |
| Chat UI | Open WebUI | 支援 pipeline filter |
| Pipeline | Open WebUI filter | inlet/outlet 攔截 |
| 遠端連線 | Tailscale | zero-config VPN |
| Container | Docker + Compose | 服務管理 |

---

## 本地執行步驟

### 此 Repo 包含的檔案

```
├── mem0/app/main.py          # FastAPI 記憶 API + 管理 UI
├── mem0/app/requirements.txt
├── mem0/app/Dockerfile
├── mem0/docker-compose.yml
├── pipelines/mem0_pipeline.py  # Open WebUI filter pipeline
├── start.sh                    # tmux 監控腳本
├── vacuum.sh                   # SQLite VACUUM 腳本
├── README.md
└── WORKFLOW.md
```

以下內容**不在 repo 中**，需自行安裝或建立：

| 項目 | 說明 |
|------|------|
| Ollama + LLM models | 體積過大（~20GB），自行下載 |
| Open WebUI | 獨立 Docker container |
| Pipelines service | 獨立 Docker container |
| `qdrant-data/` | 啟動後自動建立，存你的記憶資料 |
| `webui-data/` | 啟動後自動建立，存對話記錄 |
| `ollama-data/` | 存 model 檔案，自行建立掛載 |

---

### 前置需求

- Ubuntu 20.04+
- Docker + Docker Compose
- NVIDIA GPU（建議 VRAM ≥ 20GB 以跑 qwen2.5:32b）

---

### Step 1：Clone 此 Repo

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Step 2：安裝 Ollama 並下載模型

> Model 檔案不在 repo 中，需自行下載，qwen2.5:32b 約 19GB 請預留空間與時間。

```bash
# 安裝 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下載模型
ollama pull qwen2.5:32b       # 主力寫作 LLM，約 19GB
ollama pull nomic-embed-text  # Embedding model，約 274MB

# 確認
ollama list
```

用 Docker 跑 Ollama（可指定 GPU）：

```bash
docker run -d \
  --name ollama \
  --restart unless-stopped \
  --gpus all \
  -v /your/path/ollama-data:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama
```

預載進 VRAM，避免第一次回應太慢：

```bash
docker exec ollama ollama run qwen2.5:32b --keepalive 24h
```

### Step 3：啟動 Qdrant + mem0 service

```bash
cd mem0
docker compose up -d
```

啟動後：
- Qdrant：`http://localhost:6333`
- mem0 API：`http://localhost:8080`
- 記憶管理 UI：`http://localhost:8080/ui`

UI 帳密透過環境變數設定（見下方「環境變數」）。

### Step 4：安裝 Open WebUI + Pipelines

> Open WebUI 和 Pipelines 不在此 repo，用官方 Docker image 安裝。

```bash
# Open WebUI
docker run -d --name openwebui --restart unless-stopped \
  -p 3000:8080 \
  -v /your/path/webui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

# Pipelines
docker run -d --name pipelines --restart unless-stopped \
  -p 9099:9099 \
  ghcr.io/open-webui/pipelines:main
```

### Step 5：載入 Pipeline

1. 開啟 Open WebUI（`http://localhost:3000`）
2. Admin Panel → Settings → Pipelines
3. 輸入 `http://localhost:9099` 連接
4. 上傳此 repo 的 `pipelines/mem0_pipeline.py`
5. Valves 設定 `MEM0_URL` 為 `http://<your-host-ip>:8080`

### Step 6：（可選）Tailscale 遠端連線

讓手機或其他電腦也能連到你的 server：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4  # 查詢你的 Tailscale IP
```

手機和電腦也裝 Tailscale app，登入同一帳號，之後用 `http://<tailscale-ip>:3000` 就能從任何地方連線。

---

## API 文件

`http://localhost:8080/docs` 有 Swagger 文件。

| Method | Endpoint | 說明 |
|--------|----------|------|
| POST | `/add` | 從 messages list 提取並存入記憶 |
| POST | `/add_single` | 直接存入單條記憶字串 |
| POST | `/search` | 語意搜尋相關記憶（top-5） |
| GET | `/get/{user_id}` | 取得某 project 全部記憶 |
| GET | `/projects` | 列出所有 project |
| DELETE | `/delete/{user_id}` | 刪除整個 project 的記憶 |
| DELETE | `/delete_one/{point_id}` | 刪除單條記憶 |
| GET | `/ui` | 記憶管理 web 介面（Basic Auth） |

---

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `QDRANT_HOST` | `<host-ip>` | Qdrant 主機位址 |
| `OLLAMA_URL` | `http://<host-ip>:11434` | Ollama 服務 URL |
| `UI_USER` | （自行設定） | 記憶管理介面帳號 |
| `UI_PASS` | （自行設定） | 記憶管理介面密碼 |

---

## 目錄結構

Repo 中的檔案：

```
.
├── mem0/
│   ├── app/
│   │   ├── main.py            # FastAPI 記憶 API + /ui 管理介面
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── docker-compose.yml
├── pipelines/
│   └── mem0_pipeline.py       # Open WebUI filter pipeline
├── start.sh                   # tmux 監控腳本
├── vacuum.sh                  # SQLite VACUUM 腳本
├── README.md
└── WORKFLOW.md
```

啟動後自動建立（不在 repo 中）：

```
├── mem0/qdrant-data/          # Qdrant 向量記憶資料
├── ollama-data/               # Ollama model 檔案（需自行掛載）
└── webui-data/                # Open WebUI 對話記錄與 cache
```

---

## 維護

```bash
# 查看 logs（tmux 四視窗）
bash start.sh

# SQLite 定期清理（crontab）
0 3 * * 0 bash /path/to/vacuum.sh

# 重啟服務
cd my-project/mem0 && docker compose restart
docker restart ollama openwebui pipelines
```

記憶備份：直接備份 `mem0/qdrant-data/`，刪除重啟 container 不影響記憶。
