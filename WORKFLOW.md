# Agent Collaboration Workflow Log
# Novel-AI：有長期記憶的小說寫作 AI 助理

> 本文件記錄與 Claude (claude.ai) 協作開發的完整過程，包含四個階段的關鍵 prompt、工具使用紀錄、技術瓶頸與解法。

---

## Phase 1：發想與企劃

### 背景

我平常有在寫中文網文，但每次開新 chat 跟 AI 寫，它就忘記所有設定了——角色個性、世界觀、之前的伏筆，要一直重新貼 context 很煩。我想做一個真的能記住設定的寫作助理。

---

### 對話紀錄

**[我]**
> 我想做一個 AI project，但我還不確定要做什麼。我有台 Linux server，GPU 很多。你覺得做什麼有意思？

**[Claude]**
> 有幾個方向可以考慮：
> 1. 圖像生成工具（Stable Diffusion + ControlNet）
> 2. 語音轉文字 + 摘要 pipeline
> 3. 有長期記憶的對話 AI
>
> 你平常用 AI 最大的痛點是什麼？從痛點出發比較容易做出真的有用的東西。

**[我]**
> 我有在寫小說，最大問題是 context limit，換個 chat 它就什麼都忘了。要一直重新跟它解釋設定，很煩

**[Claude]**
> 這個問題很值得解！而且技術上很有意思——核心是要做 **跨對話的長期記憶**。
>
> 有幾個解法：
> 1. 手動貼設定（治標不治本）
> 2. Fine-tune 一個 model 讓它「記住」你的世界觀（成本很高，迭代慢）
> 3. RAG 架構——把設定存進 vector DB，每次對話自動 retrieve 相關記憶注入 system prompt
>
> 方向 3 最靈活，設定可以隨時新增修改，不用重新訓練。你想走這個方向嗎？

**[我]**
> RAG 是什麼？我有聽過但不太清楚

**[Claude]**
> RAG 是 Retrieval-Augmented Generation，概念是這樣：
>
> 與其讓 LLM 把所有知識都記在 weights 裡（fine-tune），不如在推論時動態去外部資料庫撈相關資料，塞進 prompt 讓它用。
>
> 用在你的小說場景：
> - 你的所有角色設定、世界觀、劇情筆記 → 存進 vector DB
> - 每次對話前，把你的問題向量化，去 DB 找最相關的幾條設定
> - 把找到的設定塞進 system prompt
> - LLM 看到這些設定就能「記得」
>
> 完全不需要 fine-tune，而且設定隨時可以更新。

**[我]**
> 好！這個很適合我。我想完全 local 跑，資料不想外流

**[Claude]**
> 完全可以做到，推薦這個 stack：
> - **Ollama** — local LLM inference，有 OpenAI-compatible API
> - **Qdrant** — vector DB，self-hosted
> - **nomic-embed-text** — embedding model，274MB，很輕
> - **Open WebUI** — chat 介面，支援自訂 pipeline
> - **Tailscale** — VPN，讓手機電腦都能連到你的 server

**[我]**
> Tailscale 是什麼？

**[Claude]**
> Zero-config VPN。裝好之後你的手機、電腦、server 都在同一個私有網路，不用設定 port forwarding。從咖啡廳也能連到家裡的 server，免費，個人用很夠。

**[我]**
> 那 LLM 要用哪個？

**[Claude]**
> 你目標是寫中文網文，推薦 **Qwen2.5-32B Q4 量化**：
> - 中文能力強，訓練語料包含大量中文文本
> - Q4 量化後約 19GB，多張 GPU 可以跑
> - 不需要 fine-tune，好的 system prompt + RAG 記憶注入就夠了

**[我]**
> 決定了！做「有長期記憶的中文小說寫作 AI 助理」

### 確定技術棧

| 項目 | 決定 |
|------|------|
| 核心問題 | LLM 沒有跨對話長期記憶 |
| 解法 | RAG 架構，Qdrant vector DB 語意記憶檢索 |
| LLM | qwen2.5:32b（local, via Ollama） |
| Embedding | nomic-embed-text（local, via Ollama） |
| Vector DB | Qdrant |
| Chat UI | Open WebUI + 自訂 pipeline |
| 記憶管理 | 自製 FastAPI service + 手刻 HTML/JS UI |
| 遠端連線 | Tailscale |

---

## Phase 2：架構設計與任務拆解

### 對話紀錄

**[我]**
> ok 決定做了。幫我把整個架構設計出來，然後 task 拆細一點，我要知道每個東西怎麼接在一起

**[Claude]**
> 整個系統架構：
>
> ```
> 手機/電腦（Tailscale）
>         ↓
> Open WebUI（port 3000）
>         ↓ pipeline filter
> mem0_pipeline.py（port 9099）
>   ├── inlet: 搜記憶 → inject system prompt
>   └── outlet: 非同步存對話
>         ↓ HTTP
> FastAPI mem0 service（port 8080）
>   ├── /add /search /get /delete
>   └── /ui 記憶管理介面
>         ↓
>   ┌─────┴──────┐
>   ▼            ▼
> Qdrant       Ollama（port 11434）
> (port 6333)  ├── qwen2.5:32b
>              └── nomic-embed-text
> ```

**[我]**
> top-5 搜尋是怎麼運作的？從 vector 到結果這個過程講清楚一點

**[Claude]**
> 好，完整流程：
>
> **Step 1：Query Embedding**
>
> 使用者的輸入文字丟進 `nomic-embed-text`，輸出一個 768 維的 float 向量。這個向量捕捉了文字的「語意位置」，語意相近的句子在這個 768 維空間裡距離很近。
>
> ```python
> resp = ol.embeddings(model="nomic-embed-text", prompt=user_message)
> query_vector = resp["embedding"]  # list of 768 floats
> ```
>
> **Step 2：Cosine Similarity**
>
> Qdrant 計算 query vector 跟 collection 裡每一條記憶的 cosine similarity：
>
> ```
> cosine_similarity(A, B) = (A · B) / (|A| × |B|)
> ```
>
> 點積除以兩個向量的長度積，結果介於 -1 到 1。1 表示完全相同語意，0 表示無關，-1 表示語意相反。
>
> 為什麼用 cosine 而不是 euclidean distance？因為 cosine 看的是方向，不是絕對距離，對 embedding vector 來說更準確——同一個意思用不同長度的句子說，euclidean distance 可能差很多，但 cosine similarity 會很高。
>
> **Step 3：Filter by user_id，取 Top-5**
>
> ```python
> hits = qdrant.query_points(
>     collection_name="mem0_raw",
>     query=query_vector,
>     limit=5,
>     query_filter=models.Filter(
>         must=[models.FieldCondition(
>             key="user_id",
>             match=models.MatchValue(value=user_id)
>         )]
>     )
> )
> ```
>
> `user_id` 就是 `{account_id}_{tag}`，確保只搜這個小說 project 的記憶，不會跨 project 污染。
>
> **Step 4：為什麼是 Top-5？**
>
> - Top-1 太少：只拿最相似那條，可能漏掉相關但不是最近的設定
> - Top-10+ 太多：塞太多 token 進 system prompt，增加 LLM context 負擔
> - Top-5 是「recall 夠用」跟「不塞爆 context」的平衡點
>
> **Step 5：Inject System Prompt**
>
> ```python
> memory_text = "\n".join([f"- {m['memory']}" for m in memories[:5]])
> memory_injection = """
> 【重要：以下是你必須遵守的已知設定，不可違背或自行編造】
> {memory_text}
> 【設定結束】
> """
> ```
>
> 強調語氣「不可違背或自行編造」是 Prompt Engineering 設計——告訴 LLM 這些設定優先級高於它自己的推斷。

**[我]**
> 那存記憶的時候 cosine dedup 是怎麼弄的？

**[Claude]**
> 存記憶之前先做一次 similarity check：
>
> 1. 把新記憶向量化
> 2. 在現有記憶裡找最相似的那一條
> 3. 如果 score > 0.92 → UPDATE 那條，不新增
> 4. 如果 score ≤ 0.92 → ADD 新的
>
> threshold 0.92 的選擇：
> - 0.92 以上幾乎是同一句話的改寫（例：「Arthur 是騎士」vs「Arthur 是一位騎士」）
> - 0.85-0.92 是語意相近但細節不同，應該保留兩條
> - 所以 0.92 是「真的重複才 update」的門檻

**[我]**
> 好，task 怎麼拆？

**[Claude]**
> Task list：
>
> | # | Task | 技術 |
> |---|------|------|
> | 1 | 安裝 Ollama，pull qwen2.5:32b 和 nomic-embed-text | Ollama CLI |
> | 2 | 啟動 Open WebUI + Pipelines container | Docker |
> | 3 | 實作 FastAPI mem0 service（CRUD API） | FastAPI + Qdrant Python SDK |
> | 4 | 實作 cosine dedup 邏輯 | Qdrant query_points |
> | 5 | 實作 /ui 記憶管理介面 | Plain HTML/CSS/JS |
> | 6 | 實作 Open WebUI filter pipeline | Python, httpx async |
> | 7 | 實作 @tag 多專案記憶隔離機制 | regex + dict cache |
> | 8 | Docker Compose 整合服務 | Docker Compose |
> | 9 | 設定 Tailscale 遠端連線 | Tailscale |
> | 10 | 維運腳本（tmux, vacuum） | bash |

**API 格式：**

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | /add | `{messages: [...], user_id: str}` | `{status, result}` |
| POST | /add_single | `{memory: str, user_id: str}` | `{status, id}` |
| POST | /search | `{query: str, user_id: str}` | `{memories: {results: [...]}}` |
| GET | /get/{user_id} | — | `{memories: {results: [...]}}` |
| GET | /projects | — | `{projects: [...]}` |
| DELETE | /delete/{user_id} | — | `{status}` |
| DELETE | /delete_one/{id} | — | `{status}` |

---

## Phase 3：程式碼生成與實作

### 3-1 環境建置

**[我]**
> 幫我寫 docker-compose.yml，要跑 qdrant 和 mem0 service

**[Claude]**
> ```yaml
> services:
>   qdrant:
>     image: qdrant/qdrant:latest
>     container_name: qdrant
>     restart: unless-stopped
>     ports:
>       - "6333:6333"
>     volumes:
>       - ./qdrant-data:/qdrant/storage
>
>   mem0:
>     build: ./app
>     container_name: mem0
>     restart: unless-stopped
>     ports:
>       - "8080:8080"
>     environment:
>       - QDRANT_HOST=qdrant
>       - OLLAMA_URL=http://<host-ip>:11434
>     depends_on:
>       - qdrant
> ```
>
> `<host-ip>` 是 docker bridge network 的 host IP，container 裡用這個打到 host 上的 Ollama。

### 3-2 FastAPI mem0 Service

**[我]**
> 幫我寫 main.py：
> - Qdrant / Ollama URL 從環境變數讀
> - /add 只存 user 訊息，< 8 字 skip
> - cosine dedup threshold 0.92
> - /add_single 直接存單條
> - /search top-5 filter by user_id
> - /get /delete /delete_one
> - /ui dark theme HTML，手刻，HTTP Basic Auth

**[Claude]**
> 生成完整 `main.py`。核心設計重點：
>
> embed function：
> ```python
> def embed(text: str) -> list:
>     resp = ol.embeddings(model="nomic-embed-text", prompt=text)
>     return resp["embedding"]
> ```
>
> cosine dedup：
> ```python
> hits = qdrant.query_points(
>     collection_name=COLLECTION,
>     query=vector, limit=1,
>     query_filter=...user_id filter...
> )
> if hits.points and hits.points[0].score > 0.92:
>     qdrant.upsert(id=existing_id, ...)  # UPDATE
> else:
>     qdrant.upsert(id=new_uuid, ...)     # ADD
> ```

### 3-3 Open WebUI Pipeline

**[我]**
> 幫我寫 filter pipeline：
> - inlet 搜記憶，inject system prompt，要強調設定不能違背
> - outlet 存對話記憶
> - user_id 用 {account_id}_{tag}

**[Claude]**
> 生成 `mem0_pipeline.py`。inlet 注入格式用強調語氣：
>
> ```python
> memory_injection = "\n\n【重要：以下是你必須遵守的已知設定，不可違背或自行編造】\n" + memory_text + "\n【設定結束】\n"
> ```
>
> 這是 Prompt Engineering 設計——明確告訴 LLM 記憶設定的優先級高於它自己的推斷，避免它在沒有記憶支持的情況下自己編造設定。

### 3-4 技術瓶頸一：outlet 同步阻塞

**[我]**
> 有個問題，AI 回覆完之後要等很久才顯示，感覺卡住了

**[Claude]**
> 問題在 outlet。雖然是 `async def`，但 `await client.post(...)` 會等整個 `/add` 跑完才 return，包含：
> 1. nomic-embed-text embedding（不快）
> 2. Qdrant cosine similarity check
> 3. Qdrant upsert
>
> 這整個過程可能好幾秒，block 住 response。
>
> 解法：`asyncio.create_task()` fire-and-forget：
>
> ```python
> # 改前（block response）
> async def outlet(self, body, user=None):
>     await self._save_memory(last_two, user_id)
>     return body
>
> # 改後（non-blocking）
> async def outlet(self, body, user=None):
>     asyncio.create_task(self._save_memory(last_two, user_id))
>     return body  # 立刻 return，記憶在背景存
> ```

**[我]**
> 好多了！response 正常速度了

### 3-5 技術瓶頸二：@tag 多專案記憶隔離

**[我]**
> 我發現一個問題，我在寫不同故事的時候會開很多個 chat room，但記憶是照 chat_id 分的，同一個故事的記憶就散掉了。有辦法讓同一個故事的不同 chat 共享記憶嗎

**[Claude]**
> 這是設計問題。解法是 `@tag` 機制：
>
> 在訊息裡打 `@projectname`，pipeline 自動 parse tag，把所有帶同樣 tag 的 chat 路由到同一個記憶空間。
>
> 實作：
> ```python
> def _extract_tag(self, messages):
>     for msg in messages:
>         if msg.get("role") in ["user", "system"]:
>             match = re.search(r"@([\w\u4e00-\u9fff]+)", msg.get("content", ""))
>             if match:
>                 return match.group(1)
>     return None
> ```
>
> pipeline instance 上維護 `_chat_project_map`，`inlet` 時存 `chat_id → project_id`，`outlet` 時查這個 map。
>
> 沒有 @tag 的對話完全不存記憶，避免垃圾資料污染 vector DB。

**[我]**
> 這個很聰明！裝好了可以用

### 3-6 技術瓶頸三：Qdrant API 版本

**[我]**
> `AttributeError: 'QdrantClient' object has no attribute 'search'` 這個怎麼回事

**[Claude]**
> 新版 qdrant-client 把 `.search()` 改成 `.query_points()` 了。把所有 search 呼叫換掉：
>
> ```python
> # 舊版
> qdrant.search(collection_name=..., query_vector=vector, limit=5)
>
> # 新版
> qdrant.query_points(collection_name=..., query=vector, limit=5)
> ```

### 3-7 Ollama GPU 設定

**[我]**
> 我要跑在指定的 GPU 上，怎麼指定

**[Claude]**
> docker run 時加 `--gpus all`：
>
> ```bash
> docker run -d \
>   --name ollama \
>   --restart unless-stopped \
>   --gpus all \
>   -v /path/to/ollama-data:/root/.ollama \
>   -p 11434:11434 \
>   ollama/ollama
> ```
>
> 然後預載進 VRAM：
> ```bash
> docker exec ollama ollama run qwen2.5:32b --keepalive 24h
> ```
>
> `--keepalive 24h` 讓 model 常駐在 VRAM，不 idle unload，inference latency 最小化。

### 工具使用紀錄（Phase 3）

| 工具 | 用途 |
|------|------|
| Claude (claude.ai) | 程式碼生成、debug、架構設計 |
| Ollama CLI | pull qwen2.5:32b、nomic-embed-text |
| Docker CLI | run、exec、compose up |
| curl | 手動測試 API endpoints |
| tmux | 多視窗監控 docker logs |
| nvidia-smi | 確認 GPU 使用狀況 |

---

## Phase 4：介面封裝與總結

### 4-1 記憶管理 Web UI

**[我]**
> 幫我寫 /ui 頁面：
> - project badge list，點了 load 記憶
> - 手動輸入 project id
> - 新增記憶 textarea
> - 記憶列表，每條有刪除按鈕
> - 整個 project 刪除
> - dark theme，#0f0f0f 背景，不要 framework

**[Claude]**
> 生成完整 HTML/CSS/JS，embedded 在 FastAPI `HTMLResponse` 裡。設計重點：
> - Project 用 badge 顯示，active 變藍色 border
> - 記憶列表顯示內容 + UUID + 刪除按鈕
> - 新增記憶有即時 status feedback
> - 全用 `fetch()` 打 API，無 page reload
> - HTTP Basic Auth 保護整個 `/ui`

**[我]**
> 做好了，這個介面很方便，我可以直接手動新增角色設定

### 4-2 維運腳本

**[我]**
> 幫我寫 tmux script，開四個 window 看 docker logs，已有 session 就 attach 不要重開

**[Claude]**
> `start.sh`：
> ```bash
> #!/bin/bash
> if tmux has-session -t my-project 2>/dev/null; then
>     tmux attach -t my-project
> else
>     tmux new-session -d -s my-project -n logs
>     tmux send-keys -t my-project:logs "docker logs -f ollama" Enter
>     tmux new-window -t my-project -n webui
>     tmux send-keys -t my-project:webui "docker logs -f openwebui" Enter
>     tmux new-window -t my-project -n mem0
>     tmux send-keys -t my-project:mem0 "docker logs -f mem0" Enter
>     tmux new-window -t my-project -n shell
>     tmux attach -t my-project
> fi
> ```

**[我]**
> SQLite 久了會不會變很大

**[Claude]**
> 會，Open WebUI 對話存 SQLite，長期會 fragmentation。加 cron job 定期 VACUUM：
>
> `vacuum.sh`：
> ```bash
> #!/bin/bash
> docker exec openwebui sqlite3 /app/backend/data/webui.db "VACUUM;"
> echo "$(date): VACUUM done"
> ```
> 加到 crontab 每週日凌晨 3 點跑。

### 4-3 技術成果總結

**[我]**
> 全部跑通了！幫我總結一下我這個 project 用了哪些技術，對應到課程要求

**[Claude]**
> 你的系統完整覆蓋 LLM 技術的核心應用：
>
> **Prompt Engineering**
> - 記憶注入 system prompt 格式（`【重要：以下是你必須遵守的已知設定，不可違背或自行編造】`）
> - 寫作 system prompt 設計：指定風格、格式、輸出規則
>
> **RAG 架構**
> - `nomic-embed-text` dense embedding（768 維向量）
> - Qdrant cosine similarity search，semantic retrieval top-5
> - 動態注入 retrieved memories 進 system prompt
> - Cosine dedup：score > 0.92 → UPDATE，避免語意重複堆積
>
> **本機開源模型推論**
> - Ollama 部署 `qwen2.5:32b` Q4 量化
> - 多 GPU 共用 VRAM
> - `--keepalive 24h` 常駐 VRAM，最小化 inference latency
> - OpenAI-compatible API，Open WebUI 無縫接入
>
> **Pipeline 客製化**
> - Open WebUI filter pipeline（`type = "filter"`）
> - `inlet`：非同步 retrieve 記憶，動態修改 request body
> - `outlet`：`asyncio.create_task()` fire-and-forget，non-blocking 存記憶
> - `@tag` 機制：regex parse + `_chat_project_map` 實現跨 chat 記憶共享
> - Per-project 記憶隔離：`{account_id}_{tag}` namespace
>
> **自訂 Web 介面**
> - FastAPI 實作完整 REST API（8 個 endpoints）
> - 手刻 HTML/JS 記憶管理介面，dark theme，HTTP Basic Auth

---

## 最終系統架構

```
手機/電腦（Tailscale VPN）
        ↓  http://<tailscale-ip>:3000
Open WebUI（port 3000）
        ↓  pipeline filter
Pipelines（port 9099）
  mem0_pipeline.py
  ├── inlet()
  │   ├── extract @tag → project_id
  │   ├── embed(last_user_msg) → query_vector
  │   ├── Qdrant cosine search top-5（filter by user_id）
  │   └── inject memories → system prompt
  └── outlet()
      └── asyncio.create_task(_save_memory)
              ↓ HTTP POST /add
FastAPI mem0 service（port 8080）
  ├── embed(content) → vector
  ├── cosine dedup check（threshold 0.92）
  ├── Qdrant upsert（ADD or UPDATE）
  └── /ui 記憶管理介面
              ↓
    ┌─────────┴──────────┐
    ▼                    ▼
Qdrant（port 6333）   Ollama（port 11434）
collection: mem0_raw   ├── qwen2.5:32b（Q4 量化，19GB）
vector dim: 768        └── nomic-embed-text（274MB）
distance: cosine
```

---

## 關鍵問題排除

| 問題 | 原因 | 解法 |
|------|------|------|
| response 超慢 | outlet 同步等待 embedding | `asyncio.create_task()` fire-and-forget |
| 跨 chat 記憶破碎 | user_id 綁定 chat_id | `@tag` 機制，共享記憶 namespace |
| `QdrantClient` 無 `.search()` | 新版 API 改名 | 改用 `.query_points()` |
| 記憶重複累積 | 相似內容一直新增 | cosine > 0.92 → UPDATE |
| mem0 image pull 失敗 | Docker Hub 網路限制 | 改用 `build: ./app` 自己 build |
| Pipeline 未被偵測 | Open WebUI v0.8+ 入口改變 | 從外部連接加入 port 9099，再上傳 .py |
| GPU 沒吃到，跑很慢 | model idle unload | `--keepalive 24h` 常駐 VRAM |

---

## 工具總覽

| 工具 | 說明 | 用途 |
|------|------|------|
| Claude (claude.ai) | claude-sonnet | 架構設計、程式碼生成、debug |
| Ollama | local inference | 跑 qwen2.5:32b 和 nomic-embed-text |
| qwen2.5:32b | Q4 量化，19GB | 主力 LLM |
| nomic-embed-text | 274MB，768 dim | 文字 embedding |
| Qdrant | vector DB | cosine similarity search |
| Open WebUI | chat 介面 | 對話 + pipeline 管理 |
| FastAPI + uvicorn | Python 3.11 | mem0 REST API service |
| httpx | async HTTP client | pipeline 呼叫 mem0 API |
| Docker + Compose | 容器化 | 服務部署管理 |
| Tailscale | VPN | 手機電腦遠端連線 |
| tmux | terminal multiplexer | 多視窗 log 監控 |
| curl | HTTP client | API 手動測試 |
| nvidia-smi | GPU 監控 | 確認 GPU 使用狀況 |
| crontab | job scheduler | 定期跑 vacuum.sh |
