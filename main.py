from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from pydantic import BaseModel
from typing import List
import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import ollama
import uuid
import time

app = FastAPI()

security = HTTPBasic()
UI_USER = os.getenv('UI_USER', 'esther')
UI_PASS = os.getenv('UI_PASS', 'changeme')

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, UI_USER)
    ok_pass = secrets.compare_digest(credentials.password, UI_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Unauthorized', headers={'WWW-Authenticate': 'Basic'})
    return credentials.username

QDRANT_HOST = os.getenv("QDRANT_HOST", "172.17.0.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://172.17.0.1:11434")
COLLECTION = "mem0_raw"
EMBED_MODEL = "nomic-embed-text"
VECTOR_DIM = 768

qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
ol = ollama.Client(host=OLLAMA_URL)

def ensure_collection():
    cols = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in cols:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
        )

ensure_collection()

def embed(text: str) -> list:
    resp = ol.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

class MemoryRequest(BaseModel):
    messages: list
    user_id: str

class SearchRequest(BaseModel):
    query: str
    user_id: str

class AddSingleRequest(BaseModel):
    memory: str
    user_id: str

@app.post("/add")
def add_memory(req: MemoryRequest):
    stored = []
    for msg in req.messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if not content or len(content) < 8:
                continue
            vector = embed(content)
            # 檢查是否有相似記憶（threshold 0.92）
            hits = qdrant.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=1,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=req.user_id))]
                )
            )
            if hits.points and hits.points[0].score > 0.92:
                # 太相似，更新現有記憶而不是新增
                existing_id = str(hits.points[0].id)
                qdrant.upsert(
                    collection_name=COLLECTION,
                    points=[PointStruct(
                        id=existing_id,
                        vector=vector,
                        payload={"user_id": req.user_id, "memory": content, "created_at": time.time()}
                    )]
                )
                stored.append({"id": existing_id, "memory": content, "event": "UPDATE"})
            else:
                point_id = str(uuid.uuid4())
                qdrant.upsert(
                    collection_name=COLLECTION,
                    points=[PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"user_id": req.user_id, "memory": content, "created_at": time.time()}
                    )]
                )
                stored.append({"id": point_id, "memory": content, "event": "ADD"})
    return {"status": "ok", "result": {"results": stored}}

@app.post("/add_single")
def add_single(req: AddSingleRequest):
    vector = embed(req.memory)
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector=vector,
            payload={"user_id": req.user_id, "memory": req.memory, "created_at": time.time()}
        )]
    )
    return {"status": "ok", "id": point_id}

@app.post("/search")
def search_memory(req: SearchRequest):
    vector = embed(req.query)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=5,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=req.user_id))]
        )
    )
    results = [{"memory": h.payload["memory"], "score": h.score} for h in hits.points]
    return {"memories": {"results": results}}

@app.get("/get/{user_id}")
def get_all(user_id: str):
    hits, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    results = [{"id": str(h.id), "memory": h.payload["memory"], "created_at": h.payload.get("created_at")} for h in hits]
    return {"memories": {"results": results}}

@app.get("/projects")
def get_projects():
    hits, _ = qdrant.scroll(collection_name=COLLECTION, limit=1000, with_payload=True, with_vectors=False)
    raw = list(set(h.payload["user_id"] for h in hits))
    projects = [{"id": p, "label": "_".join(p.split("_")[1:]) if "_" in p else p} for p in raw]
    return {"projects": sorted(projects, key=lambda x: x["label"])}

@app.delete("/delete/{user_id}")
def delete_all(user_id: str):
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        )
    )
    return {"status": "deleted"}

@app.delete("/delete_one/{point_id}")
def delete_one(point_id: str):
    qdrant.delete(collection_name=COLLECTION, points_selector=models.PointIdsList(points=[point_id]))
    return {"status": "deleted"}

@app.get("/ui", response_class=HTMLResponse)
def ui(username: str = Depends(check_auth)):
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>記憶管理</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
  .header { background: #1a1a1a; padding: 20px 24px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header span { font-size: 13px; color: #888; }
  .container { max-width: 900px; margin: 0 auto; padding: 24px; }
  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .card h2 { font-size: 14px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }
  select, input, textarea { width: 100%; background: #111; border: 1px solid #333; border-radius: 8px; padding: 10px 12px; color: #e0e0e0; font-size: 14px; outline: none; }
  select:focus, input:focus, textarea:focus { border-color: #555; }
  textarea { resize: vertical; min-height: 80px; }
  .btn { padding: 9px 16px; border-radius: 8px; border: none; cursor: pointer; font-size: 13px; font-weight: 500; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.8; }
  .btn-primary { background: #3b82f6; color: white; }
  .btn-danger { background: #ef4444; color: white; }
  .btn-sm { padding: 5px 10px; font-size: 12px; }
  .row { display: flex; gap: 10px; margin-top: 10px; }
  .memory-item { background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
  .memory-text { font-size: 14px; line-height: 1.5; flex: 1; }
  .memory-meta { font-size: 11px; color: #555; margin-top: 4px; }
  .badge { display: inline-block; background: #2a2a2a; border-radius: 4px; padding: 2px 8px; font-size: 12px; cursor: pointer; margin: 3px; border: 1px solid #333; }
  .badge:hover { border-color: #555; }
  .badge.active { border-color: #3b82f6; color: #3b82f6; }
  #status { font-size: 13px; color: #4ade80; margin-top: 8px; min-height: 20px; }
  .empty { color: #555; font-size: 14px; text-align: center; padding: 20px; }
  .count { font-size: 12px; color: #555; margin-bottom: 12px; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🧠 記憶管理</h1>
    <span>Esther 的小說寫作記憶庫</span>
  </div>
</div>
<div class="container">

  <div class="card">
    <h2>選擇 Project</h2>
    <div id="project-list">載入中...</div>
    <div class="row">
      <input id="custom-project" placeholder="或手動輸入 project ID" style="flex:1">
      <button class="btn btn-primary" onclick="loadMemories()">載入記憶</button>
    </div>
  </div>

  <div class="card">
    <h2>新增記憶</h2>
    <textarea id="new-memory" placeholder="輸入要手動新增的記憶內容..."></textarea>
    <div class="row">
      <button class="btn btn-primary" onclick="addMemory()">新增</button>
      <span id="status"></span>
    </div>
  </div>

  <div class="card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
      <h2 style="margin:0">記憶列表</h2>
      <button class="btn btn-danger btn-sm" onclick="deleteAll()">刪除此 Project 全部記憶</button>
    </div>
    <div id="memory-count" class="count"></div>
    <div id="memory-list"><div class="empty">請先選擇 Project</div></div>
  </div>

</div>
<script>
let currentProject = '';

async function fetchProjects() {
  const r = await fetch('/projects');
  const d = await r.json();
  const el = document.getElementById('project-list');
  if (!d.projects.length) { el.innerHTML = '<div class="empty">尚無 Project</div>'; return; }
  el.innerHTML = d.projects.map(p => `<span class="badge" onclick="selectProject('${p.id}')">${p.label}</span>`).join('');
}

function selectProject(p) {
  currentProject = p;
  document.getElementById('custom-project').value = p;
  document.querySelectorAll('.badge').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  loadMemories();
}

async function loadMemories() {
  const p = document.getElementById('custom-project').value.trim() || currentProject;
  if (!p) return alert('請選擇或輸入 Project');
  currentProject = p;
  const r = await fetch('/get/' + encodeURIComponent(p));
  const d = await r.json();
  const list = d.memories.results;
  const el = document.getElementById('memory-list');
  document.getElementById('memory-count').textContent = `共 ${list.length} 條記憶`;
  if (!list.length) { el.innerHTML = '<div class="empty">此 Project 尚無記憶</div>'; return; }
  el.innerHTML = list.map(m => `
    <div class="memory-item" id="m-${m.id}">
      <div style="flex:1">
        <div class="memory-text">${m.memory}</div>
        <div class="memory-meta">${m.id}</div>
      </div>
      <button class="btn btn-danger btn-sm" onclick="deleteOne('${m.id}')">刪除</button>
    </div>`).join('');
}

async function addMemory() {
  const p = currentProject || document.getElementById('custom-project').value.trim();
  const m = document.getElementById('new-memory').value.trim();
  if (!p) return alert('請先選擇 Project');
  if (!m) return alert('請輸入記憶內容');
  const r = await fetch('/add_single', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({memory: m, user_id: p})
  });
  const d = await r.json();
  document.getElementById('status').textContent = '✓ 已新增';
  document.getElementById('new-memory').value = '';
  setTimeout(() => document.getElementById('status').textContent = '', 2000);
  loadMemories();
}

async function deleteOne(id) {
  if (!confirm('確定刪除這條記憶？')) return;
  await fetch('/delete_one/' + id, {method: 'DELETE'});
  document.getElementById('m-' + id)?.remove();
  const items = document.querySelectorAll('[id^="m-"]').length;
  document.getElementById('memory-count').textContent = `共 ${items} 條記憶`;
}

async function deleteAll() {
  if (!currentProject) return alert('請先選擇 Project');
  if (!confirm(`確定刪除 ${currentProject} 的全部記憶？`)) return;
  await fetch('/delete/' + encodeURIComponent(currentProject), {method: 'DELETE'});
  loadMemories();
}

fetchProjects();
</script>
</body>
</html>"""
