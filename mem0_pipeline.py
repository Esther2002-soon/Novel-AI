from typing import Optional, List
import httpx
import asyncio
import re
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        MEM0_URL: str = "http://172.17.0.1:8080"
        pipelines: List[str] = ["*"]
        priority: int = 0

    def __init__(self):
        self.name = "Mem0 記憶管理"
        self.type = "filter"
        self.valves = self.Valves()
        self._chat_project_map = {}  # chat_id → project_id 暫存

    async def on_startup(self):
        print("Mem0 Pipeline 啟動")

    async def on_shutdown(self):
        pass

    def _extract_tag(self, messages: list) -> Optional[str]:
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                match = re.search(r"@([\w\u4e00-\u9fff]+)", msg.get("content", ""))
                if match:
                    return match.group(1)
        return None

    def _get_project_id(self, body: dict, user: Optional[dict], store=False) -> str:
        account_id = (user or {}).get("id", "default_user")
        chat_id = body.get("metadata", {}).get("chat_id") or body.get("chat_id", "default")
        messages = body.get("messages", [])

        tag = self._extract_tag(messages)
        if tag:
            project_id = f"{account_id}_{tag}"
            if store:
                self._chat_project_map[chat_id] = project_id
                print(f"Project tag 找到: {tag} → 存入 map[{chat_id}]")
            return project_id

        # outlet 時從 map 查
        if chat_id in self._chat_project_map:
            return self._chat_project_map[chat_id]
        # 沒有 tag 就不處理
        return None

        return f"{account_id}_{chat_id}"

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        user_id = self._get_project_id(body, user, store=True)
        if user_id is None:
            return body
        messages = body.get("messages", [])

        if not messages:
            return body

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.valves.MEM0_URL}/search",
                    json={"query": last_user_msg, "user_id": user_id}
                )
                data = resp.json()
                memories = data.get("memories", {})
                if isinstance(memories, dict):
                    memories = memories.get("results", [])
                elif not isinstance(memories, list):
                    memories = []
                print(f"[{user_id}] 找到記憶: {len(memories)} 條")
        except Exception as e:
            print(f"Mem0 search 失敗: {repr(e)}")
            memories = []

        if memories:
            memory_text = "\n".join([f"- {m['memory']}" for m in memories[:5]])
            memory_injection = f"\n\n【重要：以下是你必須遵守的已知設定，不可違背或自行編造】\n{memory_text}\n【設定結束】\n"
            has_system = False
            for msg in body["messages"]:
                if msg["role"] == "system":
                    msg["content"] += memory_injection
                    has_system = True
                    break
            if not has_system:
                body["messages"].insert(0, {
                    "role": "system",
                    "content": memory_injection
                })

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        user_id = self._get_project_id(body, user, store=False)
        if user_id is None:
            return body
        messages = body.get("messages", [])
        last_two = [m for m in messages if m["role"] in ["user", "assistant"]][-2:]

        if len(last_two) >= 2:
            asyncio.create_task(self._save_memory(last_two, user_id))

        return body

    async def _save_memory(self, messages: list, user_id: str):
        print(f"存入內容: {messages}")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.valves.MEM0_URL}/add",
                    json={"messages": messages, "user_id": user_id}
                )
                print(f"[{user_id}] 記憶已存入: {resp.status_code}")
        except Exception as e:
            print(f"Mem0 背景存入失敗: {repr(e)}")
