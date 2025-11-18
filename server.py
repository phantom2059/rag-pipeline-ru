import threading
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from factual_model import FactualModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = None
    rag_fallback: bool = True  # попытка RAG при пустом/слом. ответе


class ChatResponse(BaseModel):
    content: str


app = FastAPI(title="VibeCoding API", version="1.0.0")

_model_lock = threading.Lock()
_model_instance: Optional[FactualModel] = None


def _get_model() -> FactualModel:
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = FactualModel()
    return _model_instance


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/vibechat", response_model=ChatResponse)
def vibechat(req: ChatRequest) -> ChatResponse:
    """
    Чат-эндпоинт для «вайбкодинга»:
    - Использует строгий системный промпт для кодинга
    - Greedy decoding, большой максимум токенов
    - По желанию делает фолбэк на RAG, если ответ пустой/сломанный
    """
    model = _get_model()

    # Подготавливаем историю
    messages: List[Dict[str, str]] = [{"role": m.role, "content": m.content} for m in req.messages]

    with _model_lock:
        text = model.vibecode_chat(messages, max_new_tokens=req.max_new_tokens)
        if req.rag_fallback:
            # Если ответ плохой — пробуем RAG на последнем user-сообщении
            if not text or model._is_broken(text) or text.strip().lower() == "не знаю":
                # Берем последнюю реплику пользователя
                last_user = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user = m.get("content") or ""
                        break
                if last_user:
                    alt = model.generate(last_user)
                    if alt and alt.strip().lower() != "не знаю" and not model._is_broken(alt):
                        text = alt

    return ChatResponse(content=text or "Не знаю")


if __name__ == "__main__":
    # Локальный запуск: uvicorn server:app --host 127.0.0.1 --port 8000
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)  # слушаем только localhost


