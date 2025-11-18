from __future__ import annotations
from typing import List, Dict, Optional

from factual_model import FactualModel


class VibeChatSession:
    """
    Минимальная сессия чата для «вайбкодинга», без фронта и серверов.
    Хранит историю сообщений и использует FactualModel.vibecode_chat.
    """

    def __init__(
        self,
        model: Optional[FactualModel] = None,
        *,
        max_new_tokens: int = 2048,
        force_lang: str = "python",
        rag_fallback: bool = False,
    ) -> None:
        self.model = model or FactualModel()
        self.history: List[Dict[str, str]] = []
        self.max_new_tokens = int(max_new_tokens)
        self.force_lang = force_lang
        self.rag_fallback = bool(rag_fallback)

    def ask(self, prompt: str) -> str:
        """Добавляет user-сообщение и возвращает ответ assistant."""
        self.history.append({"role": "user", "content": str(prompt)})
        text = self.model.vibecode_chat(self.history, max_new_tokens=self.max_new_tokens)

        # Мягкий фолбэк на RAG при плохом ответе
        if self.rag_fallback:
            low = (text or "").strip().lower()
            if (not text) or self.model._is_broken(text) or low == "не знаю":
                alt = self.model.generate(prompt)
                if alt and alt.strip().lower() != "не знаю" and not self.model._is_broken(alt):
                    text = alt

        self.history.append({"role": "assistant", "content": text})
        return text or "Не знаю"

    def reset(self) -> None:
        """Сбрасывает историю."""
        self.history.clear()

    def config(
        self,
        *,
        max_new_tokens: Optional[int] = None,
        force_lang: Optional[str] = None,
        rag_fallback: Optional[bool] = None,
    ) -> None:
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)
        if force_lang is not None:
            self.force_lang = str(force_lang)
        if rag_fallback is not None:
            self.rag_fallback = bool(rag_fallback)


