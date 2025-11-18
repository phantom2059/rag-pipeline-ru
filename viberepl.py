import argparse
from typing import List, Dict

from factual_model import FactualModel


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCoding REPL (CLI чат без веба)")
    parser.add_argument("--max-new", type=int, default=2048, help="Максимум новых токенов")
    parser.add_argument("--no-rag-fallback", action="store_true", help="Отключить фолбэк через RAG")
    args = parser.parse_args()

    model = FactualModel()
    history: List[Dict[str, str]] = []

    print("VibeCoding REPL. Команды: /reset, /exit")
    while True:
        try:
            user = input("you> ").rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history.clear()
            print("(история очищена)")
            continue

        history.append({"role": "user", "content": user})
        # Базовый ответ
        text = model.vibecode_chat(history, max_new_tokens=args.max_new)

        # Фолбэк через RAG при плохом ответе
        if not args.no_rag_fallback:
            if not text or model._is_broken(text) or text.strip().lower() == "не знаю":
                text_alt = model.generate(user)
                if text_alt and text_alt.strip().lower() != "не знаю" and not model._is_broken(text_alt):
                    text = text_alt

        print("ai> " + (text or "Не знаю"))
        history.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()


