import json
from pathlib import Path
from typing import Iterable, List, Sequence, Optional

import pandas as pd


def load_questions(file_path: str | Path) -> List[str]:
    """
    Загружает список вопросов из файла любого поддерживаемого формата.

    Параметры:
        file_path: Путь к файлу с вопросами. Поддерживаются .json, .csv, .tsv, .xlsx, .txt.

    Возвращает:
        Список строк с вопросами без пустых значений.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл с вопросами не найден: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_questions_from_json(path)
    if suffix == ".csv":
        return _load_questions_from_csv(path)
    if suffix == ".tsv":
        return _load_questions_from_tsv(path)
    if suffix in {".xlsx", ".xls"}:
        return _load_questions_from_excel(path)

    # Текстовый фолбэк: одна строка = один вопрос
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def save_results(
    questions: Sequence[str],
    answers: Sequence[str],
    output_path: str | Path,
    *,
    fmt: str = "json",
) -> Path:
    """
    Сохраняет ответы в выбранном формате.

    Параметры:
        questions: Список исходных вопросов (по порядку).
        answers: Полученные ответы с тем же количеством элементов.
        output_path: Путь для сохранения результата.
        fmt: Формат сохранения. Доступно: 'json', 'json_pairs', 'csv', 'tsv'.

    Возвращает:
        Путь к сохранённому файлу.
    """
    if len(questions) != len(answers):
        raise ValueError("Количество вопросов и ответов должно совпадать")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "json":
        payload = list(answers)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    if fmt == "json_pairs":
        payload = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    if fmt == "csv":
        df = pd.DataFrame({"question": questions, "answer": answers})
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    if fmt == "tsv":
        df = pd.DataFrame({"question": questions, "answer": answers})
        df.to_csv(path, index=False, sep="\t", encoding="utf-8")
        return path

    raise ValueError(f"Неизвестный формат сохранения: {fmt}")


def convert_hf_dataset(
    dataset_name: str,
    output_path: str | Path,
    *,
    split: Optional[str] = None,
    text_column: str = "text",
    title_column: Optional[str] = None,
    fmt: str = "json",
    limit: Optional[int] = None,
) -> Path:
    """
    Конвертирует датасет из Hugging Face в поддерживаемый формат.

    Параметры:
        dataset_name: Имя датасета на Hugging Face (например, "jtatman/python-code-dataset-500k").
        output_path: Путь для сохранения конвертированного файла.
        split: Раздел датасета для загрузки (например, "train", "test"). Если None, используется первый доступный.
        text_column: Название столбца с текстом/кодом.
        title_column: Название столбца с заголовком (опционально).
        fmt: Формат сохранения: 'json', 'csv', 'tsv'. По умолчанию 'json'.
        limit: Максимальное количество записей для сохранения (опционально).

    Возвращает:
        Путь к сохранённому файлу.

    Пример:
        >>> convert_hf_dataset(
        ...     "jtatman/python-code-dataset-500k",
        ...     "python_code.json",
        ...     text_column="code",
        ...     limit=1000
        ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Для использования этой функции необходимо установить библиотеку datasets: "
            "pip install datasets"
        )

    # Загрузка датасета
    print(f"[CONVERT] Загрузка датасета: {dataset_name}")
    if split is None:
        ds = load_dataset(dataset_name)
        # Используем первый доступный split
        split = list(ds.keys())[0]
        print(f"[CONVERT] Используется split: {split}")
    else:
        ds = load_dataset(dataset_name, split=split)

    # Преобразование в список словарей
    records = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        record = {}
        if text_column in item:
            record["text"] = str(item[text_column])
        else:
            # Если столбца нет, пытаемся найти первый текстовый столбец
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    record["text"] = str(value)
                    break
            if "text" not in record:
                raise ValueError(
                    f"Столбец '{text_column}' не найден. Доступные столбцы: {list(item.keys())}"
                )

        if title_column and title_column in item:
            record["title"] = str(item[title_column])
        else:
            record["title"] = ""

        records.append(record)

    print(f"[CONVERT] Загружено записей: {len(records)}")

    # Сохранение в выбранном формате
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "json":
        # Формат для RAG: массив объектов с ключом 'items'
        payload = {"items": records}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "csv":
        df = pd.DataFrame(records)
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "tsv":
        df = pd.DataFrame(records)
        df.to_csv(path, index=False, sep="\t", encoding="utf-8")
    else:
        raise ValueError(f"Неподдерживаемый формат: {fmt}. Используйте 'json', 'csv' или 'tsv'")

    print(f"[CONVERT] Сохранено в: {path}")
    return path


def generate_config(
    *,
    model_name: str,
    model_dir: str | Path = "./model",
    quantize: bool = True,
    bits: int = 4,
    rag_dir: str | Path = "./rag",
    top_k: int = 3,
    threshold: float = 0.38,
    max_ctx_chars: int = 1400,
    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
    max_new_tokens: int = 48,
    seed: int = 127,
    output_path: str | Path = "config.json",
) -> Path:
    """
    Генерирует файл конфигурации config.json на основе переданных параметров.

    Параметры:
        model_name: Имя модели на HuggingFace или путь к локальной папке.
        model_dir: Папка, где хранится локальная копия модели.
        quantize: Нужна ли квантование модели.
        bits: Количество бит для квантования (4 или 8).
        rag_dir: Путь к каталогу с RAG артефактами.
        top_k: Количество чанков для Retrieval.
        threshold: Порог похожести для принятия RAG ответа.
        max_ctx_chars: Максимальная длина контекста, передаваемого модели.
        embedding_model: Имя SentenceTransformer для построения индекса.
        max_new_tokens: Лимит новых токенов при генерации.
        seed: Фиксированный seed для воспроизводимости.
        output_path: Куда сохранить конфиг (по умолчанию config.json).

    Возвращает:
        Путь к созданному файлу конфигурации.
    """
    payload = {
        "model": {
            "name": model_name,
            "dir": str(Path(model_dir).expanduser().resolve()),
            "quantize": bool(quantize),
            "bits": int(bits),
        },
        "rag": {
            "dir": str(Path(rag_dir).expanduser().resolve()),
            "top_k": int(top_k),
            "threshold": float(threshold),
            "max_ctx_chars": int(max_ctx_chars),
            "embedding_model": embedding_model,
        },
        "generation": {
            "max_new_tokens": int(max_new_tokens),
            "seed": int(seed),
        },
    }

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_questions_from_json(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], Iterable):
            return _stringify_items(data["questions"])
        raise ValueError("JSON-объект должен содержать ключ 'questions'")
    if isinstance(data, list):
        return _stringify_items(data)
    raise ValueError("JSON-файл должен содержать массив или объект с ключом 'questions'")


def _load_questions_from_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "question" in df.columns:
        series = df["question"]
    else:
        series = df.iloc[:, 0]
    return _stringify_items(series.tolist())


def _load_questions_from_tsv(path: Path) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    if "question" in df.columns:
        series = df["question"]
    else:
        series = df.iloc[:, 0]
    return _stringify_items(series.tolist())


def _load_questions_from_excel(path: Path) -> List[str]:
    df = pd.read_excel(path)
    if "question" in df.columns:
        series = df["question"]
    else:
        series = df.iloc[:, 0]
    return _stringify_items(series.tolist())


def _stringify_items(items: Iterable) -> List[str]:
    out: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("question", "")).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out

