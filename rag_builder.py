import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Не удалось импортировать faiss. Установите faiss-cpu или faiss-gpu."
    ) from exc

try:
    from lem_worker import lemmatize_text as _lemmatize_ru
except Exception:  # pragma: no cover
    _lemmatize_ru = None


_RE_PUNCT = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_RE_SPACE = re.compile(r"\s+", flags=re.UNICODE)


def create_rag_index(
    data_path: str | Path,
    *,
    output_dir: str | Path = "./rag",
    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    text_column: str = "text",
    title_column: str = "title",
) -> Path:
    """
    Строит полный RAG-индекс (chunks.parquet, index.faiss, meta.json) из входных данных.

    Параметры:
        data_path: Путь к исходным данным (.json/.csv/.tsv/.xlsx/.txt).
        output_dir: Каталог для сохранения артефактов RAG.
        embedding_model: Имя SentenceTransformer для векторизации.
        chunk_size: Размер чанка (в символах).
        chunk_overlap: Перекрытие между чанками (в символах).
        text_column: Название текстового столбца в таблице.
        title_column: Название столбца с заголовком/темой.

    Возвращает:
        Путь к каталогу с созданными артефактами.
    """
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RAG_BUILDER] Загрузка данных: {data_path}")
    documents = _load_documents(data_path, text_column=text_column, title_column=title_column)
    print(f"[RAG_BUILDER] Документов: {len(documents)}")

    print("[RAG_BUILDER] Формирование чанков...")
    chunks = _chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[RAG_BUILDER] Получено чанков: {len(chunks)}")

    parquet_path = output_dir / "chunks.parquet"
    _write_chunks_parquet(chunks, parquet_path)
    print(f"[RAG_BUILDER] Сохранены чанки: {parquet_path}")

    print("[RAG_BUILDER] Подготовка текстов для эмбеддингов...")
    texts_for_encoding = [_normalize_with_lemma(text) for _, text in chunks]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RAG_BUILDER] Загрузка SentenceTransformer ({embedding_model}) на {device}")
    retr_model = SentenceTransformer(embedding_model, device=device)

    print("[RAG_BUILDER] Вычисление эмбеддингов...")
    embeddings = retr_model.encode(
        texts_for_encoding,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    index_path = output_dir / "index.faiss"
    _write_faiss_index(embeddings, index_path)
    print(f"[RAG_BUILDER] Сохранён индекс: {index_path}")

    meta_path = output_dir / "meta.json"
    _write_meta(
        meta_path,
        embedding_model=embedding_model,
        dim=embeddings.shape[1],
        chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"[RAG_BUILDER] Сохранён meta.json: {meta_path}")
    return output_dir


def _load_documents(path: Path, *, text_column: str, title_column: str) -> List[Tuple[str, str]]:
    """Загружает документы и возвращает список (title, text)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_from_json(path, text_column=text_column, title_column=title_column)
    if suffix == ".csv":
        return _load_from_dataframe(pd.read_csv(path), text_column=text_column, title_column=title_column)
    if suffix == ".tsv":
        return _load_from_dataframe(pd.read_csv(path, sep="\t"), text_column=text_column, title_column=title_column)
    if suffix in {".xlsx", ".xls"}:
        return _load_from_dataframe(pd.read_excel(path), text_column=text_column, title_column=title_column)

    text = path.read_text(encoding="utf-8")
    return [("", text)]


def _load_from_json(
    path: Path,
    *,
    text_column: str,
    title_column: str,
) -> List[Tuple[str, str]]:
    """Загружает документы из JSON (массив объектов или объект с ключом 'items')."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            data = data["items"]
        else:
            raise ValueError("JSON объект должен содержать массив в ключе 'items'")
    if not isinstance(data, list):
        raise ValueError("JSON должен содержать список документов")
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get(title_column) or "").strip()
        text = str(item.get(text_column) or "").strip()
        if text:
            rows.append((title, text))
    return rows


def _load_from_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str,
    title_column: str,
) -> List[Tuple[str, str]]:
    """Извлекает документы из DataFrame."""
    if text_column not in df.columns:
        text_series = df.iloc[:, 0]
    else:
        text_series = df[text_column]
    if title_column in df.columns:
        title_series = df[title_column]
    else:
        title_series = pd.Series([""] * len(text_series))
    rows = []
    for title, text in zip(title_series.tolist(), text_series.tolist()):
        text = (str(text or "")).strip()
        title = (str(title or "")).strip()
        if text:
            rows.append((title, text))
    return rows


def _chunk_documents(
    documents: Sequence[Tuple[str, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, str]]:
    """Разбивает тексты на чанки фиксированного размера."""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size должен быть больше chunk_overlap")

    chunks: List[Tuple[str, str]] = []
    step = chunk_size - chunk_overlap
    for title, text in documents:
        text = text.strip()
        if not text:
            continue
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size].strip()
            if len(chunk) < 20:
                continue
            chunks.append((title, chunk))
            if start + chunk_size >= len(text):
                break
    return chunks


def _write_chunks_parquet(chunks: Sequence[Tuple[str, str]], path: Path) -> None:
    """Сохраняет чанки в Parquet формате."""
    table = pa.Table.from_pydict(
        {
            "title": [title for title, _ in chunks],
            "text": [text for _, text in chunks],
        }
    )
    pq.write_table(table, path)


def _write_faiss_index(embeddings: np.ndarray, path: Path) -> None:
    """Создаёт FAISS индекс и сохраняет на диск."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(path))


def _write_meta(
    path: Path,
    *,
    embedding_model: str,
    dim: int,
    chunks: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Сохраняет metadata JSON для последующей загрузки."""
    payload = {
        "model": embedding_model,
        "dim": dim,
        "chunks": chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_with_lemma(text: str) -> str:
    """Применяет нормализацию и лемматизацию к тексту для эмбеддингов."""
    base = _norm(text)
    if _lemmatize_ru is None:
        return base
    try:
        return _lemmatize_ru(base)
    except Exception:
        return base


def _norm(s: str) -> str:
    """Повторяет нормализацию из factual_model (нижний регистр + фильтр знаков)."""
    t = (s or "").lower().strip()
    t = _RE_PUNCT.sub(" ", t)
    t = _RE_SPACE.sub(" ", t)
    return t.strip()

