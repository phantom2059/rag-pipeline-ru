# RAG Pipeline (Russian)

**RAG-пайплайн для русскоязычных вопросно-ответных систем**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0866FF?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## Overview

Полноценный RAG-пайплайн, объединяющий плотный векторный поиск по базе знаний с генерацией ответов через квантованную LLM. Спроектирован для русскоязычных фактологических QA-задач с поддержкой морфологической лемматизации, математического fallback и автоматического переключения GPU/CPU.

**Модель**: [Mistral-7B-Saiga](https://huggingface.co/Gaivoronsky/Mistral-7B-Saiga) (4-bit NF4 или 8-bit INT8 квантование через bitsandbytes)
**Эмбеддинги**: `sentence-transformers/all-MiniLM-L12-v2`
**Индекс**: FAISS (dense retrieval, cosine similarity)

---

## Architecture

Система обрабатывает запрос через три пути:

1. **Math shortcut** — если запрос содержит вычислимое выражение, ответ рассчитывается напрямую без инференса модели.
2. **RAG path** — запрос нормализуется и лемматизируется через pymorphy2, FAISS возвращает top-k чанков, модель генерирует ответ с учётом контекста (срабатывает при score выше `DECISION_THRESHOLD`).
3. **Fallback** — генерация без контекста, когда RAG недоступен или scores слишком низкие.

Постобработка удаляет эхо-ответы (повторение вопроса), валидирует длину, обрабатывает запросы на перевод.

---

## Project structure

```
├── factual_model.py       # Основной класс: загрузка модели, RAG-логика, генерация
├── rag_builder.py         # Построение индекса из CSV / XLSX / JSON / TXT
├── model_downloader.py    # Скачивание модели + квантование + генерация config.json
├── solution.py            # CLI-точка входа (argparse)
├── utils.py               # Хелперы I/O, генерация конфигов
├── lem_worker.py          # Лемматизация русского языка (pymorphy2)
├── quantize_model.py      # Утилиты квантования
├── quick_start.ipynb      # Пошаговый ноутбук
├── saiga_benchmark.ipynb  # Ноутбук с бенчмарками
├── vibechat.py            # Интерактивный чат
└── requirements.txt
```

RAG-артефакты (генерируются `rag_builder.py`):

```
rag/
├── meta.json          # Метаданные embedding-модели и параметры
├── index.faiss        # Плотный векторный индекс
└── chunks.parquet     # Текстовые чанки (сжатые)
```

---

## Quick start

### Установка

```bash
pip install -U torch transformers accelerate sentencepiece bitsandbytes
pip install -U pandas openpyxl pyarrow pymorphy2 tabulate tqdm
pip install -U faiss-gpu-cu121  # или faiss-cpu
pip install -U sentence-transformers
```

### Скачивание модели + построение индекса

```python
from model_downloader import download_model
from rag_builder import create_rag_index

download_model()           # скачивает Mistral-7B-Saiga, квантует, пишет config.json
create_rag_index("data/")  # строит FAISS-индекс из ваших документов
```

Готовый RAG-индекс по Википедии доступен для скачивания: [Яндекс.Диск](https://disk.yandex.ru/d/tj4taNOiHZnPew).

### Запуск

**Одиночный запрос:**

```python
from factual_model import FactualModel

model = FactualModel()
print(model.generate("Что такое квантовая механика?"))
```

**Батч-обработка (CLI):**

```bash
python solution.py \
  --input questions.csv \
  --output answers.csv \
  --format csv \
  --batch-size 8 \
  --config config.json
```

**Ноутбук:** `quick_start.ipynb` — установка, скачивание модели, построение RAG, инференс и табличный вывод в одном потоке.

**Google Colab:** `colab_test.ipynb` — клон репо, установка зависимостей, инференс в один клик. Используйте GPU runtime (T4 или лучше); для Colab рекомендуется 8-bit квантование.

---

## Configuration

Все параметры хранятся в `config.json` (генерируется автоматически через `download_model()`):

```json
{
  "model": {
    "name": "Gaivoronsky/Mistral-7B-Saiga",
    "dir": "./model/",
    "quantize": true,
    "bits": 8
  },
  "rag": {
    "dir": "./rag/",
    "top_k": 3,
    "threshold": 0.38,
    "max_ctx_chars": 1400,
    "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"
  },
  "generation": {
    "max_new_tokens": 48,
    "seed": 127
  }
}
```

Переопределение параметров в рантайме через `config_overrides` в конструкторе `FactualModel`.

---

## Key features

- **Автоматическое определение GPU** — работает на CUDA при наличии, прозрачно переключается на CPU
- **Гибкое квантование** — `download_model()` поддерживает 4-bit NF4 и 8-bit INT8 через bitsandbytes
- **RAG builder** — единая функция для подготовки FAISS-индекса с лемматизацией и нормализацией из любого табличного или текстового источника
- **Батч-обработка** — `generate_batch()` работает с любым форматом входных данных (JSON, CSV, XLSX, TXT)
- **Интерактивный чат** — `vibechat.py` для диалогового QA в терминале
- **Multi-path resolution** — математический shortcut, RAG-генерация и fallback без контекста

---

## Requirements

- Python 3.8+
- CUDA (опционально — автоматический CPU fallback)
- 16 GB RAM минимум (24 GB+ рекомендуется для GPU-инференса)

---

## License

MIT
