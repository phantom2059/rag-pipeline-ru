import os
import re
import json
import sys
import torch
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)


DEFAULT_CONFIG: Dict[str, Dict] = {
    "model": {
        "name": "Gaivoronsky/Mistral-7B-Saiga",
        "dir": "./model",
        "quantize": True,
        "bits": 8,
    },
    "rag": {
        "dir": "./rag",
        "top_k": 3,
        "threshold": 0.38,
        "max_ctx_chars": 1400,
        "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
    },
    "generation": {
        "max_new_tokens": 48,
        "seed": 127,
    },
}



SYSTEM_PROMPT = (
    "Ты — лаконичный и полезный ассистент на русском языке.\n\n"
    "ПРАВИЛА:\n"
    "1. Отвечай кратко и по делу.\n"
    "2. Если ответа не знаешь — отвечай: 'Не знаю'.\n"
    "3. Можно использовать общеизвестные факты без излишней осторожности.\n"
    "4. Переводы — только перевод без кавычек и комментариев.\n"
)

# Специальный промпт для режима кодинга (максимальная точность и детализация)
VIBECODE_SYSTEM_PROMPT = (
    "Ты — профессиональный программист и эксперт по написанию кода.\n"
    "Твоя задача — давать максимально точные, полные и рабочие решения.\n"
    "Правила:\n"
    "1. Пиши чистый, эффективный и хорошо документированный код.\n"
    "2. Если требуется объяснение — делай его кратким, но понятным.\n"
    "3. Не обрезай код, предоставляй полные листинги.\n"
    "4. Игнорируй ограничения на длину ответа, приоритет — качество кода."
)

# Строгий фолбэк-промпт для перегенерации «сломанных» ответов
STRICT_FALLBACK_PROMPT = (
    "Ты — краткий и точный ассистент.\n"
    "Правила:\n"
    "1) Одно завершённое предложение (до 12 слов).\n"
    "2) Не повторяй вопрос, не добавляй ролей.\n"
    "3) Если не уверен или вопрос абсурден — ответь: 'Не знаю'.\n"
    "4) Формулы допускаются; можно кратко расшифровать символы: 'где …'.\n"
    "5) Для переводов отвечай только переводом, без кавычек."
)

# System-промпт для RAG-режима
RAG_SYSTEM_PROMPT = (
    "Ты — краткий и точный ассистент на русском языке.\n\n"
    "СТРОГИЕ ПРАВИЛА:\n"
    "1. Отвечай одним коротким предложением (до 10 слов).\n"
    "2. Используй ТОЛЬКО предоставленный контекст.\n"
    "3. Если в контексте нет ответа — отвечай ровно: 'Не знаю'.\n"
    "4. Не рассуждай, не объясняй, не добавляй лишних деталей.\n"
    "5. Не повторяй вопрос и не добавляй префиксы ролей.\n"
)

def _merge_dict(base: Dict, updates: Dict) -> Dict:
    """Рекурсивно объединяет словари, не теряя вложенные структуры."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _load_runtime_config(config_path: Optional[str | Path], overrides: Optional[Dict]) -> Dict:
    """Загружает конфигурацию из файла и применяет переопределения во время выполнения."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _merge_dict(cfg, data)
            except Exception as exc:
                print(f"[CONFIG] Ошибка чтения {path}: {exc}", file=sys.stderr)
    if overrides:
        _merge_dict(cfg, overrides)
    return cfg


def _has_local_model(path: Path) -> bool:
    """Проверяет, существует ли локальная папка с моделью."""
    return path.is_dir() and any(path.iterdir())


class FactualModel:
    """Основной класс LLM+RAG с автоматической загрузкой модели и конфигурации."""

    def __init__(
        self,
        config_path: str | Path | None = "config.json",
        *,
        config_overrides: Optional[Dict] = None,
    ):
        """
        Инициализирует пайплайн генерации ответов.

        Параметры:
            config_path: путь к config.json (если отсутствует — используются значения по умолчанию).
            config_overrides: словарь с параметрами для принудительного переопределения.
        """
        self._config = _load_runtime_config(config_path, config_overrides)

        model_cfg = self._config.get("model", {})
        self.model_repo = model_cfg.get("name", DEFAULT_CONFIG["model"]["name"])
        self.model_dir = Path(model_cfg.get("dir", "./model")).expanduser().resolve()
        self.quantize = bool(model_cfg.get("quantize", True))
        self.quant_bits = int(model_cfg.get("bits", 4))
        self._local_model_available = _has_local_model(self.model_dir)
        self.model_name = str(self.model_dir if self._local_model_available else self.model_repo)

        gen_cfg = self._config.get("generation", {})
        self.max_new_tokens = int(gen_cfg.get("max_new_tokens", DEFAULT_CONFIG["generation"]["max_new_tokens"]))
        self.seed = int(gen_cfg.get("seed", DEFAULT_CONFIG["generation"]["seed"]))

        rag_cfg = self._config.get("rag", {})
        self.TOP_K = int(rag_cfg.get("top_k", DEFAULT_CONFIG["rag"]["top_k"]))
        self.DECISION_THRESHOLD = float(rag_cfg.get("threshold", DEFAULT_CONFIG["rag"]["threshold"]))
        self.MAX_CTX_CHARS = int(rag_cfg.get("max_ctx_chars", DEFAULT_CONFIG["rag"]["max_ctx_chars"]))
        self.RAG_DIR = Path(rag_cfg.get("dir", "./rag")).expanduser().resolve()
        self._rag_search_paths = list(dict.fromkeys([
            self.RAG_DIR,
            Path("./rag").resolve(),
            Path("./artifacts/rag").resolve(),
            Path("./data/rag").resolve(),
            Path("/app/rag"),
        ]))

        self.model, self.tokenizer, self.generation_config = self._load_model()
        # --- RAG state (ленивая инициализация) ---
        self._rag_inited = False
        self._rag_ok = False
        self._retr_model = None
        self._faiss_index = None
        self._pf = None
        self._rag_meta = {}
        self._rg_sizes = []
        self._rg_cum = []  # префиксные суммы по row-group
        self._lemmatize_ru = None

    def _load_model(self):
        # Отключаем оффлайн-режим для возможности скачивания модели при первом запуске
        # Модель автоматически скачается и закэшируется
        model_source = self.model_name
        local_only = self._local_model_available
        print(f"[MODEL] Loading model from {model_source}", file=sys.stderr)
        
        # Загрузка токенизатора из локальной папки (без сети)
        print(f"[MODEL] Loading tokenizer...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
        print(f"[MODEL] ✓ Tokenizer loaded (vocab_size={len(tokenizer)})", file=sys.stderr)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            gen_cfg = GenerationConfig.from_pretrained(model_source, local_files_only=local_only)
        except Exception:
            gen_cfg = GenerationConfig(
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Проверяем CUDA перед загрузкой модели
        use_cuda = False
        if torch.cuda.is_available():
            try:
                # Пробуем получить информацию о CUDA - если там ошибка, упадем здесь
                _ = torch.cuda.get_device_properties(0)
                # Пробуем выделить небольшой тензор для проверки реальной работоспособности
                test_tensor = torch.zeros(1, device='cuda:0')
                del test_tensor
                torch.cuda.empty_cache()
                use_cuda = True
                print(f"[MODEL] CUDA available and accessible", file=sys.stderr)
            except Exception as e:
                print(f"[MODEL] ⚠ CUDA error during initialization: {e}", file=sys.stderr)
                print(f"[MODEL] Falling back to CPU mode", file=sys.stderr)
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                use_cuda = False
        
        # Настройки для безопасной работы с CUDA
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Пробуем отключить warmup через переменную окружения (если поддерживается)
        # Также сбрасываем BNB_CUDA_VERSION для автодетекта, если есть проблемы
        if use_cuda:
            # Убираем BNB_CUDA_VERSION для автодетекта версии CUDA
            if 'BNB_CUDA_VERSION' in os.environ:
                original_bnb = os.environ.get('BNB_CUDA_VERSION')
                print(f"[MODEL] BNB_CUDA_VERSION was set to {original_bnb}, trying autodetect...", file=sys.stderr)
                os.environ.pop('BNB_CUDA_VERSION', None)
        
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

        common_kwargs = {
            'torch_dtype': dtype,
            'low_cpu_mem_usage': True,
            'offload_folder': 'offload',
        }
        if use_cuda:
            common_kwargs['device_map'] = {'': 0}
            common_kwargs['max_memory'] = {0: '24GiB', 'cpu': '48GiB'}
        else:
            common_kwargs['max_memory'] = {'cpu': '48GiB'}

        print(f"[MODEL] Loading config...", file=sys.stderr)
        cfg = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
        print(f"[MODEL] ✓ Config loaded (model_type={cfg.model_type}, hidden_size={cfg.hidden_size})", file=sys.stderr)

        # Защита от попытки наложить BitsAndBytes на уже квантованную модель (AWQ/GPTQ)
        is_prequantized = any(x in self.model_name.upper() for x in ["AWQ", "GPTQ"])
        if is_prequantized and self.quantize:
            print(f"[MODEL] ⚠ Модель {self.model_name} уже квантована (AWQ/GPTQ). Отключаем BitsAndBytes.", file=sys.stderr)
            self.quantize = False

        quant_cfg = None
        if self.quantize:
            if self.quant_bits == 4:
                print("[MODEL] Готовим NF4 квантование (4-bit)", file=sys.stderr)
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            elif self.quant_bits == 8:
                print("[MODEL] Готовим INT8 квантование (8-bit)", file=sys.stderr)
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=not use_cuda,
                )
            else:
                print(f"[MODEL] ⚠ Неизвестная глубина квантования {self.quant_bits}, отключаем квантование", file=sys.stderr)

        def _load_with_kwargs(extra_kwargs: Dict) -> AutoModelForCausalLM:
            load_kwargs = dict(common_kwargs)
            load_kwargs.update(extra_kwargs)
            if load_kwargs.get('device_map') is None:
                load_kwargs.pop('device_map', None)
            if quant_cfg is not None and use_cuda:
                load_kwargs['quantization_config'] = quant_cfg
            return AutoModelForCausalLM.from_pretrained(
                model_source,
                config=cfg,
                local_files_only=local_only,
                **load_kwargs,
            )

        if use_cuda:
            try:
                model = _load_with_kwargs({})
            except (RuntimeError, Exception) as e:
                error_msg = str(e)
                if any(token in error_msg.lower() for token in ("cuda", "ecc", "uncorrectable")):
                    print(f"[MODEL] ⚠ CUDA error during loading: {error_msg}", file=sys.stderr)
                    print(f"[MODEL] Retrying with simplified device_map...", file=sys.stderr)
                    try:
                        model = _load_with_kwargs({'device_map': 'cuda:0'})
                        print("[MODEL] ✓✓✓ Model loaded with simplified device_map", file=sys.stderr)
                    except Exception as e2:
                        print(f"[MODEL] ⚠ Second attempt also failed: {e2}", file=sys.stderr)
                        print(f"[MODEL] Falling back to CPU mode without quantization...", file=sys.stderr)
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        quant_cfg = None
                        common_kwargs.clear()
                        common_kwargs.update({
                            'torch_dtype': torch.float16,
                            'low_cpu_mem_usage': True,
                            'offload_folder': 'offload',
                            'max_memory': {'cpu': '48GiB'},
                        })
                        use_cuda = False
                        model = _load_with_kwargs({})
                        model = model.to('cpu')
                else:
                    raise
        else:
            if quant_cfg is not None:
                print("[MODEL] ⚠ Квантование требует CUDA. Загружаем модель на CPU без квантования.", file=sys.stderr)
            print(f"[MODEL] Loading model on CPU...", file=sys.stderr)
            quant_cfg = None
            model = _load_with_kwargs({})
            model = model.to('cpu')
        
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        model.eval()
        print(f"[MODEL] ✓✓✓ Model loaded successfully (device={next(model.parameters()).device})", file=sys.stderr)
        return model, tokenizer, gen_cfg

    # --------------------
    # RAG: инициализация и утилиты
    # --------------------
    def _init_rag(self) -> None:
        """Ленивая инициализация артефактов RAG (faiss, parquet, лемматизация)."""
        if self._rag_inited:
            return
        self._rag_inited = True
        print(f"[RAG] Starting RAG initialization...", file=sys.stderr)
        print(f"[RAG] RAG_DIR: {self.RAG_DIR.absolute()}", file=sys.stderr)
        try:
            # Импорты зависят от дополнительных пакетов — делаем внутри
            # Если библиотеки отсутствуют, приложение должно упасть, а не продолжать без RAG
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                print(f"[RAG] ✓ SentenceTransformer imported", file=sys.stderr)
            except ImportError as e:
                print(f"[RAG] ✗ Failed to import SentenceTransformer: {e}", file=sys.stderr)
                raise RuntimeError(f"RAG requires sentence-transformers library but it's not available: {e}") from e
            try:
                import faiss  # type: ignore
                print(f"[RAG] ✓ faiss imported", file=sys.stderr)
            except ImportError as e:
                print(f"[RAG] ✗ Failed to import faiss: {e}", file=sys.stderr)
                raise RuntimeError(f"RAG requires faiss library but it's not available: {e}") from e
            try:
                import pyarrow.parquet as pq  # type: ignore
                print(f"[RAG] ✓ pyarrow imported", file=sys.stderr)
            except ImportError as e:
                print(f"[RAG] ✗ Failed to import pyarrow: {e}", file=sys.stderr)
                raise RuntimeError(f"RAG requires pyarrow library but it's not available: {e}") from e

            meta_path = self.RAG_DIR / 'meta.json'
            index_path = self.RAG_DIR / 'index.faiss'
            chunks_path = self.RAG_DIR / 'chunks.parquet'
            
            print(f"[RAG] Checking files:", file=sys.stderr)
            print(f"[RAG]   RAG_DIR exists: {self.RAG_DIR.exists()}", file=sys.stderr)
            print(f"[RAG]   RAG_DIR is_dir: {self.RAG_DIR.is_dir()}", file=sys.stderr)
            if self.RAG_DIR.exists() and self.RAG_DIR.is_dir():
                print(f"[RAG]   Contents of {self.RAG_DIR}:", file=sys.stderr)
                try:
                    for item in sorted(self.RAG_DIR.iterdir()):
                        print(f"[RAG]     - {item.name} ({'dir' if item.is_dir() else 'file'})", file=sys.stderr)
                except Exception as e:
                    print(f"[RAG]     Error listing directory: {e}", file=sys.stderr)
            print(f"[RAG]   meta.json: {meta_path} (exists={meta_path.exists()})", file=sys.stderr)
            print(f"[RAG]   index.faiss: {index_path} (exists={index_path.exists()})", file=sys.stderr)
            print(f"[RAG]   chunks.parquet: {chunks_path} (exists={chunks_path.exists()})", file=sys.stderr)
            
            # Проверяем альтернативные пути на случай, если архив распаковался в другом месте
            if not (meta_path.exists() and index_path.exists() and chunks_path.exists()):
                print(f"[RAG] ✗ Missing RAG files, checking alternative locations...", file=sys.stderr)
                for alt_dir in self._rag_search_paths:
                    alt_meta = alt_dir / 'meta.json'
                    alt_index = alt_dir / 'index.faiss'
                    alt_chunks = alt_dir / 'chunks.parquet'
                    if alt_meta.exists() and alt_index.exists() and alt_chunks.exists():
                        print(f"[RAG]   Found RAG artifacts at: {alt_dir}", file=sys.stderr)
                        self.RAG_DIR = alt_dir.resolve()
                        meta_path = alt_meta
                        index_path = alt_index
                        chunks_path = alt_chunks
                        print(f"[RAG]   Updated RAG_DIR to: {self.RAG_DIR}", file=sys.stderr)
                        break
                
                if not (meta_path.exists() and index_path.exists() and chunks_path.exists()):
                    print(f"[RAG] ✗ Missing RAG files", file=sys.stderr)
                    missing = []
                    if not meta_path.exists():
                        missing.append(f"meta.json: {meta_path}")
                    if not index_path.exists():
                        missing.append(f"index.faiss: {index_path}")
                    if not chunks_path.exists():
                        missing.append(f"chunks.parquet: {chunks_path}")
                    raise FileNotFoundError(f"RAG requires all files to be present, but missing: {', '.join(missing)}")

            with open(meta_path, 'r', encoding='utf-8') as f:
                self._rag_meta = json.load(f)
            emb_model_name = self._rag_meta.get('model', 'sentence-transformers/all-MiniLM-L12-v2')
            print(f"[RAG] Loading embedding model: {emb_model_name}", file=sys.stderr)

            # Эмбеддер на CPU/GPU автоматически
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[RAG] Loading SentenceTransformer on {device}...", file=sys.stderr)
            self._retr_model = SentenceTransformer(emb_model_name, device=device)
            print(f"[RAG] ✓ SentenceTransformer loaded", file=sys.stderr)

            # Индекс FAISS
            print(f"[RAG] Loading FAISS index...", file=sys.stderr)
            self._faiss_index = faiss.read_index(str(index_path))
            print(f"[RAG] ✓ FAISS index loaded: {self._faiss_index.ntotal} vectors, dim={self._faiss_index.d}", file=sys.stderr)

            # Parquet с чанками
            print(f"[RAG] Loading Parquet file...", file=sys.stderr)
            self._pf = pq.ParquetFile(str(chunks_path))
            meta = self._pf.metadata
            rg_sizes = []
            for i in range(meta.num_row_groups):
                rg_sizes.append(meta.row_group(i).num_rows)
            self._rg_sizes = rg_sizes
            # Префиксные суммы
            self._rg_cum = []
            s = 0
            for n in rg_sizes:
                s += n
                self._rg_cum.append(s)
            print(f"[RAG] ✓ Parquet loaded: {sum(rg_sizes)} total chunks in {len(rg_sizes)} row groups", file=sys.stderr)

            # Лемматизация
            try:
                from lem_worker import lemmatize_text as _lemmatize_ru  # type: ignore
                self._lemmatize_ru = _lemmatize_ru
                print(f"[RAG] ✓ Lemmatization enabled", file=sys.stderr)
            except Exception as e:
                self._lemmatize_ru = None
                print(f"[RAG] ⚠ Lemmatization disabled: {e}", file=sys.stderr)

            self._rag_ok = True
            print(f"[RAG] ✓✓✓ RAG initialization SUCCESS (TOP_K={self.TOP_K}, THRESHOLD={self.DECISION_THRESHOLD})", file=sys.stderr)
        except (RuntimeError, FileNotFoundError) as e:
            # Критические ошибки (отсутствие библиотек или файлов) — падаем сразу
            print(f"[RAG] ✗✗✗ RAG initialization FAILED (CRITICAL): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise  # Пробрасываем исключение дальше - приложение должно упасть
        except Exception as e:
            # Неожиданные ошибки — тоже падаем
            print(f"[RAG] ✗✗✗ RAG initialization FAILED (UNEXPECTED): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"RAG initialization failed unexpectedly: {e}") from e

    _re_punct = re.compile(r"[^\w\s]+", flags=re.UNICODE)
    _re_space = re.compile(r"\s+", flags=re.UNICODE)

    def _norm(self, s: str) -> str:
        t = (s or '').lower().strip()
        t = self._re_punct.sub(' ', t)
        t = self._re_space.sub(' ', t)
        return t.strip()

    def _norm_lemma(self, s: str) -> str:
        n = self._norm(s)
        if self._lemmatize_ru is None:
            return n
        try:
            return self._lemmatize_ru(n)
        except Exception:
            return n

    def _rows_by_indices(self, indices: List[int]) -> List[tuple[str, str]]:
        # Возвращает пары (title, text) по индексам строк в Parquet
        if self._pf is None or not self._rg_cum:
            return [("", "") for _ in indices]
        from bisect import bisect_right
        out: List[tuple[str, str]] = []
        cache: dict[int, object] = {}
        for idx in indices:
            if idx is None or idx < 0:
                out.append(("", ""))
                continue
            g = bisect_right(self._rg_cum, idx)
            prev = self._rg_cum[g - 1] if g > 0 else 0
            off = idx - prev
            try:
                if g not in cache:
                    table = self._pf.read_row_group(g, columns=['title', 'text'])
                    cache[g] = table
                else:
                    table = cache[g]
                # pyarrow.Table → списки
                title_arr = table.column('title').to_pylist()
                text_arr = table.column('text').to_pylist()
                if 0 <= off < len(title_arr):
                    out.append((str(title_arr[off] or ''), str(text_arr[off] or '')))
                else:
                    out.append(("", ""))
            except Exception:
                out.append(("", ""))
        return out

    def _retrieve_context(self, query: str, k: int | None = None) -> tuple[list[float], list[int], str]:
        if not self._rag_ok or self._retr_model is None or self._faiss_index is None:
            return [], [], ""
        kk = k or self.TOP_K
        qn = self._norm_lemma(query)
        try:
            vec = self._retr_model.encode([qn], normalize_embeddings=True).astype('float32')
        except Exception:
            return [], [], ""
        try:
            D, I = self._faiss_index.search(vec, kk)
        except Exception:
            return [], [], ""
        indices = I[0].tolist() if len(I) else []
        scores = [float(x) for x in (D[0].tolist() if len(D) else [])]
        pairs = self._rows_by_indices(indices)
        per = max(200, self.MAX_CTX_CHARS // max(1, kk))
        parts: list[str] = []
        for (title, text), sc in zip(pairs, scores):
            if not text:
                continue
            t = (text or '').strip()
            piece = t[:per]
            if title:
                parts.append(f"{title}: {piece}")
            else:
                parts.append(piece)
        ctx = "\n\n".join(parts)[: self.MAX_CTX_CHARS]
        return scores, indices, ctx

    def _build_rag_prompt(self, question: str, ctx: str) -> str:
        user = (
            "Используй приведённый контекст для ответа. Если в контексте нет ответа — ответь: 'Не знаю'.\n"
            f"Контекст:\n{ctx}\n\nВопрос: {question}"
        )
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_strict_prompt(self, user_text: str) -> str:
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": STRICT_FALLBACK_PROMPT},
            {"role": "user", "content": user_text}
        ], tokenize=False, add_generation_prompt=True)

    def _is_broken(self, ans: str) -> bool:
        if not ans or not ans.strip():
            return True
        t = ans.strip()
        low = t.lower()
        words = [w for w in re.findall(r"\w+", low)]
        letters = sum(1 for c in t if c.isalpha())
        if t.endswith("?"):
            return True
        if re.search(r"(?:^|\s)не\.?$", low):
            return True
        if len(t) <= 2 or letters <= 2:
            return True
        if len(words) <= 1 and not re.fullmatch(r"\d+[\.,]?\d*", t):
            return True
        if low in {"энергии?", "тяжести?", "?", "ла?", "формула воды?"}:
            return True
        return False

    def _generate_raw(self, question: str, *, max_new_tokens: int | None = None, strict: bool = False) -> str:
        prompt = self._build_strict_prompt(question) if strict else self._build_prompt(question)
        data = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        data.pop('token_type_ids', None)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        out_ids = self.model.generate(
            **data,
            generation_config=self.generation_config,
            max_new_tokens=(max_new_tokens or self.max_new_tokens),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )[0]

        new_ids = out_ids[len(data['input_ids'][0]):]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Чистка эхо/маркеров assistant
        t = raw
        if isinstance(question, str) and t.lower().startswith(question.strip().lower()):
            t = t[len(question):].lstrip(" \\–—-:.,!?\"'\n\t")
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        while lines and lines[0].lower().startswith('assistant'):
            lines.pop(0)
        t = lines[0] if lines else ""
        while t.lower().rstrip().endswith('assistant:') or t.lower().rstrip().endswith('assistant'):
            tt = t.rstrip(); ll = tt.lower()
            if ll.endswith('assistant:'):
                t = tt[:-len('assistant:')]
            elif ll.endswith('assistant'):
                t = tt[:-len('assistant')]
            else:
                break
            t = t.rstrip()

        return t

    @torch.inference_mode()
    def generate(self, question: str) -> str:
        """Генерирует короткий ответ на один вопрос."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Быстрый путь: детерминированная математика
        m = self._maybe_answer_math(question)
        if m is not None:
            return m

        # Пытаемся использовать RAG
        try:
            self._init_rag()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[RAG] RAG initialization failed: {e}", file=sys.stderr)
            print(f"[RAG] Continuing without RAG (fallback mode)", file=sys.stderr)
            self._rag_ok = False
        if self._rag_ok:
            scores, _, ctx = self._retrieve_context(question, k=self.TOP_K)
            best = max(scores) if scores else 0.0
            print(f"[RAG] Query: '{question[:60]}{'...' if len(question) > 60 else ''}'", file=sys.stderr)
            print(f"[RAG] Scores: {scores}, best={best:.4f}, threshold={self.DECISION_THRESHOLD}, ctx_len={len(ctx)}", file=sys.stderr)
            if not scores or best < self.DECISION_THRESHOLD or not ctx.strip():
                reason = []
                if not scores:
                    reason.append("no_scores")
                if scores and best < self.DECISION_THRESHOLD:
                    reason.append(f"low_score_{best:.4f}<{self.DECISION_THRESHOLD}")
                if not ctx.strip():
                    reason.append("empty_context")
                print(f"[RAG] ✗ RAG rejected: {', '.join(reason)} -> falling back to regular generation", file=sys.stderr)
                return 'Не знаю'
            
            print(f"[RAG] ✓ RAG accepted (score={best:.4f}), generating with context", file=sys.stderr)

            # Генерация с контекстом
            prompt = self._build_rag_prompt(question, ctx)
            data = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            data = {k: v.to(self.model.device) for k, v in data.items()}
            data.pop('token_type_ids', None)

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

            out_ids = self.model.generate(
                **data,
                generation_config=self.generation_config,
                max_new_tokens=64,  # Увеличиваем для RAG, как в ноутбуке
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )[0]
            new_ids = out_ids[len(data['input_ids'][0]):]
            raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            text = self._postprocess(raw, question)
            if self._is_broken(text):
                return 'Не знаю'
            return text

        # Фолбэк без RAG — старый путь
        if not self._rag_inited:
            print(f"[RAG] RAG not initialized, using regular generation", file=sys.stderr)
        elif not self._rag_ok:
            print(f"[RAG] RAG initialization failed, using regular generation", file=sys.stderr)
        text = self._generate_raw(question, max_new_tokens=self.max_new_tokens, strict=False)
        text = self._postprocess(text, question)
        if self._is_broken(text):
            t2 = self._generate_raw(question, max_new_tokens=self.max_new_tokens, strict=True)
            t2 = self._postprocess(t2, question)
            if self._is_broken(t2):
                return 'Не знаю'
            return t2
        return text

    def _postprocess(self, text: str, question: str | None = None) -> str:
        t = (text or '').strip()
        if not t:
            return 'Не знаю'

        # Удаляем отражение вопроса в начале
        if isinstance(question, str):
            q = question.strip().lower()
            if q and t.lower().startswith(q):
                t = t[len(question):].lstrip(" \t\n\r-–—:.,!?\"'")

        # Сохраняем только первое законченное предложение
        first_end = len(t)
        for sep in '.!?':
            pos = t.find(sep)
            if pos != -1:
                first_end = min(first_end, pos + 1)
        if first_end != len(t):
            t = t[:first_end].strip()

        # Сводим частичные отказы к "Не знаю"
        low = t.lower()
        refusal_triggers = [
            'не знаю', 'не могу ответить', 'затрудняюсь', 'нет данных',
            "i don't know", 'i do not know', 'unknown', 'no idea', 'cannot answer'
        ]
        if any(p in low for p in refusal_triggers):
            return 'Не знаю'

        # Явно сломанные/обрывочные ответы
        letters = sum(1 for c in t if c.isalpha())
        if t.endswith('?') and low not in ('да?', 'нет?'):
            return 'Не знаю'
        if re.search(r"(?:^|\s)не\.?$", low):
            return 'Не знаю'
        if letters <= 2 and low not in ('да', 'нет', 'да.', 'нет.'):
            return 'Не знаю'

        # Режим перевода
        if isinstance(question, str) and 'переведи на английский' in question.lower():
            t_clean = t.replace('"', '').replace("'", '').replace('«', '').replace('»', '').strip()
            m = re.findall(r"[A-Za-z][A-Za-z .,!?'\-]*", t_clean)
            if m:
                eng = m[0].strip().strip(' .')
                return eng if eng else 'Не знаю'
            return 'Не знаю'

        # Жёсткий лимит по количеству слов (чуть больше для полноты)
        words = t.split()
        if len(words) > 14:
            t = ' '.join(words[:14]).rstrip(',;:')
            if not t.endswith('.'):  # добавим точку для завершенности
                t += '.'

        # Убираем пустой результат
        return t or 'Не знаю'

    # --------------------
    # Простая математика
    # --------------------
    def _maybe_answer_math(self, question: str | None) -> str | None:
        if not isinstance(question, str):
            return None
        q = question.strip().lower()

        def _to_float(s: str) -> float:
            return float(s.replace(',', '.'))

        def _fmt_number(val: float) -> str:
            if abs(val - int(round(val))) < 1e-9:
                return f"{int(round(val))}."
            s = f"{val:.10g}"
            if '.' in s:
                # Используем запятую в качестве десятичного разделителя
                s = s.replace('.', ',')
            return f"{s}."

        # Сумма/разность/умножение/деление (словами)
        m = re.search(r"сколько\s+будет\s+(\d+)\s*(плюс|минус|умножить\s+на|разделить\s+на|\/|:|\*|x|×|\+|\-)\s*(\d+)", q)
        if m:
            a = float(m.group(1)); op = m.group(2); b = float(m.group(3))
            if op in ('плюс', '+'):
                return _fmt_number(a + b)
            if op in ('минус', '-'):
                return _fmt_number(a - b)
            if op in ('умножить на', '*', 'x', '×'):
                return _fmt_number(a * b)
            if op in ('разделить на', '/', ':'):
                if b == 0:
                    return 'Не знаю'
                return _fmt_number(a / b)

        # Выражения вида "144 разделить на 12"
        m = re.search(r"(\d+)\s*(разделить\s+на|\/|:)\s*(\d+)", q)
        if m:
            a = float(m.group(1)); b = float(m.group(3))
            if b == 0:
                return 'Не знаю'
            return _fmt_number(a / b)

        m = re.search(r"(\d+)\s*(умножить\s+на|\*|x|×)\s*(\d+)", q)
        if m:
            a = float(m.group(1)); b = float(m.group(3))
            return _fmt_number(a * b)

        m = re.search(r"сумма\s+чисел\s+(\d+)\s*и\s*(\d+)", q)
        if m:
            a = float(m.group(1)); b = float(m.group(2))
            return _fmt_number(a + b)

        m = re.search(r"сколько\s+будет\s+(\d+)\s*(минус)\s*(\d+)", q)
        if m:
            a = float(m.group(1)); b = float(m.group(3))
            return _fmt_number(a - b)

        # Проценты
        m = re.search(r"(\d+)\s*процент(?:а|ов)?\s*от\s*(\d+)", q)
        if m:
            p = float(m.group(1)); base = float(m.group(2))
            return _fmt_number(base * p / 100.0)

        # Среднее арифметическое
        m = re.search(r"среднее\s+арифметическое\s+чисел\s+([\d,\s]+)", q)
        if m:
            nums_str = m.group(1)
            nums = [int(s) for s in re.findall(r"\d+", nums_str)]
            if nums:
                return _fmt_number(sum(nums) / len(nums))

        # Корни
        m = re.search(r"квадратн[а-яё]*\s+корень\s+из\s+(\d+)", q)
        if m:
            import math
            n = int(m.group(1));
            r = math.isqrt(n)
            if r * r == n:
                return _fmt_number(r)
            return 'Не знаю'

        # Степени: "2 в пятой степени"
        m = re.search(r"(\d+)\s+в\s+([а-яё]+)\s+степени", q)
        if m:
            base = int(m.group(1)); word = m.group(2)
            ord_map = {
                'второй': 2, 'третьей': 3, 'третьей': 3, 'четвёртой': 4, 'четвертой': 4,
                'пятой': 5, 'шестой': 6, 'седьмой': 7, 'восьмой': 8, 'девятой': 9, 'десятой': 10
            }
            exp = ord_map.get(word, None)
            if exp is not None:
                return _fmt_number(float(base ** exp))

        # Площадь круга: радиуса R (π=...)
        m = re.search(r"площадь\s+круга\s+радиуса\s+(\d+(?:[\.,]\d+)?)", q)
        if m:
            r = _to_float(m.group(1))
            mpi = re.search(r"π\s*=\s*(\d+(?:[\.,]\d+)?)", q)
            pi = _to_float(mpi.group(1)) if mpi else 3.1415926535
            val = pi * (r ** 2)
            return _fmt_number(val)

        return None

    def generate_batch(self, questions: List[str], batch_size: int = 8) -> List[str]:
        """
        Обрабатывает список вопросов батчами.

        Параметры:
            questions: список строк.
            batch_size: размер батча для генерации.
        """
        outputs: List[str] = []
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Лениво инициализируем RAG
        try:
            self._init_rag()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[RAG] RAG initialization failed: {e}", file=sys.stderr)
            print(f"[RAG] Continuing without RAG (fallback mode)", file=sys.stderr)
            self._rag_ok = False
        use_rag = self._rag_ok
        for start in range(0, len(questions), batch_size):
            chunk = questions[start:start + batch_size]

            # Быстрый путь: математика
            quick_answers: dict[int, str] = {}
            gen_candidates: list[str] = []
            idx_map: list[int] = []
            rag_prompts: list[str] = []
            rag_idx_map: list[int] = []

            for i, q in enumerate(chunk):
                m = self._maybe_answer_math(q)
                if m is not None:
                    quick_answers[i] = m
                    continue

                if use_rag:
                    scores, _, ctx = self._retrieve_context(q, k=self.TOP_K)
                    best = max(scores) if scores else 0.0
                    if not scores or best < self.DECISION_THRESHOLD or not ctx.strip():
                        if start == 0 and i == 0:  # Логируем только для первого вопроса в батче
                            reason = []
                            if not scores:
                                reason.append("no_scores")
                            if scores and best < self.DECISION_THRESHOLD:
                                reason.append(f"low_score_{best:.4f}<{self.DECISION_THRESHOLD}")
                            if not ctx.strip():
                                reason.append("empty_context")
                            print(f"[RAG] ✗ Batch query rejected: {', '.join(reason)}", file=sys.stderr)
                        quick_answers[i] = 'Не знаю'
                    else:
                        if start == 0 and i == 0:
                            print(f"[RAG] ✓ Batch query accepted (score={best:.4f})", file=sys.stderr)
                        rag_prompts.append(self._build_rag_prompt(q, ctx))
                        rag_idx_map.append(i)
                else:
                    gen_candidates.append(q)
                    idx_map.append(i)

            decoded_by_idx: dict[int, str] = {}

            # Генерация RAG-промптов
            if rag_prompts:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                enc = self.tokenizer(
                    rag_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                attention_mask = enc.get('attention_mask')
                if attention_mask is not None:
                    input_lengths = attention_mask.sum(dim=1)
                else:
                    input_lengths = torch.tensor([enc['input_ids'].shape[1]] * enc['input_ids'].shape[0])
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                enc.pop('token_type_ids', None)
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                out = self.model.generate(
                    **enc,
                    generation_config=self.generation_config,
                    max_new_tokens=64,  # Увеличиваем для RAG, как в ноутбуке
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                for j in range(out.shape[0]):
                    new_ids = out[j, int(input_lengths[j]):]
                    raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    t = self._postprocess(raw, None)
                    if self._is_broken(t):
                        t = 'Не знаю'
                    decoded_by_idx[rag_idx_map[j]] = t

            # Генерация без RAG (фолбэк)
            if gen_candidates:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                prompts = [self._build_prompt(q) for q in gen_candidates]
            
                enc = self.tokenizer(
                    prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                attention_mask = enc.get('attention_mask')
                if attention_mask is not None:
                    input_lengths = attention_mask.sum(dim=1)
                else:
                    input_lengths = torch.tensor([enc['input_ids'].shape[1]] * enc['input_ids'].shape[0])
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                enc.pop('token_type_ids', None)
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                out = self.model.generate(
                    **enc,
                    generation_config=self.generation_config,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

                for j in range(out.shape[0]):
                    new_ids = out[j, int(input_lengths[j]):]
                    raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                    t = raw.strip()
                    q = gen_candidates[j]
                    if isinstance(q, str) and t.lower().startswith(q.strip().lower()):
                        t = t[len(q):].lstrip(" \\–—-:.,!?\"'\n\t")
                    lines = [l.strip() for l in t.splitlines() if l.strip()]
                    while lines and lines[0].lower().startswith('assistant'):
                        lines.pop(0)
                    t = lines[0] if lines else ""
                    while t.lower().rstrip().endswith('assistant:') or t.lower().rstrip().endswith('assistant'):
                        tt = t.rstrip(); ll = tt.lower()
                        if ll.endswith('assistant:'):
                            t = tt[:-len('assistant:')]
                        elif ll.endswith('assistant'):
                            t = tt[:-len('assistant')]
                        else:
                            break
                        t = t.rstrip()
                    t = self._postprocess(t, q)
                    if self._is_broken(t):
                        # точечный строгий фолбэк
                        t2 = self._generate_raw(q, max_new_tokens=self.max_new_tokens, strict=True)
                        t2 = self._postprocess(t2, q)
                        t = 'Не знаю' if self._is_broken(t2) else t2
                    decoded_by_idx[idx_map[j]] = t

            # Сбор результата в исходном порядке
            for i, q in enumerate(chunk):
                if i in quick_answers:
                    outputs.append(quick_answers[i])
                else:
                    outputs.append(decoded_by_idx.get(i, 'Не знаю'))

        return outputs

    @torch.inference_mode()
    def vibecode(self, question: str) -> str:
        """
        Режим "Вайбкодинг": генерирует подробный и точный код.
        Игнорирует ограничения на длину и строгие RAG-правила.
        """
        # Используем специальный промпт
        messages = [
            {"role": "system", "content": VIBECODE_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        data = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        data.pop('token_type_ids', None)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        # Генерируем с настройками для кодинга:
        # - max_new_tokens=2048 (чтобы влез длинный код)
        # - do_sample=False (greedy decoding для максимальной воспроизводимости и точности)
        # - repetition_penalty=1.1 (чтобы не зацикливался)
        out_ids = self.model.generate(
            **data,
            generation_config=self.generation_config,
            max_new_tokens=2048, 
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )[0]

        new_ids = out_ids[len(data['input_ids'][0]):]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        # Убираем мусор из начала ответа, если модель решила повторить роль
        if text.lower().startswith("assistant"):
             text = text[9:].strip(": \n")
             
        return text


    @torch.inference_mode()
    def vibecode_chat(self, messages: List[Dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        """
        Чат-режим для кодинга. Принимает историю сообщений и генерирует ответ.
        Всегда добавляет системный промпт для строгого кодинга.
        """
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        msgs: List[Dict[str, str]] = []
        # Принудительно ставим системный промпт в начало
        msgs.append({"role": "system", "content": VIBECODE_SYSTEM_PROMPT})
        # Добавляем остальную историю (user/assistant/system — как есть)
        for m in messages or []:
            role = m.get("role", "user")
            content = str(m.get("content", "") or "")
            if not content.strip():
                continue
            msgs.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        enc.pop("token_type_ids", None)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        out_ids = self.model.generate(
            **enc,
            generation_config=self.generation_config,
            max_new_tokens=(max_new_tokens or 2048),
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )[0]
        new_ids = out_ids[len(enc["input_ids"][0]):]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if text.lower().startswith("assistant"):
            text = text[9:].strip(": \n")
        return text

    @torch.inference_mode()
    def vibecode_with_rag_fallback(self, question: str, *, max_new_tokens: int | None = None) -> str:
        """
        Генерирует код в режиме vibecode. Если ответ пустой/сломанный/«Не знаю»,
        делает фолбэк через стандартную generate() (которая использует RAG).
        """
        primary = self.vibecode(question if isinstance(question, str) else str(question))
        if not primary or self._is_broken(primary) or primary.strip().lower() == "не знаю":
            alt = self.generate(question)
            return alt if alt and alt.strip().lower() != "не знаю" else primary or "Не знаю"
        return primary

