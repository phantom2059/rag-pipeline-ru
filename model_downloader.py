import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import generate_config


def download_model(
    model_name: str = "Gaivoronsky/Mistral-7B-Saiga",
    *,
    model_dir: str | Path = "./model",
    quantize: bool = True,
    bits: int = 8,
    config_path: str | Path = "config.json",
) -> Path:
    """
    Скачивает модель из HuggingFace и сохраняет её в локальную папку.

    Параметры:
        model_name: Имя модели или ссылка вида https://huggingface.co/{repo}.
        model_dir: Папка, куда сохранить веса и токенизатор.
        quantize: Применять ли квантование при загрузке (4 или 8 бит).
        bits: Глубина квантования. Допустимо только 4 или 8.
        config_path: Куда сохранять автоматически сгенерированный config.json.

    Возвращает:
        Путь к локальной папке модели.
    """
    resolved_model_dir = Path(model_dir).expanduser().resolve()
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    repo_id = _extract_repo_id(model_name)
    print(f"[MODEL_DOWNLOADER] Модель: {repo_id}")
    print(f"[MODEL_DOWNLOADER] Целевая папка: {resolved_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Проверка на пре-квантованные модели (AWQ, GPTQ)
    is_prequantized = any(x in repo_id.upper() for x in ["AWQ", "GPTQ"])
    if is_prequantized:
        print(f"[MODEL_DOWNLOADER] ⚠ Обнаружена пре-квантованная модель ({repo_id}).")
        print(f"[MODEL_DOWNLOADER] ⚠ Принудительно отключаем bitsandbytes-квантование и используем snapshot_download.")
        quantize = False
        
        from huggingface_hub import snapshot_download
        print("[MODEL_DOWNLOADER] Скачивание файлов модели (snapshot_download)...")
        snapshot_download(repo_id=repo_id, local_dir=resolved_model_dir, local_dir_use_symlinks=False)
        print("[MODEL_DOWNLOADER] ✓ Файлы скачаны.")
        
        # Сохраняем токенизатор поверх (чтобы был config.json токенизатора)
        tokenizer.save_pretrained(resolved_model_dir)
    else:
        quant_config = _build_quant_config(quantize=quantize, bits=bits)
        device_map = _select_device_map()
    
        print("[MODEL_DOWNLOADER] Загрузка весов...")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=_preferred_dtype(),
        )
        if quantize and tokenizer.pad_token_id is not None:
            model.resize_token_embeddings(len(tokenizer))
    
        print("[MODEL_DOWNLOADER] Сохранение модели...")
        model.save_pretrained(resolved_model_dir)
        tokenizer.save_pretrained(resolved_model_dir)

    print("[MODEL_DOWNLOADER] Генерация config.json...")
    generate_config(
        model_name=repo_id,
        model_dir=resolved_model_dir,
        quantize=quantize,
        bits=bits,
        output_path=config_path,
    )
    print(f"[MODEL_DOWNLOADER] Готово. Конфиг: {Path(config_path).resolve()}")
    return resolved_model_dir


def _extract_repo_id(value: str) -> str:
    """Извлекает имя репозитория HuggingFace из ссылки или строки."""
    if value.startswith("http"):
        match = re.search(r"huggingface\.co/([^/?#]+/[^/?#]+)", value)
        if not match:
            raise ValueError(f"Не удалось извлечь имя репозитория из ссылки: {value}")
        return match.group(1)
    return value


def _build_quant_config(*, quantize: bool, bits: int) -> Optional[BitsAndBytesConfig]:
    """Создаёт конфигурацию квантования bitsandbytes или возвращает None."""
    if not quantize:
        return None
    if bits not in (4, 8):
        raise ValueError("bits может быть только 4 или 8")

    dtype = _preferred_dtype()
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=not torch.cuda.is_available(),
    )


def _select_device_map():
    """Определяет карту устройств для загрузки модели (GPU при наличии)."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1024**3
            print(f"[MODEL_DOWNLOADER] CUDA доступна (GPU {props.name}, {total_gb:.1f} GB)")
            return {"": 0}
        except Exception as exc:
            print(f"[MODEL_DOWNLOADER] Предупреждение CUDA: {exc}. Используем CPU.")
    return None


def _preferred_dtype():
    """Возвращает предпочтительный dtype (bf16 при поддержке GPU)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

