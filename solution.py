import argparse
import sys
from pathlib import Path

from factual_model import FactualModel
from utils import load_questions, save_results


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Быстрый запуск RAG-пайплайна: загрузка вопросов, инференс, сохранение ответов."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input.json"),
        help="Файл с вопросами (JSON/CSV/TSV/XLSX/TXT).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.json"),
        help="Файл для сохранения ответов.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "json_pairs", "csv", "tsv"),
        default="json",
        help="Формат сохранения ответов.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Размер батча для генерации.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Путь к config.json (если отсутствует — будут использованы значения по умолчанию).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"[MAIN] Loading questions from {args.input}...", file=sys.stderr)
    questions = load_questions(args.input)
    print(f"[MAIN] ✓ Loaded {len(questions)} questions", file=sys.stderr)

    print("[MAIN] Initializing FactualModel...", file=sys.stderr)
    model = FactualModel(config_path=args.config)
    print("[MAIN] ✓ Model initialized", file=sys.stderr)

    print(f"[MAIN] Generating answers (batch_size={args.batch_size})...", file=sys.stderr)
    answers = model.generate_batch(questions, batch_size=args.batch_size)
    print(f"[MAIN] ✓ Generated {len(answers)} answers", file=sys.stderr)

    print(f"[MAIN] Saving results to {args.output} ({args.format})...", file=sys.stderr)
    save_results(questions, answers, args.output, fmt=args.format)
    print("[MAIN] ✓✓✓ Solution completed successfully", file=sys.stderr)


if __name__ == "__main__":
    main()
