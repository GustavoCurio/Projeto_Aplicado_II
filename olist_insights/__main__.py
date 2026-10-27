"""Automação reexecutável: relatório de indicadores e avaliação do classificador."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .analytics import build_evidence, load_orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/portfolio"))
    parser.add_argument("--train", action="store_true", help="Também treina e avalia o classificador")
    parser.add_argument("--model-path", type=Path, default=Path("models/reviews.joblib"))
    args = parser.parse_args()
    started = perf_counter()
    frame = load_orders(args.data_dir)
    evidence = build_evidence(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"evidencias": evidence}
    if args.train:
        from .sentiment import train_model
        reviews = pd.read_csv(args.data_dir / "olist_order_reviews_dataset.csv", usecols=["review_score", "review_comment_message"])
        result["classificador"] = train_model(reviews, args.model_path)["metrics"]
    result["tempo_execucao_segundos"] = round(perf_counter() - started, 3)
    result["impacto"] = "Tempo de execução medido nesta máquina. Economia de horas de trabalho não foi medida."
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Relatório salvo em {output}; {len(frame)} pedidos; {result['tempo_execucao_segundos']} s.")


if __name__ == "__main__":
    main()
