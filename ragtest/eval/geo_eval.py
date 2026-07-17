#!/usr/bin/env python3
"""Evaluate Top-1, Top-5, and Administrative Chain Accuracy.

----------------------
    sample_id
        Unique QA identifier. 
    reference_candidate_ids
        One correct candidate ID or a list of acceptable IDs.
    retrieved_candidate_ids
        Ranked candidate IDs returned by the retriever. Rank order must be
        preserved. 
    reference_admin_chain
    predicted_admin_chain

Recommended cell formats
------------------------
    reference_candidate_ids:  POI_001
    retrieved_candidate_ids:  ["POI_003", "POI_001", "POI_009"]
    reference_admin_chain:     ["PROV_A", "CITY_A", "DIST_A"]
    predicted_admin_chain:     PROV_A > CITY_A > DIST_A

Candidate IDs should be stable unique identifiers, not fuzzy place-name text.
Administrative chains should already be parsed from the system output. This
script intentionally avoids LLM-based extraction from free-form answers because
that would introduce another source of evaluation uncertainty.

Outputs
-------
    geo_row_scores.csv
    geo_summary.csv
    geo_bootstrap_ci.csv
    geo_run_metadata.json

"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("geo_eval")

GEO_METRICS = (
    "top_1_accuracy",
    "top_5_accuracy",
    "administrative_chain_accuracy",
)

RAGAS_METRIC_ALIASES = {
    "context_recall": {"context_recall", "llm_context_recall"},
    "faithfulness": {"faithfulness"},
    "answer_relevance": {
        "answer_relevance",
        "answer_relevancy",
        "response_relevance",
        "response_relevancy",
    },
    "contextual_precision": {
        "contextual_precision",
        "context_precision",
        "llm_context_precision_with_reference",
    },
}

COLUMN_ALIASES = {
    "sample_id": {
        "sample_id",
        "id",
        "qa_id",
        "question_id",
        "no",
        "no.",
        "序号",
    },
    "reference_candidate_ids": {
        "reference_candidate_ids",
        "reference_candidate_id",
        "correct_candidate_ids",
        "correct_candidate_id",
        "gold_candidate_ids",
        "gold_candidate_id",
        "ground_truth_candidate_ids",
        "ground_truth_candidate_id",
        "reference_id",
        "正确候选id",
        "真实候选id",
    },
    "retrieved_candidate_ids": {
        "retrieved_candidate_ids",
        "retrieved_candidate_id",
        "ranked_candidate_ids",
        "candidate_ids",
        "topk_candidate_ids",
        "retrieval_candidate_ids",
        "检索候选id",
        "候选id列表",
    },
    "reference_admin_chain": {
        "reference_admin_chain",
        "gold_admin_chain",
        "ground_truth_admin_chain",
        "reference_chain",
        "administrative_chain_reference",
        "真实行政链",
        "参考行政链",
    },
    "predicted_admin_chain": {
        "predicted_admin_chain",
        "prediction_admin_chain",
        "generated_admin_chain",
        "response_admin_chain",
        "predicted_chain",
        "administrative_chain_prediction",
        "预测行政链",
        "生成行政链",
    },
}

ADMIN_LEVEL_ORDER = (
    "country",
    "province",
    "city",
    "prefecture",
    "district",
    "county",
    "township",
    "town",
    "subdistrict",
    "street",
    "community",
    "village",
)

MISSING_STRINGS = {"", "nan", "none", "null", "unknown", "unavailable", "[redacted]"}


@dataclass(frozen=True)
class GeoRecord:
    sample_id: str
    reference_candidate_ids: list[str]
    retrieved_candidate_ids: list[str]
    reference_admin_chain: list[str]
    predicted_admin_chain: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate deterministic geospatial retrieval metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Evaluation data file.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("geo_results"), help="Output directory."
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name; default is the first sheet.")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help=(
            "Treat the file as headerless columns A-D: reference IDs, retrieved IDs, "
            "reference chain, predicted chain."
        ),
    )
    parser.add_argument("--id-col", default=None, help="Optional sample-ID column name.")
    parser.add_argument("--reference-id-col", default=None, help="Gold candidate-ID column.")
    parser.add_argument("--retrieved-ids-col", default=None, help="Ranked candidate-ID column.")
    parser.add_argument("--reference-chain-col", default=None, help="Gold administrative-chain column.")
    parser.add_argument("--predicted-chain-col", default=None, help="Predicted administrative-chain column.")
    parser.add_argument(
        "--list-separator",
        default="|||",
        help="Separator for multiple candidate IDs in one cell; default: |||.",
    )
    parser.add_argument(
        "--chain-separator",
        default=">",
        help="Coarse-to-fine administrative-chain separator; default: >.",
    )
    parser.add_argument(
        "--admin-match-mode",
        choices=("strict", "reference_levels"),
        default="strict",
        help=(
            "strict requires identical chains; reference_levels requires all gold levels "
            "to match and ignores extra predicted finer levels."
        ),
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive ID and administrative-name comparison.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate the first N valid rows.")
    parser.add_argument(
        "--ragas-scores",
        type=Path,
        default=None,
        help="Optional ragas_row_scores.csv to merge by sample_id.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="QA-level bootstrap iterations; use 0 to disable.",
    )
    parser.add_argument(
        "--confidence-level", type=float, default=0.95, help="Bootstrap confidence level."
    )
    parser.add_argument(
        "--random-seed", type=int, default=20260608, help="Deterministic bootstrap seed."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations cannot be negative.")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be between 0 and 1.")
    if not args.list_separator:
        parser.error("--list-separator cannot be empty.")
    if not args.chain_separator:
        parser.error("--chain-separator cannot be empty.")
    return args


def import_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install them with: pip install -U pandas numpy openpyxl"
        ) from exc
    return pd, np


def normalize_column_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower().lstrip("\ufeff")
    text = re.sub(r"[\s\-/\\()\[\]:]+", "_", text)
    return text.strip("_")


def normalize_identifier(value: Any, case_sensitive: bool) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_STRINGS:
        return ""
    return text if case_sensitive else text.casefold()


def is_missing(value: Any, pd: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
    except (TypeError, ValueError):
        pass
    return False


def read_table(path: Path, sheet: str | None, no_header: bool) -> Any:
    pd, _ = import_dependencies()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    header = None if no_header else 0
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, header=header, dtype=object)
    if suffix == ".csv":
        return pd.read_csv(path, header=header, dtype=object, encoding="utf-8-sig")
    if suffix == ".jsonl":
        if no_header:
            raise ValueError("--no-header is not applicable to JSONL input.")
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        if no_header:
            raise ValueError("--no-header is not applicable to JSON input.")
        return pd.read_json(path, dtype=False)
    raise ValueError("Supported input formats are .xlsx, .xls, .csv, .json, and .jsonl.")


def resolve_column(
    columns: Iterable[Any], canonical: str, explicit: str | None, required: bool = True
) -> Any | None:
    original = list(columns)
    normalized = {normalize_column_name(column): column for column in original}
    if explicit is not None:
        if explicit in original:
            return explicit
        normalized_explicit = normalize_column_name(explicit)
        if normalized_explicit in normalized:
            return normalized[normalized_explicit]
        raise ValueError(f"Column {explicit!r} not found. Available columns: {original}")

    aliases = {normalize_column_name(alias) for alias in COLUMN_ALIASES[canonical]}
    matches = [original_name for name, original_name in normalized.items() if name in aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple columns match {canonical!r}: {matches}. Pass an explicit column option."
        )
    if required:
        raise ValueError(
            f"Could not identify {canonical!r}. Available columns: {original}. "
            "Rename the column or pass an explicit column option."
        )
    return None


def parse_serialized(value: Any) -> Any:
    """Return a parsed JSON/Python list or dict when one is present."""
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text or text[0] not in "[{(" or text[-1] not in "]})":
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value


def parse_identifier_list(
    value: Any, separator: str, case_sensitive: bool, pd: Any
) -> list[str]:
    if is_missing(value, pd):
        return []
    parsed = parse_serialized(value)
    if isinstance(parsed, Mapping):
        values = list(parsed.values())
    elif isinstance(parsed, (list, tuple, set)):
        values = list(parsed)
    else:
        text = str(parsed).strip()
        values = text.split(separator) if separator in text else [text]
    normalized = [normalize_identifier(item, case_sensitive) for item in values]
    return [item for item in normalized if item]


def canonical_admin_key(value: Any) -> str:
    key = normalize_column_name(value)
    aliases = {
        "state": "province",
        "province_state": "province",
        "prefecture_city": "city",
        "county_district": "district",
        "district_county": "district",
        "township_subdistrict": "township",
        "subdistrict_township": "township",
    }
    return aliases.get(key, key)


def parse_admin_chain(
    value: Any, separator: str, case_sensitive: bool, pd: Any
) -> list[str]:
    if is_missing(value, pd):
        return []
    parsed = parse_serialized(value)
    if isinstance(parsed, Mapping):
        normalized_map = {
            canonical_admin_key(key): normalize_identifier(item, case_sensitive)
            for key, item in parsed.items()
        }
        ordered = [normalized_map[level] for level in ADMIN_LEVEL_ORDER if normalized_map.get(level)]
        known = set(ADMIN_LEVEL_ORDER)
        ordered.extend(value for key, value in normalized_map.items() if key not in known and value)
        return ordered
    if isinstance(parsed, (list, tuple)):
        values = list(parsed)
    else:
        text = str(parsed).strip()
        if separator in text:
            values = text.split(separator)
        elif "|||" in text:
            values = text.split("|||")
        else:
            values = [text]
    normalized = [normalize_identifier(item, case_sensitive) for item in values]
    return [item for item in normalized if item]


def load_records(args: argparse.Namespace) -> list[GeoRecord]:
    pd, _ = import_dependencies()
    df = read_table(args.input, args.sheet, args.no_header)
    if df.empty:
        raise ValueError("The input table contains no rows.")

    if args.no_header:
        if len(df.columns) < 4:
            raise ValueError("A headerless input must contain at least four columns (A-D).")
        reference_id_col, retrieved_ids_col, reference_chain_col, predicted_chain_col = list(
            df.columns[:4]
        )
        id_col = None
        LOGGER.warning(
            "Headerless mode: A=reference IDs, B=ranked retrieved IDs, "
            "C=reference chain, D=predicted chain."
        )
    else:
        id_col = resolve_column(df.columns, "sample_id", args.id_col, required=False)
        reference_id_col = resolve_column(
            df.columns, "reference_candidate_ids", args.reference_id_col
        )
        retrieved_ids_col = resolve_column(
            df.columns, "retrieved_candidate_ids", args.retrieved_ids_col
        )
        reference_chain_col = resolve_column(
            df.columns, "reference_admin_chain", args.reference_chain_col
        )
        predicted_chain_col = resolve_column(
            df.columns, "predicted_admin_chain", args.predicted_chain_col
        )

    records: list[GeoRecord] = []
    invalid: list[str] = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        references = parse_identifier_list(
            row[reference_id_col], args.list_separator, args.case_sensitive, pd
        )
        retrieved = parse_identifier_list(
            row[retrieved_ids_col], args.list_separator, args.case_sensitive, pd
        )
        reference_chain = parse_admin_chain(
            row[reference_chain_col], args.chain_separator, args.case_sensitive, pd
        )
        predicted_chain = parse_admin_chain(
            row[predicted_chain_col], args.chain_separator, args.case_sensitive, pd
        )

        if not any((references, retrieved, reference_chain, predicted_chain)):
            continue
        missing = []
        if not references:
            missing.append("reference_candidate_ids")
        if not retrieved:
            missing.append("retrieved_candidate_ids")
        if not reference_chain:
            missing.append("reference_admin_chain")
        if not predicted_chain:
            missing.append("predicted_admin_chain")
        if missing:
            invalid.append(f"row {position}: missing {', '.join(missing)}")
            continue

        sample_id = ""
        if id_col is not None and not is_missing(row[id_col], pd):
            sample_id = str(row[id_col]).strip()
        if not sample_id:
            sample_id = f"QA_{len(records) + 1:04d}"
        records.append(
            GeoRecord(
                sample_id=sample_id,
                reference_candidate_ids=references,
                retrieved_candidate_ids=retrieved,
                reference_admin_chain=reference_chain,
                predicted_admin_chain=predicted_chain,
            )
        )

    if invalid:
        preview = "\n".join(invalid[:10])
        extra = f"\n... and {len(invalid) - 10} more" if len(invalid) > 10 else ""
        raise ValueError(f"Input validation failed:\n{preview}{extra}")
    if not records:
        raise ValueError("No valid geospatial evaluation rows were found.")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("sample_id values must be unique.")
    if args.limit is not None:
        records = records[: args.limit]
    return records


def first_correct_rank(retrieved: Sequence[str], references: Sequence[str]) -> int | None:
    correct = set(references)
    for rank, candidate_id in enumerate(retrieved, start=1):
        if candidate_id in correct:
            return rank
    return None


def administrative_chain_score(
    reference: Sequence[str], predicted: Sequence[str], mode: str
) -> tuple[int, int, float]:
    matched_levels = sum(
        1 for gold, prediction in zip(reference, predicted) if gold == prediction
    )
    level_accuracy = matched_levels / len(reference)
    if mode == "strict":
        exact = int(list(reference) == list(predicted))
    elif mode == "reference_levels":
        exact = int(len(predicted) >= len(reference) and list(predicted[: len(reference)]) == list(reference))
    else:
        raise ValueError(f"Unsupported administrative match mode: {mode}")
    return exact, matched_levels, level_accuracy


def evaluate_records(records: Sequence[GeoRecord], args: argparse.Namespace) -> Any:
    pd, _ = import_dependencies()
    rows = []
    for record in records:
        rank = first_correct_rank(record.retrieved_candidate_ids, record.reference_candidate_ids)
        chain_score, matched_levels, level_accuracy = administrative_chain_score(
            record.reference_admin_chain,
            record.predicted_admin_chain,
            args.admin_match_mode,
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "reference_candidate_ids": json.dumps(
                    record.reference_candidate_ids, ensure_ascii=False
                ),
                "retrieved_candidate_ids": json.dumps(
                    record.retrieved_candidate_ids, ensure_ascii=False
                ),
                "first_correct_rank": rank,
                "reference_admin_chain": json.dumps(
                    record.reference_admin_chain, ensure_ascii=False
                ),
                "predicted_admin_chain": json.dumps(
                    record.predicted_admin_chain, ensure_ascii=False
                ),
                "admin_reference_levels": len(record.reference_admin_chain),
                "admin_predicted_levels": len(record.predicted_admin_chain),
                "admin_matched_levels": matched_levels,
                "admin_level_accuracy": level_accuracy,
                "top_1_accuracy": int(rank is not None and rank <= 1),
                "top_5_accuracy": int(rank is not None and rank <= 5),
                "administrative_chain_accuracy": chain_score,
            }
        )
    result = pd.DataFrame(rows)
    if not (result["top_5_accuracy"] >= result["top_1_accuracy"]).all():
        raise AssertionError("Top-5 Accuracy must never be lower than Top-1 Accuracy.")
    return result


def summarize_metrics(metric_df: Any, metrics: Sequence[str]) -> Any:
    pd, np = import_dependencies()
    rows = []
    for metric in metrics:
        values = pd.to_numeric(metric_df[metric], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Metric {metric!r} contains missing or non-numeric scores.")
        array = values.to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "n": len(array),
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(metric_df: Any, metrics: Sequence[str], args: argparse.Namespace) -> Any:
    pd, np = import_dependencies()
    columns = [
        "metric",
        "n",
        "iterations",
        "confidence_level",
        "seed",
        "mean",
        "ci_lower",
        "ci_upper",
        "distance_to_lower",
        "distance_to_upper",
    ]
    if args.bootstrap_iterations == 0:
        return pd.DataFrame(columns=columns)

    values = metric_df.loc[:, list(metrics)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Bootstrap input contains missing or non-numeric scores.")
    matrix = values.to_numpy(dtype=float)
    n = matrix.shape[0]
    rng = np.random.default_rng(args.random_seed)
    # The same QA indices are reused for every metric in each iteration.
    indices = rng.integers(0, n, size=(args.bootstrap_iterations, n))
    sampled_means = matrix[indices, :].mean(axis=1)
    observed = matrix.mean(axis=0)
    alpha = 1.0 - args.confidence_level
    lower = np.quantile(sampled_means, alpha / 2.0, axis=0)
    upper = np.quantile(sampled_means, 1.0 - alpha / 2.0, axis=0)

    rows = []
    for index, metric in enumerate(metrics):
        rows.append(
            {
                "metric": metric,
                "n": n,
                "iterations": args.bootstrap_iterations,
                "confidence_level": args.confidence_level,
                "seed": args.random_seed,
                "mean": float(observed[index]),
                "ci_lower": float(lower[index]),
                "ci_upper": float(upper[index]),
                "distance_to_lower": float(observed[index] - lower[index]),
                "distance_to_upper": float(upper[index] - observed[index]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def resolve_ragas_metrics(df: Any) -> tuple[Any, list[str]]:
    normalized = {normalize_column_name(column): column for column in df.columns}
    selected: dict[str, Any] = {}
    for canonical, aliases in RAGAS_METRIC_ALIASES.items():
        normalized_aliases = {normalize_column_name(alias) for alias in aliases}
        matches = [original for name, original in normalized.items() if name in normalized_aliases]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one Ragas column for {canonical!r}; found {matches}."
            )
        selected[canonical] = matches[0]
    renamed = df.rename(columns={original: canonical for canonical, original in selected.items()})
    return renamed, list(RAGAS_METRIC_ALIASES)


def merge_ragas_scores(geo_df: Any, path: Path) -> tuple[Any, list[str]]:
    pd, _ = import_dependencies()
    if not path.exists():
        raise FileNotFoundError(f"Ragas score file does not exist: {path}")
    ragas = pd.read_csv(path, encoding="utf-8-sig")
    sample_col = resolve_column(ragas.columns, "sample_id", explicit=None, required=True)
    if sample_col != "sample_id":
        ragas = ragas.rename(columns={sample_col: "sample_id"})
    if ragas["sample_id"].duplicated().any():
        raise ValueError("The Ragas score file contains duplicate sample_id values.")
    ragas, ragas_metrics = resolve_ragas_metrics(ragas)
    keep = ["sample_id", *ragas_metrics]
    merged = geo_df.merge(ragas[keep], on="sample_id", how="outer", indicator=True, validate="one_to_one")
    unmatched = merged.loc[merged["_merge"] != "both", ["sample_id", "_merge"]]
    if not unmatched.empty:
        preview = unmatched.head(10).to_dict(orient="records")
        raise ValueError(f"sample_id mismatch between geospatial and Ragas files: {preview}")
    merged = merged.drop(columns=["_merge"])
    return merged, [*ragas_metrics, *GEO_METRICS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_csv(df: Any, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote %s", path)


def write_outputs(
    records: Sequence[GeoRecord], geo_df: Any, args: argparse.Namespace
) -> tuple[Any, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geo_summary = summarize_metrics(geo_df, GEO_METRICS)
    geo_bootstrap = bootstrap_metrics(geo_df, GEO_METRICS, args)
    write_csv(geo_df, args.output_dir / "geo_row_scores.csv")
    write_csv(geo_summary, args.output_dir / "geo_summary.csv")
    write_csv(geo_bootstrap, args.output_dir / "geo_bootstrap_ci.csv")

    combined_summary = None
    combined_bootstrap = None
    if args.ragas_scores is not None:
        combined_df, combined_metrics = merge_ragas_scores(geo_df, args.ragas_scores)
        combined_summary = summarize_metrics(combined_df, combined_metrics)
        combined_bootstrap = bootstrap_metrics(combined_df, combined_metrics, args)
        write_csv(combined_df, args.output_dir / "combined_row_scores.csv")
        write_csv(combined_summary, args.output_dir / "combined_summary.csv")
        write_csv(combined_bootstrap, args.output_dir / "combined_bootstrap_ci.csv")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "rows_evaluated": len(records),
        "metrics": list(GEO_METRICS),
        "top_k_values": [1, 5],
        "candidate_matching": "NFKC-normalized exact ID match",
        "case_sensitive": args.case_sensitive,
        "admin_match_mode": args.admin_match_mode,
        "administrative_chain_order": "coarse_to_fine",
        "bootstrap_iterations": args.bootstrap_iterations,
        "confidence_level": args.confidence_level,
        "random_seed": args.random_seed,
        "ragas_scores_file": str(args.ragas_scores.resolve()) if args.ragas_scores else None,
        "ragas_scores_sha256": sha256_file(args.ragas_scores) if args.ragas_scores else None,
        "package_versions": {
            package: package_version(package) for package in ("pandas", "numpy", "openpyxl")
        },
    }
    metadata_path = args.output_dir / "geo_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", metadata_path)
    return geo_summary, combined_summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        records = load_records(args)
        LOGGER.info("Loaded %d valid geospatial evaluation rows", len(records))
        geo_df = evaluate_records(records, args)
        geo_summary, combined_summary = write_outputs(records, geo_df, args)

        print("\nGeospatial evaluation results")
        print(geo_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if combined_summary is not None:
            print("\nCombined Ragas and geospatial results")
            print(combined_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Evaluation cancelled by user.")
        return 130
    except Exception as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed traceback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
