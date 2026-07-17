#!/usr/bin/env python3
"""Run a reproducible BM25-RAG baseline.

The script accepts an dataset and one user question, then returns:

1. the ranked Top-5 address candidates (configurable with ``--top-k``);
2. the retrieved context assembled from those candidates; and
3. an LLM response grounded in the retrieved context.

Required dataset fields
-----------------------
``semantic_text``
    Text used for BM25 indexing.

``sample_id`` is strongly recommended as a stable candidate identifier. 

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


LOGGER = logging.getLogger("bm25_rag")

TEXT_COLUMN_ALIASES = (
    "semantic_text",
    "text",
    "semantic_description",
)

ID_COLUMN_ALIASES = (
    "sample_id",
    "candidate_id",
    "poi_id",
    "id",
    "样本编号",
    "候选编号",
)

PLACE_NAME_ALIASES = (
    "place_name",
    "toponym",
    "name",
    "地点名称",
    "地名",
)

# These fields make the Top-k output useful while keeping the full STA text.
CANDIDATE_OUTPUT_FIELDS = (
    "place_name",
    "toponym",
    "province",
    "city",
    "district",
    "detailed_address",
    "longitude",
    "latitude",
    "poi_type",
    "category_level_1",
    "category_level_2",
    "category_level_3",
    "category_level_4",
    "category_level_5",
    "administrative_granularity",
    "query_difficulty",
)

EMPTY_STRINGS = {"", "nan", "none", "null", "nat", "<na>"}
TOKEN_CONTENT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
LEXICAL_SPAN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*"
)


@dataclass(frozen=True)
class STADocument:
    """One indexed STA record."""

    candidate_id: str
    semantic_text: str
    metadata: dict[str, Any]


class BM25Index:
    """Small, dependency-light implementation of Okapi BM25.

    The IDF variant is ``log(1 + (N - df + 0.5) / (df + 0.5))``, which keeps
    rare-term weights positive and is widely used in BM25 implementations.
    """

    def __init__(
        self,
        documents: Sequence[STADocument],
        tokenizer: Callable[[str], list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("The dataset contains no indexable documents.")
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.documents = list(documents)
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)

        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequency: defaultdict[str, int] = defaultdict(int)

        for document in self.documents:
            tokens = tokenizer(document.semantic_text)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            for term in frequencies:
                document_frequency[term] += 1

        if not any(self.document_lengths):
            raise ValueError("STA texts produced no valid tokens.")

        self.document_count = len(self.documents)
        self.average_document_length = sum(self.document_lengths) / self.document_count
        self.inverse_document_frequency = {
            term: math.log(
                1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> list[float]:
        """Return one BM25 score per document."""

        query_terms = self.tokenizer(query)
        if not query_terms:
            raise ValueError("The question contains no valid retrieval terms.")

        scores = [0.0] * self.document_count
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            length_normalizer = self.k1 * (
                1.0 - self.b + self.b * document_length / self.average_document_length
            )

            # Repeating a query term does not repeatedly inflate its score.
            for term in set(query_terms):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + length_normalizer
                scores[index] += self.inverse_document_frequency.get(term, 0.0) * (
                    numerator / denominator
                )
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[STADocument, float]]:
        """Return the Top-k documents with deterministic tie breaking."""

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        scores = self.score(query)
        ranked_indices = sorted(
            range(self.document_count), key=lambda index: (-scores[index], index)
        )[: min(top_k, self.document_count)]

        if ranked_indices and scores[ranked_indices[0]] <= 0:
            LOGGER.warning(
                "No query term matched the corpus; returned candidates have zero scores."
            )

        return [(self.documents[index], scores[index]) for index in ranked_indices]


class JiebaTokenizer:
    """Deterministic Chinese/Latin tokenizer for the BM25 baseline."""

    def __init__(self, custom_terms: Sequence[str] = ()) -> None:
        try:
            import jieba
        except ImportError as exc:
            raise RuntimeError(
                "jieba is required for Chinese BM25 tokenization. "
                "Install it with: pip install jieba"
            ) from exc

        self.jieba = jieba
        self.jieba.setLogLevel(logging.WARNING)
        self.jieba.initialize()

        # Keeping complete toponyms as lexical terms improves address-name matching.
        for term in custom_terms:
            normalized = normalize_text(term)
            if normalized:
                self.jieba.add_word(normalized, freq=10_000_000)

    def __call__(self, text: str) -> list[str]:
        normalized = normalize_text(text).casefold()
        return [
            token
            for token in (part.strip() for part in self.jieba.lcut(normalized, HMM=False))
            if token and TOKEN_CONTENT_RE.search(token)
        ]


class CJKNgramTokenizer:
    """Version-independent analyzer for Chinese and Latin address text.

    Chinese spans yield character unigrams and bigrams. Latin words and
    numeric/address expressions are retained as complete lowercase terms.
    This analyzer is deterministic across machines and does not use a learned
    segmentation model or domain-specific synonym expansion.
    """

    def __call__(self, text: str) -> list[str]:
        normalized = normalize_text(text).casefold()
        tokens: list[str] = []
        for match in LEXICAL_SPAN_RE.finditer(normalized):
            span = match.group(0)
            if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", span):
                tokens.extend(span)
                tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
            else:
                tokens.append(span)
        return tokens


def create_tokenizer(name: str, place_names: Sequence[str]) -> Callable[[str], list[str]]:
    if name == "jieba":
        return JiebaTokenizer(place_names)
    if name == "cjk_ngram":
        return CJKNgramTokenizer()
    raise ValueError(f"Unsupported tokenizer: {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Top-k STA records with BM25 and generate a grounded response."
    )
    parser.add_argument("--data", required=True, type=Path, help="dataset file.")
    parser.add_argument(
        "--question",
        default=None,
        help="User question. If omitted, the script prompts for one interactively.",
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name; default: first sheet.")
    parser.add_argument("--text-col", default=None, help="text column name.")
    parser.add_argument("--id-col", default=None, help="Candidate-ID column name.")
    parser.add_argument("--place-name-col", default=None, help="Place-name column name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates; default: 5.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 term saturation parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 length normalization parameter.")
    parser.add_argument(
        "--tokenizer",
        choices=("cjk_ngram", "jieba"),
        default="cjk_ngram",
        help="Lexical analyzer; default: deterministic cjk_ngram.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bm25_rag_result.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Return retrieval results without calling the generator LLM.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("BM25_API_BASE", "https://api.siliconflow.cn/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="SILICONFLOW_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BM25_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Generator model ID.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries performed by the OpenAI-compatible client.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if args.k1 <= 0:
        parser.error("--k1 must be greater than 0.")
    if not 0 <= args.b <= 1:
        parser.error("--b must be between 0 and 1.")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive.")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative.")
    return args


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace without changing the underlying content."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if text.strip().casefold() in EMPTY_STRINGS:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "_", normalize_text(value).casefold()).strip("_")


def safe_json_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolve_column(
    columns: Sequence[Any], explicit: str | None, aliases: Sequence[str], label: str
) -> Any | None:
    """Resolve a column case-insensitively while preserving its original label."""

    lookup = {normalize_column_name(column): column for column in columns}
    if explicit:
        resolved = lookup.get(normalize_column_name(explicit))
        if resolved is None:
            raise ValueError(f"Column {explicit!r} for {label} was not found.")
        return resolved
    for alias in aliases:
        resolved = lookup.get(normalize_column_name(alias))
        if resolved is not None:
            return resolved
    return None


def load_dataframe(path: Path, sheet: str | None = None) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install it with: pip install pandas") from exc

    if not path.exists():
        raise FileNotFoundError(f"STA dataset not found: {path}")

    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Supported dataset formats are XLSX, XLS, XLSM, CSV, JSON, and JSONL.")


def load_sta_documents(
    path: Path,
    *,
    sheet: str | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
    place_name_col: str | None = None,
) -> tuple[list[STADocument], list[str]]:
    """Load and validate records, returning documents and custom toponym terms."""

    dataframe = load_dataframe(path, sheet)
    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    text_column = resolve_column(dataframe.columns, text_col, TEXT_COLUMN_ALIASES, "STA text")
    if text_column is None:
        raise ValueError(
            "No text column was found. Use --text-col to identify the enhanced-text field."
        )
    id_column = resolve_column(dataframe.columns, id_col, ID_COLUMN_ALIASES, "candidate ID")
    name_column = resolve_column(
        dataframe.columns, place_name_col, PLACE_NAME_ALIASES, "place name"
    )

    documents: list[STADocument] = []
    place_names: list[str] = []
    seen_ids: set[str] = set()

    for row_position, (_, row) in enumerate(dataframe.iterrows(), start=1):
        semantic_text = normalize_text(row[text_column])
        if not semantic_text:
            LOGGER.warning("Skipped row %d because its STA text is empty.", row_position)
            continue

        raw_id = row[id_column] if id_column is not None else None
        candidate_id = normalize_text(raw_id) or f"DOC_{row_position:06d}"
        if candidate_id in seen_ids:
            raise ValueError(
                f"Duplicate candidate ID {candidate_id!r}. Top-k evaluation requires unique IDs."
            )
        seen_ids.add(candidate_id)

        metadata: dict[str, Any] = {}
        for field in CANDIDATE_OUTPUT_FIELDS:
            resolved = resolve_column(dataframe.columns, None, (field,), field)
            if resolved is not None:
                value = safe_json_value(row[resolved])
                if value is not None and normalize_text(value):
                    metadata[field] = value

        place_name = normalize_text(row[name_column]) if name_column is not None else ""
        if place_name:
            place_names.append(place_name)

        documents.append(
            STADocument(
                candidate_id=candidate_id,
                semantic_text=semantic_text,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError("No valid records remained after validation.")
    return documents, place_names


def build_context(ranked: Sequence[tuple[STADocument, float]]) -> str:
    """Preserve candidate boundaries when constructing the generation context."""

    chunks = []
    for rank, (document, score) in enumerate(ranked, start=1):
        chunks.append(
            f"[Candidate {rank}; ID={document.candidate_id}; BM25={score:.6f}]\n"
            f"{document.semantic_text}"
        )
    return "\n\n".join(chunks)


def format_candidates(ranked: Sequence[tuple[STADocument, float]]) -> list[dict[str, Any]]:
    candidates = []
    for rank, (document, score) in enumerate(ranked, start=1):
        candidate = {
            "rank": rank,
            "candidate_id": document.candidate_id,
            "bm25_score": round(float(score), 8),
            **document.metadata,
            "semantic_text": document.semantic_text,
        }
        candidates.append(candidate)
    return candidates


def generate_response(
    *,
    question: str,
    context: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
) -> str:
    """Generate an answer constrained to the retrieved  evidence."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required. Install it with: pip install openai") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=request_timeout,
        max_retries=max_retries,
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a geospatial address question-answering assistant. "
                    "Answer only from the supplied retrieved context. Preserve place names, "
                    "administrative levels, addresses, categories, and coordinates exactly. "
                    "Respond in the same language as the question. "
                    "If the context does not contain enough evidence, explicitly state that "
                    "the answer cannot be determined from the retrieved context."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{context}",
            },
        ],
    )

    response = completion.choices[0].message.content
    if not response or not response.strip():
        raise RuntimeError("The generator returned an empty response.")
    return response.strip()


def run_bm25_rag(
    *,
    data_path: Path,
    question: str,
    sheet: str | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
    place_name_col: str | None = None,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    tokenizer_name: str = "cjk_ngram",
    generate: bool = True,
    api_base: str = "https://api.siliconflow.cn/v1",
    api_key_env: str = "",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 512,
    request_timeout: float = 180.0,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Programmatic entry point for retrieval and optional answer generation."""

    normalized_question = normalize_text(question)
    if not normalized_question:
        raise ValueError("The user question is empty.")

    documents, place_names = load_sta_documents(
        data_path,
        sheet=sheet,
        text_col=text_col,
        id_col=id_col,
        place_name_col=place_name_col,
    )
    tokenizer = create_tokenizer(tokenizer_name, place_names)
    index = BM25Index(documents, tokenizer, k1=k1, b=b)
    ranked = index.search(normalized_question, top_k=top_k)
    context = build_context(ranked)

    response: str | None = None
    if generate:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is empty. "
                "Set it before generation, or use --no-generate."
            )
        response = generate_response(
            question=normalized_question,
            context=context,
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )

    return {
        "question": normalized_question,
        "retrieval_method": "Okapi BM25",
        "retrieval_parameters": {
            "top_k": min(top_k, len(documents)),
            "k1": k1,
            "b": b,
            "tokenizer": (
                "jieba (HMM disabled; corpus toponyms added as custom terms)"
                if tokenizer_name == "jieba"
                else "deterministic Chinese character unigrams/bigrams plus Latin tokens"
            ),
            "corpus_size": len(documents),
        },
        "top_k_candidates": format_candidates(ranked),
        "context": context,
        "generator": None if not generate else model,
        "response": response,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    question = normalize_text(args.question)
    if not question:
        if not sys.stdin.isatty():
            raise ValueError("Provide --question when standard input is non-interactive.")
        question = normalize_text(input("User question: "))

    result = run_bm25_rag(
        data_path=args.data,
        question=question,
        sheet=args.sheet,
        text_col=args.text_col,
        id_col=args.id_col,
        place_name_col=args.place_name_col,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        tokenizer_name=args.tokenizer,
        generate=not args.no_generate,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    LOGGER.info("Wrote BM25-RAG result to %s", args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error
