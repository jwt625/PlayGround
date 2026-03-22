#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus-wide OFC 2026 word maps.")
    parser.add_argument("--index", default="paper_text_index.json")
    parser.add_argument("--extracted-text-dir", default="extracted_text")
    parser.add_argument("--top-terms", default="corpus_top_terms.csv")
    parser.add_argument("--top-bigrams", default="corpus_top_bigrams.csv")
    parser.add_argument("--top-trigrams", default="corpus_top_trigrams.csv")
    parser.add_argument("--wordmap-png", default="wordmap.png")
    parser.add_argument("--wordmap-svg", default="wordmap.svg")
    parser.add_argument("--tfidf-json", default="paper_tfidf_keywords.json")
    parser.add_argument("--extra-stopwords-path", default="")
    parser.add_argument("--top-k", type=int, default=250)
    return parser.parse_args()


def ensure_nltk() -> None:
    for package in ("wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)


def load_records(index_path: Path) -> dict[str, dict]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["papers"]


def resolve_text_path(record: dict, extracted_text_dir: Path) -> Path:
    raw_path = record.get("text_path", "")
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
        local_candidate = extracted_text_dir / candidate.name
        if local_candidate.exists():
            return local_candidate
    pdf_name = record.get("pdf_name", "")
    if pdf_name:
        stem = Path(pdf_name).stem
        for suffix in (".txt", ".md"):
            candidate = extracted_text_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve extracted text for record: {record.get('pdf_name') or raw_path}")


def boilerplate_patterns() -> list[re.Pattern]:
    raw = [
        r"^\s*ofc\s+2026\b",
        r"^\s*978-\d",
        r"^\s*isbn\b",
        r"^\s*doi\b",
        r"^\s*authorized licensed use\b",
        r"^\s*downloaded on\b",
        r"^\s*copyright\b",
        r"^\s*©",
        r"^\s*page\s+\d+\b",
        r"^\s*[mtw][a-z]?\d+[a-z]?\.\d+\b",
    ]
    return [re.compile(pattern, re.I) for pattern in raw]


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def repeated_noise_lines(records: dict[str, dict]) -> set[str]:
    doc_counts: Counter[str] = Counter()
    extracted_text_dir = Path("extracted_text")
    for record in records.values():
        if record.get("status") != "success":
            continue
        text_path = resolve_text_path(record, extracted_text_dir)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        unique_lines = {
            normalize_line(line)
            for line in text.splitlines()
            if normalize_line(line)
        }
        for line in unique_lines:
            if len(line) <= 140:
                doc_counts[line] += 1
    min_docs = max(8, math.ceil(len(records) * 0.03))
    return {line for line, count in doc_counts.items() if count >= min_docs}


def load_extra_stopwords(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    stopwords: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().lower()
        if not value or value.startswith("#"):
            continue
        stopwords.add(value)
    return stopwords


def clean_text(raw_text: str, repeated_lines: set[str], lemmatizer: WordNetLemmatizer, stopwords: set[str]) -> tuple[str, list[str]]:
    kept_lines: list[str] = []
    patterns = boilerplate_patterns()
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue
        if line in repeated_lines:
            continue
        if any(pattern.search(line) for pattern in patterns):
            continue
        if line.lower() in {"references", "reference"}:
            break
        kept_lines.append(line)

    text = "\n".join(kept_lines).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
    text = re.sub(r"[^a-z0-9+\-./\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    raw_tokens = re.findall(r"[a-z0-9]+(?:[+\-./][a-z0-9]+)*", text)
    tokens: list[str] = []
    for token in raw_tokens:
        if token.isdigit():
            continue
        if re.search(r"\d\.\d", token):
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if token in stopwords:
            continue
        normalized = token
        if re.fullmatch(r"[a-z]+", token):
            normalized = lemmatizer.lemmatize(token, "n")
            normalized = lemmatizer.lemmatize(normalized, "v")
        if normalized in stopwords:
            continue
        tokens.append(normalized)
    return " ".join(tokens), tokens


def write_counter_csv(path: Path, rows: list[tuple[str, int]], value_header: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([value_header, "count"])
        writer.writerows(rows)


def top_ngrams(docs: list[list[str]], n: int, top_k: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    banned_tokens = {
        "journal",
        "lightwave",
        "technology",
        "natural",
        "science",
        "foundation",
        "state",
        "laboratory",
        "opt",
        "commun",
        "netw",
        "pi",
        "leq",
        "qquad",
    }
    for tokens in docs:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            gram_tokens = tokens[i : i + n]
            if len(set(gram_tokens)) == 1:
                continue
            if any(token in banned_tokens for token in gram_tokens):
                continue
            counter.update([" ".join(gram_tokens)])
    return counter.most_common(top_k)


def build_tfidf_keywords(cleaned_docs: dict[str, str], top_n: int = 12) -> dict[str, list[dict[str, float]]]:
    filenames = list(cleaned_docs.keys())
    corpus = [cleaned_docs[name] for name in filenames]
    if not corpus:
        return {}
    min_df = 1 if len(corpus) < 3 else 2
    max_df = 1.0 if len(corpus) < 3 else 0.85
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-z0-9][a-z0-9+\-./]*\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )
    matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()
    output: dict[str, list[dict[str, float]]] = {}
    for idx, filename in enumerate(filenames):
        row = matrix.getrow(idx)
        if row.nnz == 0:
            output[filename] = []
            continue
        pairs = sorted(zip(row.indices, row.data), key=lambda item: item[1], reverse=True)[:top_n]
        output[filename] = [
            {"term": features[feature_idx], "score": round(float(score), 6)}
            for feature_idx, score in pairs
        ]
    return output


def main() -> None:
    args = parse_args()
    ensure_nltk()

    index_path = Path(args.index).resolve()
    extracted_text_dir = Path(args.extracted_text_dir).resolve()
    records = load_records(index_path)
    repeated_lines = repeated_noise_lines(records)

    lemmatizer = WordNetLemmatizer()
    stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
    stopwords.update(
        {
            "figure",
            "fig",
            "table",
            "eq",
            "equation",
            "section",
            "ref",
            "reference",
            "author",
            "abstract",
            "email",
            "mail",
            "address",
            "mathbf",
            "mathbb",
            "mathcal",
            "mathrm",
            "forall",
            "vol",
            "pp",
            "conference",
            "use",
            "et",
            "al",
            "using",
            "show",
            "shown",
            "respectively",
            "result",
            "results",
            "paper",
            "proposed",
            "demonstrate",
            "demonstrated",
            "based",
            "optical",
            "signal",
            "fiber",
            "fibre",
            "power",
            "network",
            "transmission",
            "channel",
            "data",
            "loss",
            "laser",
            "bandwidth",
            "phase",
            "wavelength",
            "time",
            "link",
            "rate",
            "frequency",
            "communication",
            "measure",
            "design",
            "output",
            "process",
            "high",
            "noise",
            "technology",
            "mode",
            "device",
            "gain",
            "spectrum",
            "experimental",
            "measurement",
            "method",
            "photonic",
            "photonics",
            "operation",
            "filter",
            "amplifier",
            "range",
            "input",
            "setup",
            "architecture",
            "receiver",
            "silicon",
            "core",
            "path",
            "control",
            "waveguide",
            "application",
            "light",
            "modulator",
            "cable",
            "test",
            "function",
            "parameter",
            "structure",
            "distribution",
            "transmitter",
            "db",
            "dbm",
            "nm",
            "km",
            "ghz",
            "ber",
            "ieee",
            "ofc",
            "china",
            "usa",
            "san",
            "francisco",
            "ca",
        }
    )
    stopwords.update(load_extra_stopwords(Path(args.extra_stopwords_path)) if args.extra_stopwords_path else set())

    cleaned_docs: dict[str, str] = {}
    token_docs: dict[str, list[str]] = {}
    unigram_counter: Counter[str] = Counter()

    for pdf_name, record in sorted(records.items()):
        if record.get("status") != "success":
            continue
        text_path = resolve_text_path(record, extracted_text_dir)
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text, tokens = clean_text(raw_text, repeated_lines, lemmatizer, stopwords)
        cleaned_docs[pdf_name] = cleaned_text
        token_docs[pdf_name] = tokens
        unigram_counter.update(tokens)

    top_terms = unigram_counter.most_common(args.top_k)
    top_bigrams = top_ngrams(list(token_docs.values()), 2, args.top_k)
    top_trigrams = top_ngrams(list(token_docs.values()), 3, args.top_k)
    tfidf_keywords = build_tfidf_keywords(cleaned_docs)

    write_counter_csv(Path(args.top_terms).resolve(), top_terms, "term")
    write_counter_csv(Path(args.top_bigrams).resolve(), top_bigrams, "bigram")
    write_counter_csv(Path(args.top_trigrams).resolve(), top_trigrams, "trigram")
    Path(args.tfidf_json).resolve().write_text(
        json.dumps(tfidf_keywords, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    wordcloud = WordCloud(
        width=2200,
        height=1400,
        background_color="white",
        colormap="viridis",
        max_words=300,
    ).generate_from_frequencies(dict(top_terms))
    wordcloud.to_file(str(Path(args.wordmap_png).resolve()))
    Path(args.wordmap_svg).resolve().write_text(wordcloud.to_svg(), encoding="utf-8")

    print(
        json.dumps(
            {
                "documents": len(cleaned_docs),
                "repeated_noise_lines": len(repeated_lines),
                "top_term": top_terms[0] if top_terms else None,
                "top_bigram": top_bigrams[0] if top_bigrams else None,
                "top_trigram": top_trigrams[0] if top_trigrams else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
