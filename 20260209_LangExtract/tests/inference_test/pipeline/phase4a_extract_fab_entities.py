#!/usr/bin/env python3
"""Phase 4a: Extract fabrication entities from corpus markdown using LangExtract.

This phase focuses on process steps, recipes, equipment, materials,
parameters, and site/facility mentions with source-grounded spans.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "semiconductor_processing_dataset"
    / "processed_documents"
    / "text_extracted"
    / "marker"
)
DEFAULT_METADATA_DIR = (
    PROJECT_ROOT / "semiconductor_processing_dataset" / "processed_documents" / "metadata"
)
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4a_fab_entities_raw.jsonl"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4a_fab_entities_summary.json"
DEFAULT_SCHEMA_FILE = PROJECT_ROOT / "tests" / "inference_test" / "pipeline" / "config" / "fab_entity_schema.json"


SECTION_HEADER_RE = re.compile(
    r"^#{1,6}\s*(methods?|experimental|fabrication|device fabrication|supplementary|appendix|materials?)",
    re.IGNORECASE,
)
KEYWORD_RE = re.compile(
    r"\b(fabricat|deposi|etch|anneal|sputter|evaporat|lithograph|clean|pattern|oxid|junction|process|recipe|cleanroom|facility)\w*\b",
    re.IGNORECASE,
)
STRONG_FAB_SECTION_RE = re.compile(
    r"(fabricat|method|process|experimental|materials?|procedure|recipe|cleanroom|device_fabrication)",
    re.IGNORECASE,
)
NOISY_SECTION_RE = re.compile(
    r"(table[_\s]?of[_\s]?contents?|acknowledg|abstract|introduction|background|theory|discussion|implication|reference|appendix|curriculum)",
    re.IGNORECASE,
)
EQUATION_LIKE_RE = re.compile(r"([=\\{}_^]|frac|omega|rho|sigma|hbar|mathcal)")
EQUIPMENT_CUE_RE = re.compile(
    r"(etch|sputter|evapor|deposition|chamber|reactor|lithograph|aligner|microscope|fridge|refrigerator|laser|plasma|ald|rie|icp|beam|hemt|filter|isolator|tool|system|furnace|mill|resonator)",
    re.IGNORECASE,
)
PROCESS_ACTION_RE = re.compile(
    r"(deposit|deposition|etch|clean|anneal|sputter|evaporat|lithograph|pattern|oxid|mill|bond|weld|spin|rinse|develop|liftoff|lift[-\s]?off|strip|coat|bake|load|unload|transfer|dicing|wire\s?bond|ald|pecvd|rie|icp|uv[-\s]?ozone|boe|hf)",
    re.IGNORECASE,
)
SAMPLE_CONTEXT_CUE_RE = re.compile(
    r"(chip|wafer|substrate|qubit|junction|resonator|capacitor|cavity|device|transmon|sample|stack|pad|bridge|cpw|line)",
    re.IGNORECASE,
)
CHAPTER_SECTION_RE = re.compile(r"^\d(_\d+){0,3}$")
GENERIC_PROCESS_TEXT = {
    "fabricated",
    "deposited",
    "measured",
    "implemented",
    "designed",
    "assume",
    "modeled",
}
MATERIAL_STOPWORDS = {
    "currents",
    "voltages",
    "wires",
    "inductors",
    "capacitors",
    "transformers",
    "materials",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4a: Extract fabrication entities with LangExtract")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--schema-file", type=Path, default=DEFAULT_SCHEMA_FILE)
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all")
    parser.add_argument("--model-id", default=os.getenv("FAB_MODEL_ID") or os.getenv("MODEL_ID") or "gemini-2.5-flash")
    parser.add_argument(
        "--provider",
        default=os.getenv("FAB_PROVIDER", ""),
        help="Optional LangExtract provider name/class (e.g., OpenAILanguageModel, GeminiLanguageModel). "
        "If empty and API_URL/API_TOKEN are set, defaults to openai-compatible mode.",
    )
    parser.add_argument("--max-char-buffer", type=int, default=1200)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--extraction-passes", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts-per-doc", type=int, default=2)
    parser.add_argument("--allow-ungrounded", action="store_true", help="Keep records without valid char spans.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip endpoint preflight check.")
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def load_metadata_hints(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    hints: dict[str, dict[str, Any]] = {}
    if not metadata_dir.exists():
        return hints
    for fp in sorted(metadata_dir.glob("*.json")):
        if fp.name.upper() == "TEMPLATE.JSON":
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        hints[fp.stem] = {
            "institution": data.get("institution"),
            "department": data.get("department"),
            "source_type": data.get("source_type"),
            "title": data.get("title"),
            "year": data.get("year"),
        }
    return hints


def load_allowed_entity_classes(schema_file: Path) -> set[str]:
    if not schema_file.exists():
        return {
            "process_step",
            "recipe",
            "equipment",
            "material",
            "parameter",
            "site",
            "cleanroom",
            "facility",
            "sample_context",
        }
    data = json.loads(schema_file.read_text())
    vals = data.get("entity_classes") or []
    return {str(v).strip() for v in vals if str(v).strip()}


def preflight_inference(api_url: str, api_token: str, model_id: str) -> tuple[bool, str]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(api_url.rstrip("/") + "/chat/completions", json=payload, headers=headers, timeout=30)
        if resp.status_code >= 300:
            return False, f"Preflight failed: HTTP {resp.status_code} {resp.text[:180]}"
        return True, "ok"
    except Exception as exc:
        return False, f"Preflight exception: {exc}"


def select_fabrication_text(full_text: str, max_chars: int = 30000) -> str:
    lines = full_text.splitlines()

    # 1) Heading-window strategy.
    selected_chunks: list[str] = []
    for i, line in enumerate(lines):
        if SECTION_HEADER_RE.match(line.strip()):
            end = min(i + 220, len(lines))
            chunk = "\n".join(lines[i:end]).strip()
            if chunk:
                selected_chunks.append(chunk)

    # 2) Keyword-neighborhood fallback.
    if not selected_chunks:
        hit_idx = [i for i, line in enumerate(lines) if KEYWORD_RE.search(line)]
        for idx in hit_idx[:120]:
            s = max(0, idx - 2)
            e = min(len(lines), idx + 6)
            chunk = "\n".join(lines[s:e]).strip()
            if chunk:
                selected_chunks.append(chunk)

    if not selected_chunks:
        selected = full_text
    else:
        selected = "\n\n".join(selected_chunks)

    if len(selected) > max_chars:
        return selected[:max_chars]
    return selected


def build_prompt_and_examples(lx):
    prompt = (
        "Extract fabrication knowledge entities from superconducting-device/process text. "
        "Use exact source spans for extraction_text, do not paraphrase extraction_text, "
        "and preserve order of appearance. "
        "Return entities of classes: process_step, recipe, equipment, material, parameter, "
        "site, cleanroom, facility, sample_context. "
        "For each entity, provide attributes with section_hint and confidence in [high, medium, low]. "
        "If site/facility is inferred rather than explicit, set inferred=true in attributes."
    )

    example_text = (
        "We deposit 100 nm aluminum by electron-beam evaporation at 1 A/s. "
        "Then we pattern with photolithography and etch in 7:1 BOE for 45 s. "
        "Devices were fabricated in the Princeton University Micro/Nano Fabrication Laboratory."
    )

    examples = [
        lx.data.ExampleData(
            text=example_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class="process_step",
                    extraction_text="deposit 100 nm aluminum by electron-beam evaporation",
                    attributes={
                        "step_name": "metal_deposition",
                        "equipment": "electron-beam evaporation",
                        "materials": ["aluminum"],
                        "section_hint": "fabrication",
                        "confidence": "high",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="parameter",
                    extraction_text="100 nm",
                    attributes={
                        "name": "thickness",
                        "value": "100",
                        "unit": "nm",
                        "applies_to": "metal_deposition",
                        "section_hint": "fabrication",
                        "confidence": "high",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="equipment",
                    extraction_text="electron-beam evaporation",
                    attributes={
                        "tool_type": "deposition",
                        "section_hint": "fabrication",
                        "confidence": "high",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="recipe",
                    extraction_text="etch in 7:1 BOE for 45 s",
                    attributes={
                        "process_type": "wet_etching",
                        "materials": ["BOE"],
                        "parameters": ["7:1", "45 s"],
                        "section_hint": "fabrication",
                        "confidence": "high",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="material",
                    extraction_text="BOE",
                    attributes={
                        "material_type": "etchant",
                        "section_hint": "fabrication",
                        "confidence": "high",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="cleanroom",
                    extraction_text="Princeton University Micro/Nano Fabrication Laboratory",
                    attributes={
                        "institution": "Princeton University",
                        "inferred": "false",
                        "section_hint": "acknowledgements",
                        "confidence": "high",
                    },
                ),
            ],
        )
    ]
    return prompt, examples


def _safe_relpath(fp: Path) -> str:
    try:
        return str(fp.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(fp)


def _normalize_section_hint(v: Any) -> str:
    s = str(v or "unknown").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _as_text(v: Any) -> str:
    if isinstance(v, (str, int, float)):
        return str(v).strip()
    if isinstance(v, dict):
        for key in ("text", "value", "name"):
            vv = v.get(key)
            if isinstance(vv, (str, int, float)) and str(vv).strip():
                return str(vv).strip()
        return ""
    return ""


def _section_allowed(entity_class: str, section_hint: str) -> bool:
    section_norm = _normalize_section_hint(section_hint)
    in_strong_fab = bool(STRONG_FAB_SECTION_RE.search(section_norm))
    is_chapter_section = bool(CHAPTER_SECTION_RE.match(section_norm))

    # Location entities are frequently found in acknowledgements/affiliations.
    if entity_class in {"site", "cleanroom", "facility"}:
        return not re.search(r"(reference|table_of_contents)", section_norm)

    # Process-heavy and equipment classes should stay in fabrication/method style sections.
    if entity_class in {"process_step", "recipe", "parameter", "equipment"}:
        if NOISY_SECTION_RE.search(section_norm) and not in_strong_fab:
            return False
        return in_strong_fab or is_chapter_section

    # Material/context can appear in broader sections, but keep out obvious front/back matter.
    if entity_class in {"material", "sample_context"}:
        if re.search(r"(table_of_contents|reference|curriculum|abstract|background|introduction|implications)", section_norm):
            return False
        return True

    return True


def _low_signal_extraction(entity_class: str, text: str, section_hint: str = "unknown") -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) > 260:
        return True
    section_norm = _normalize_section_hint(section_hint)
    in_strong_fab = bool(STRONG_FAB_SECTION_RE.search(section_norm))
    is_chapter_section = bool(CHAPTER_SECTION_RE.match(section_norm))

    words = len(t.split())
    if entity_class in {"process_step", "recipe", "parameter"} and words > 35:
        return True
    if entity_class == "process_step" and t.lower() in GENERIC_PROCESS_TEXT:
        return True
    if entity_class == "process_step" and EQUATION_LIKE_RE.search(t) and not KEYWORD_RE.search(t):
        return True
    if entity_class == "process_step" and not (in_strong_fab or is_chapter_section) and not PROCESS_ACTION_RE.search(t):
        return True
    if entity_class == "equipment" and not EQUIPMENT_CUE_RE.search(t):
        return True
    if entity_class == "sample_context" and not SAMPLE_CONTEXT_CUE_RE.search(t):
        return True
    if entity_class == "material" and t.lower() in MATERIAL_STOPWORDS:
        return True

    # Reject mostly symbolic expressions for non-parameter classes.
    alpha = sum(ch.isalpha() for ch in t)
    ratio = alpha / max(1, len(t))
    if entity_class != "parameter" and ratio < 0.25:
        return True

    return False


def to_output_extractions(
    selected_text: str,
    extractions: list[Any],
    allowed_entity_classes: set[str],
    allow_ungrounded: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ext in extractions:
        if ext.extraction_class not in allowed_entity_classes:
            continue
        attrs = ext.attributes if isinstance(ext.attributes, dict) else {}
        ext_text = _as_text(getattr(ext, "extraction_text", None))
        section_hint = _normalize_section_hint(attrs.get("section_hint", "unknown"))
        if not _section_allowed(ext.extraction_class, section_hint):
            continue
        if _low_signal_extraction(ext.extraction_class, ext_text, section_hint):
            continue

        ci = ext.char_interval
        start = ci.start_pos if ci else None
        end = ci.end_pos if ci else None

        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(selected_text):
            source_text = selected_text[start:end]
        else:
            match_text = ext_text
            pos = selected_text.find(match_text) if match_text else -1
            if pos >= 0:
                start = pos
                end = pos + len(match_text)
                source_text = selected_text[start:end]
            else:
                source_text = ext_text
                start = None
                end = None
                if not allow_ungrounded:
                    continue

        if _low_signal_extraction(ext.extraction_class, str(source_text), section_hint):
            continue

        rows.append(
            {
                "extraction_class": ext.extraction_class,
                "extraction_text": ext_text,
                "attributes": attrs,
                "source_text": source_text,
                "char_start": start,
                "char_end": end,
                "section_hint": section_hint,
            }
        )
    return rows


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    # Ensure local langextract repository package is importable.
    local_langextract_root = PROJECT_ROOT / "langextract"
    if str(local_langextract_root) not in sys.path:
        sys.path.insert(0, str(local_langextract_root))

    try:
        import langextract as lx
        from langextract import prompt_validation as pv
        from langextract import providers as lx_providers
    except Exception as exc:
        raise SystemExit(
            "Failed to import langextract. Ensure local repo exists at ./langextract "
            f"and dependencies are installed. Error: {exc}"
        )
    lx_providers.load_builtins_once()
    lx_providers.load_plugins_once()

    prompt, examples = build_prompt_and_examples(lx)
    metadata_hints = load_metadata_hints(args.metadata_dir)
    allowed_entity_classes = load_allowed_entity_classes(args.schema_file)

    input_files = sorted(args.input_dir.resolve().glob("*.md"))
    if args.max_docs > 0:
        input_files = input_files[: args.max_docs]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    if args.reset_output and args.output_file.exists():
        args.output_file.unlink()

    done_docs: set[str] = set()
    if args.output_file.exists():
        with args.output_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                doc_id = row.get("document_id")
                if doc_id:
                    done_docs.add(doc_id)

    processed = 0
    skipped = 0
    failures = 0
    total_extractions = 0
    class_counts: dict[str, int] = {}
    preflight_status = "skipped"

    api_url = os.getenv("API_URL")
    api_token = os.getenv("API_TOKEN")
    if not args.skip_preflight and api_url and api_token:
        ok, msg = preflight_inference(api_url, api_token, args.model_id)
        if not ok:
            raise SystemExit(msg)
        preflight_status = "ok"

    with args.output_file.open("a", encoding="utf-8") as out_f:
        for fp in input_files:
            doc_id = fp.stem
            if doc_id in done_docs:
                skipped += 1
                continue

            full_text = fp.read_text(encoding="utf-8", errors="ignore")
            selected_text = select_fabrication_text(full_text)
            if not selected_text.strip():
                skipped += 1
                continue

            hint = metadata_hints.get(doc_id, {})
            additional_context = (
                f"metadata_hint institution={hint.get('institution')} "
                f"department={hint.get('department')} "
                f"title={hint.get('title')} year={hint.get('year')}"
            )

            provider = (args.provider or "").strip() or None

            extract_kwargs: dict[str, Any] = {
                "text_or_documents": selected_text,
                "prompt_description": prompt,
                "examples": examples,
                "model_id": args.model_id,
                "max_char_buffer": args.max_char_buffer,
                "extraction_passes": args.extraction_passes,
                "max_workers": args.max_workers,
                "additional_context": additional_context,
                "prompt_validation_level": pv.PromptValidationLevel.WARNING,
                "temperature": args.temperature,
                "show_progress": False,
            }

            # Reuse existing project inference endpoint when available.
            if api_url and api_token:
                if provider is None:
                    provider = "OpenAILanguageModel"
                extract_kwargs["config"] = lx.factory.ModelConfig(
                    model_id=args.model_id,
                    provider=provider,
                    provider_kwargs={
                        "api_key": api_token,
                        "base_url": api_url,
                    },
                )
                extract_kwargs.pop("model_id", None)

            last_err = None
            res = None
            for attempt in range(args.max_attempts_per_doc):
                try:
                    attempt_kwargs = dict(extract_kwargs)
                    if attempt > 0:
                        attempt_kwargs["extraction_passes"] = 1
                        attempt_kwargs["max_char_buffer"] = min(args.max_char_buffer, 900)
                    res = lx.extract(**attempt_kwargs)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    continue
            if last_err is not None or res is None:
                failures += 1
                out_f.write(
                    json.dumps(
                        {
                            "document_id": doc_id,
                            "source_file": _safe_relpath(fp),
                            "status": "error",
                            "error": str(last_err),
                        }
                    )
                    + "\n"
                )
                continue

            annotated = res[0] if isinstance(res, list) else res
            doc_extractions = to_output_extractions(
                selected_text=selected_text,
                extractions=annotated.extractions or [],
                allowed_entity_classes=allowed_entity_classes,
                allow_ungrounded=args.allow_ungrounded,
            )

            for e in doc_extractions:
                c = e.get("extraction_class")
                if c:
                    class_counts[c] = class_counts.get(c, 0) + 1

            row = {
                "document_id": doc_id,
                "source_file": _safe_relpath(fp),
                "status": "ok",
                "model_id": args.model_id,
                "selected_text_chars": len(selected_text),
                "metadata_hint": hint,
                "extraction_count": len(doc_extractions),
                "extractions": doc_extractions,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            processed += 1
            total_extractions += len(doc_extractions)

    summary = {
        "input_dir": str(args.input_dir),
        "output_file": str(args.output_file),
        "model_id": args.model_id,
        "preflight": preflight_status,
        "total_files_considered": len(input_files),
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "total_extractions": total_extractions,
        "class_counts": class_counts,
    }
    args.summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
