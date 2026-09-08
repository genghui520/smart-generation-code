"""Fill currently measurable SMP paper metrics into a copy of the PDF.

The source manuscript is not available, so this script overlays only the cells
supported by ``evaluation/offline_experiment_results.json`` and appends an
audit note. The original PDF is never modified.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = letter


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def draw_replacement(
    page: canvas.Canvas,
    *,
    center_x: float,
    top: float,
    bottom: float,
    width: float,
    text: str,
    font_size: float,
) -> None:
    """White out one table cell value and draw centered replacement text."""
    y0 = PAGE_HEIGHT - bottom - 0.8
    height = bottom - top + 1.8
    page.setFillColor(white)
    page.rect(center_x - width / 2, y0, width, height, stroke=0, fill=1)
    page.setFillColor(black)
    page.setFont("Helvetica", font_size)
    page.drawCentredString(center_x, PAGE_HEIGHT - bottom + 0.6, text)


def make_metrics_overlay(metrics: dict[str, Any], baselines: dict[str, Any]) -> PdfReader:
    aggregate = metrics["aggregate"]
    task = pct(aggregate["scenario_task_success"]["rate"])
    compile_rate = pct(aggregate["compile_success"]["rate"])
    mean_coverage = pct(aggregate["per_run_api_coverage"]["mean"])
    generation_time = f"{aggregate['generation_time_seconds']['mean']:.1f}"
    union_coverage = pct(aggregate["dataset_union_api_coverage"]["rate"])
    discriminability = pct(aggregate["discriminability_proxy"]["rate"]) + "*"
    semantic = pct(aggregate["semantic_annotation_completeness_proxy"]["rate"]) + "*"

    stream = io.BytesIO()
    page = canvas.Canvas(stream, pagesize=letter)

    # Table 1, Coordinate Motion baseline rows. These are one-run code-only
    # evaluations; the audit note states the sample size and failure policy.
    baseline_by_method = {row["method"]: row for row in baselines["rows"]}
    row_tops = {"GPT-Engineer": 104.37, "MetaGPT": 110.29, "ChatDev 1.0": 116.22}
    for method, top in row_tops.items():
        row = baseline_by_method[method]
        values = [
            "100%" if row["task_success"] else "0%",
            "100%" if row["compile_success"] else "0%",
            pct(row["api_coverage"]),
            f"{row['time_seconds']:.1f}",
            str(row["total_tokens"]),
        ]
        for center, value, width in zip(
            (425.84, 461.15, 494.88, 522.11, 547.16), values, (31, 31, 28, 23, 26)
        ):
            draw_replacement(
                page,
                center_x=center,
                top=top,
                bottom=top + 5.18,
                width=width,
                text=value,
                font_size=4.7,
            )

    # Table 1, Coordinate Motion / SMPAgent row (page 4, top=122.14 pt).
    for center, value, width in [
        (425.84, task, 31.0),
        (461.15, compile_rate, 31.0),
        (494.88, mean_coverage, 28.0),
        (522.11, generation_time, 23.0),
    ]:
        draw_replacement(
            page,
            center_x=center,
            top=122.14,
            bottom=127.32,
            width=width,
            text=value,
            font_size=5.0,
        )

    # Table 2, FOCAS row (page 4, top=292.65 pt).
    for center, value, width in [
        (384.99, union_coverage, 42.0),
        (440.55, discriminability, 47.0),
        (516.44, semantic, 48.0),
    ]:
        draw_replacement(
            page,
            center_x=center,
            top=292.65,
            bottom=300.71,
            width=width,
            text=value,
            font_size=7.2,
        )

    # Asterisk note between Table 2 and the RQ3 heading.
    page.setFillColor(black)
    page.setFont("Helvetica", 4.8)
    page.drawString(
        320.0,
        PAGE_HEIGHT - 339.0,
        "* Offline proxy metric; definitions and limitations are provided in the appended audit note.",
    )

    # Table 3, SMPAgent row (page 4, top=500.69 pt).
    for center, value, width in [
        (414.63, task, 33.0),
        (454.50, mean_coverage, 31.0),
        (491.14, generation_time, 28.0),
    ]:
        draw_replacement(
            page,
            center_x=center,
            top=500.69,
            bottom=507.67,
            width=width,
            text=value,
            font_size=6.5,
        )

    page.save()
    stream.seek(0)
    return PdfReader(stream)


def draw_wrapped(
    page: canvas.Canvas,
    text: str,
    *,
    x: float,
    y: float,
    max_width: float,
    font: str = "Times-Roman",
    size: float = 9.5,
    leading: float = 13.0,
) -> float:
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if page.stringWidth(candidate, font, size) <= max_width:
            line = candidate
            continue
        page.setFont(font, size)
        page.drawString(x, y, line)
        y -= leading
        line = word
    if line:
        page.setFont(font, size)
        page.drawString(x, y, line)
        y -= leading
    return y


def make_audit_note(metrics: dict[str, Any], baselines: dict[str, Any]) -> PdfReader:
    aggregate = metrics["aggregate"]
    stream = io.BytesIO()
    page = canvas.Canvas(stream, pagesize=letter)
    margin = 54.0
    width = PAGE_WIDTH - 2 * margin
    y = PAGE_HEIGHT - 54.0

    page.setFont("Times-Bold", 16)
    page.drawString(margin, y, "Offline Evaluation Note for Tables 1-3")
    y -= 24
    page.setFont("Times-Roman", 9.5)
    page.drawString(margin, y, "Evaluation date: 2 September 2026")
    y -= 24

    paragraphs = [
        "Scope. The reported values were recomputed from six existing FOCAS coordinate-motion run directories (run_001-run_006). No simulator was started and no new communication traffic was generated for this evaluation.",
        "Experimental unit. Each run directory is treated as one repeated system run. These historical runs were collected during iterative development rather than through a preregistered randomized benchmark; therefore, the results are descriptive and no inferential test is reported.",
        "Task Success. A run is successful when an executable was compiled, the NC program completed, position changed, feed changed and included a non-zero value, and run=3/motion=1 was observed. Three of six runs met this criterion (50.0%).",
        "Compilation Success. All six runs contain a non-empty generated executable and no compiler error output (6/6, 100.0%).",
        "API Coverage. Per-run coverage is the number of distinct planned exact cnc_* functions with at least one EW_OK return divided by the number of distinct planned exact cnc_* functions. Mean coverage was 57.8% +/- 19.1% (n=6). The union-level FOCAS dataset coverage shown in Table 2 was 14/69 (20.3%).",
        "Generation Time. PlanningAgent and CodeGenerationAgent durations were summed while ExecutionAgent time was excluded. The mean was 463.4 +/- 154.5 s across the five runs with timing metadata (n=5).",
        "Discriminability proxy. Among state, position, and feed response fields with at least two observations, 11/12 fields changed (91.7%). This is an offline response-field proxy, not packet-byte discriminability.",
        "Semantic Consistency proxy. All 4,943 FOCAS log rows contained a step identifier, phase, input object, return code, response function/data, and a non-empty semantic label (100.0%). This measures annotation completeness, not packet-to-operation alignment accuracy.",
        "Framework baselines. GPT-Engineer, MetaGPT, and ChatDev 1.0 were each run once on the same coordinate-motion prompt with gpt-5.6-sol, temperature 0.1, code-generation-only mode, the official 32-bit FANUC header/library, and the same scorer. GPT-Engineer: task 0%, compilation 0%, API coverage 100%, 199.6 s, 8,914 tokens. MetaGPT: task 0%, compilation 0%, API coverage 100%, 210.5 s, 28,237 tokens. ChatDev 1.0: task 100%, compilation 100%, API coverage 100%, 116.3 s, 14,263 tokens. These are n=1 descriptive results, so no uncertainty estimate or significance claim is made.",
        "Failure policy. GPT-Engineer produced a source file but used incompatible FOCAS argument types/fields. MetaGPT's Engineer2 produced source text, but the framework rejected it as a command-protocol violation (KeyError: command_name) before writing the file; the exact serialized response was extracted for diagnosis and also failed compilation because cnc_actf and cnc_absolute used incompatible prototypes. Therefore both are task failures. ChatDev 2.0 was excluded because the cited paper baseline corresponds to ChatDev 1.0.",
        "Matched SMPAgent rerun. Two fresh code-only attempts using the same prompt, gpt-5.6-sol, and no simulator ended before any planner output because the upstream provider returned HTTP 524 after 120 seconds. These attempts are treated as infrastructure-blocked and are not inserted into Table 1. The SMPAgent row therefore remains the pre-existing six-run descriptive result and is not a directly matched n=1 comparison.",
        "Unavailable results. Single-Agent, w/o RouterAgent, Program Management, Tool Management, EZSocket, LSV/2, S7Comm, and RQ4 field-recovery accuracy remain TBD because no corresponding artifacts or ground truth are present.",
    ]
    for paragraph in paragraphs:
        y = draw_wrapped(page, paragraph, x=margin, y=y, max_width=width)
        y -= 8

    page.setFont("Times-Italic", 8.5)
    page.drawString(
        margin,
        38,
        "Generated from evaluation/offline_experiment_results.json; original SMP.pdf preserved.",
    )
    page.save()
    stream.seek(0)
    return PdfReader(stream)


def build_pdf(source: Path, metrics_path: Path, baselines_path: Path, output: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    source_reader = PdfReader(source)
    if len(source_reader.pages) < 4:
        raise ValueError("Expected SMP.pdf to contain at least four pages")

    overlay = make_metrics_overlay(metrics, baselines).pages[0]
    note = make_audit_note(metrics, baselines).pages[0]
    writer = PdfWriter()
    for index, source_page in enumerate(source_reader.pages):
        if index == 3:
            source_page.merge_page(overlay)
        writer.add_page(source_page)
    writer.add_page(note)
    writer.add_metadata(
        {
            "/Title": "SMP - current offline experiment metrics",
            "/Subject": "Existing-artifact evaluation; no new simulator traffic",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("evaluation/offline_experiment_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/SMP_metrics_filled.pdf"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("evaluation/baseline_comparison.json"),
    )
    args = parser.parse_args()
    build_pdf(args.source, args.metrics, args.baselines, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
