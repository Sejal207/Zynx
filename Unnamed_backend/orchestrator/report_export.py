"""JSON / Markdown / PDF formatting for the Agent and Human Think-Aloud
reports, used by both the Save Report archive and the standalone Export
downloads.

This module never recomputes scores, re-classifies transcript segments, or
invents findings -- it only formats data that already exists in the Agent
report (scorecard/report/friction, produced by evaluator_agent.py,
reporter_agent.py and friction.py) or the Human report (the full
build_human_ux_report() output, produced by human_ux_report.py). Agent and
Human formatting are kept in clearly separate functions -- never a shared
generic template -- because the two report schemas are not the same shape.
"""
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

logger = logging.getLogger("ux-report-export")

# ============================================================================
# Fonts -- registered once. Arial covers Latin script (the primary evidence
# text everywhere: Romanized Hinglish display_transcript, English UI text).
# Nirmala (Windows' bundled Indic UI font) covers Devanagari, used ONLY for
# the optional raw_transcript field. If either registration fails (e.g. a
# non-Windows deployment without these files), formatting falls back to
# ReportLab's built-in Helvetica for Latin text, and Devanagari raw-transcript
# text is replaced with a safe, explicit placeholder rather than silently
# rendering missing-glyph boxes.
# ============================================================================
_FONTS_READY = False
BODY_FONT = "Helvetica"
BODY_FONT_BOLD = "Helvetica-Bold"
DEVANAGARI_FONT: Optional[str] = None
_HAS_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

_WIN_FONT_DIR = r"C:\Windows\Fonts"


def _ensure_fonts() -> None:
    global _FONTS_READY, BODY_FONT, BODY_FONT_BOLD, DEVANAGARI_FONT
    if _FONTS_READY:
        return
    _FONTS_READY = True

    arial = os.path.join(_WIN_FONT_DIR, "arial.ttf")
    arial_bold = os.path.join(_WIN_FONT_DIR, "arialbd.ttf")
    try:
        if os.path.exists(arial):
            pdfmetrics.registerFont(TTFont("Arial", arial))
            BODY_FONT = "Arial"
            if os.path.exists(arial_bold):
                pdfmetrics.registerFont(TTFont("Arial-Bold", arial_bold))
                BODY_FONT_BOLD = "Arial-Bold"
            else:
                BODY_FONT_BOLD = "Arial"
    except Exception:
        logger.warning("Could not register Arial TTF; falling back to core Helvetica.")

    nirmala = os.path.join(_WIN_FONT_DIR, "Nirmala.ttc")
    try:
        if os.path.exists(nirmala):
            pdfmetrics.registerFont(TTFont("Nirmala", nirmala, subfontIndex=0))
            DEVANAGARI_FONT = "Nirmala"
    except Exception:
        logger.warning("Could not register a Devanagari-capable font; raw-script "
                        "transcript text will be replaced with a safe placeholder.")
        DEVANAGARI_FONT = None


def safe_script_text(text: Optional[str]) -> str:
    """Returns text safe to render in BODY_FONT. If the text contains
    Devanagari and no Devanagari-capable font was registered, returns an
    explicit placeholder instead of letting missing glyphs render as boxes."""
    if not text:
        return ""
    if _HAS_DEVANAGARI.search(text) and not DEVANAGARI_FONT:
        return "[original-script transcript omitted -- no embeddable font available for this script]"
    return text


def _devanagari_safe_font(text: Optional[str]) -> str:
    """Which registered font can actually render this text."""
    if text and _HAS_DEVANAGARI.search(text) and DEVANAGARI_FONT:
        return DEVANAGARI_FONT
    return BODY_FONT


# ============================================================================
# Shared PDF styling / page chrome
# ============================================================================
NAVY = colors.HexColor("#0B1F3A")
CHARCOAL = colors.HexColor("#262B33")
GREY = colors.HexColor("#6B7280")
LIGHT_GREY = colors.HexColor("#E3E6EA")
ROW_ALT = colors.HexColor("#F5F6F8")
PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def _styles(font: str, font_bold: str) -> Dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName=font_bold, fontSize=18, leading=22,
                                 textColor=NAVY, spaceAfter=4, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", fontName=font, fontSize=10, leading=13,
                                    textColor=GREY, spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName=font_bold, fontSize=13, leading=16,
                              textColor=NAVY, spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=font_bold, fontSize=10.5, leading=13,
                              textColor=NAVY, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=font, fontSize=9.5, leading=14,
                                textColor=CHARCOAL, spaceAfter=6),
        "meta": ParagraphStyle("meta", fontName=font, fontSize=8.5, leading=12,
                                textColor=GREY, spaceAfter=3),
        "quote": ParagraphStyle("quote", fontName=font, fontSize=9.5, leading=13,
                                 textColor=CHARCOAL, leftIndent=10, spaceAfter=4),
        "label": ParagraphStyle("label", fontName=font_bold, fontSize=7.5, leading=10,
                                 textColor=NAVY, spaceAfter=2),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=8.5, leading=11, textColor=CHARCOAL),
        "cell_head": ParagraphStyle("cell_head", fontName=font_bold, fontSize=8.5,
                                     leading=11, textColor=colors.white),
        "cell_deva": ParagraphStyle("cell_deva", fontName=DEVANAGARI_FONT or font,
                                     fontSize=9, leading=13, textColor=CHARCOAL),
    }


def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.6, color=LIGHT_GREY, spaceBefore=2, spaceAfter=10)


def _table(rows: List[List[str]], col_widths: List[float], styles: Dict[str, ParagraphStyle],
           header: bool = True) -> Table:
    data = []
    for r_idx, row in enumerate(rows):
        style = styles["cell_head"] if (header and r_idx == 0) else styles["cell"]
        data.append([Paragraph(str(c) if c is not None else "—", style) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for i in range(1 if header else 0, len(data)):
        if (i - (1 if header else 0)) % 2 == 1:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def _page_chrome(title: str, mode_label: str, report_id: Optional[str]):
    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 8)
        canvas.setFillColor(GREY)
        canvas.drawString(MARGIN, PAGE_H - 12 * mm, f"ZYNX — {mode_label}")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, title)
        canvas.setStrokeColor(LIGHT_GREY)
        canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)

        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.drawString(MARGIN, 9 * mm, f"Report ID: {report_id or '(unsaved)'}")
        canvas.drawCentredString(PAGE_W / 2, 9 * mm, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, "Zynx UX Intelligence Platform")
        canvas.restoreState()
    return _draw


def _render_pdf(story: list, title: str, mode_label: str, report_id: Optional[str]) -> bytes:
    import io
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=18 * mm, bottomMargin=18 * mm,
        title=title,
    )
    chrome = _page_chrome(title, mode_label, report_id)
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    return buf.getvalue()


def _md_escape(text: Optional[str]) -> str:
    """Escapes characters that would otherwise corrupt Markdown table
    structure or be misread as formatting. Multilingual text (Devanagari,
    Romanized Hinglish, punctuation) passes through untouched -- only
    Markdown-significant ASCII characters are escaped."""
    if not text:
        return ""
    text = str(text)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("`", "'")
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    total = max(0, round(ms / 1000))
    return f"{total // 60:02d}:{total % 60:02d}"


# ============================================================================
# AGENT report -- JSON
# ============================================================================

def agent_report_to_dict(payload: Dict[str, Any], report_id: Optional[str], generated_at: str) -> Dict[str, Any]:
    """Preserves exactly the fields already present in the Agent evaluation's
    own data (scorecard from evaluator_agent.py, report from
    reporter_agent.py, friction from friction.py) -- nothing invented."""
    return {
        "report_id": report_id,
        "mode": "agent",
        "generated_at": generated_at,
        "run_id": payload.get("run_id"),
        "target_url": payload.get("target_url"),
        "task_description": payload.get("task_description"),
        "persona": payload.get("persona"),
        "task_completed": payload.get("task_completed"),
        "scorecard": payload.get("scorecard") or {},
        "report": payload.get("report") or {},
        "friction": payload.get("friction"),
    }


# ============================================================================
# AGENT report -- Markdown
# ============================================================================

def agent_report_to_markdown(payload: Dict[str, Any], report_id: Optional[str], generated_at: str) -> str:
    scorecard = payload.get("scorecard") or {}
    report = payload.get("report") or {}
    friction = payload.get("friction") or {}
    persona = payload.get("persona") or {}

    lines: List[str] = []
    lines.append("# ZYNX — Autonomous Agent UX Evaluation Report")
    lines.append("")
    lines.append(f"**Report ID:** {report_id or '(unsaved)'}  ")
    lines.append(f"**Generated:** {generated_at}  ")
    lines.append(f"**Mode:** Autonomous Agent  ")
    if payload.get("run_id"):
        lines.append(f"**Run ID:** {payload['run_id']}  ")
    lines.append("")

    lines.append("## 1. Evaluation Overview")
    lines.append(f"- **Website:** {payload.get('target_url', 'N/A')}")
    lines.append(f"- **Task:** {payload.get('task_description', 'N/A')}")
    outcome = payload.get("task_completed")
    outcome_text = "Completed" if outcome is True else ("Not completed" if outcome is False else "Unknown")
    lines.append(f"- **Task Outcome:** {outcome_text}")
    if scorecard.get("overall_ux_score") is not None:
        lines.append(f"- **Overall UX Score:** {scorecard['overall_ux_score']}/10")
    lines.append("")

    if persona:
        lines.append("## 2. Website, Persona, and Task")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for k, v in persona.items():
            if v not in (None, ""):
                lines.append(f"| {_md_escape(k.replace('_', ' ').title())} | {_md_escape(v)} |")
        lines.append("")

    dims = [
        ("Effectiveness", scorecard.get("effectiveness")),
        ("Efficiency", scorecard.get("efficiency")),
        ("Learnability", scorecard.get("learnability")),
        ("Cognitive Load", scorecard.get("cognitive_load")),
    ]
    if any(d for _, d in dims):
        lines.append("## 4. Existing UX Scorecard")
        lines.append("| Dimension | Score | Rationale |")
        lines.append("|---|---|---|")
        for label, dim in dims:
            if dim:
                lines.append(f"| {label} | {dim.get('score', '—')}/10 | {_md_escape(dim.get('rationale', ''))} |")
        if scorecard.get("system1_system2_ratio"):
            s = scorecard["system1_system2_ratio"]
            lines.append(f"| System 1 / System 2 | {s.get('system1_percent', '—')}% / {s.get('system2_percent', '—')}% | {_md_escape(s.get('rationale', ''))} |")
        lines.append("")
        if scorecard.get("psychological_biases"):
            lines.append("### Psychological Bias Analysis")
            lines.append("| Bias | Score |")
            lines.append("|---|---|")
            for k, v in scorecard["psychological_biases"].items():
                lines.append(f"| {_md_escape(k.replace('_', ' ').title())} | {v}/10 |")
            lines.append("")

    if report.get("executive_summary") or report.get("psychological_friction_analysis"):
        lines.append("## 5. Key UX Findings")
        if report.get("executive_summary"):
            lines.append(f"**Executive Summary:** {report['executive_summary']}")
            lines.append("")
        if report.get("psychological_friction_analysis"):
            lines.append(f"**Psychological Friction Analysis:** {report['psychological_friction_analysis']}")
            lines.append("")
        if scorecard.get("top_positives"):
            lines.append("**Top Positives:**")
            for p in scorecard["top_positives"]:
                lines.append(f"- {p}")
            lines.append("")
        if scorecard.get("top_issues"):
            lines.append("**Top Issues:**")
            for p in scorecard["top_issues"]:
                lines.append(f"- {p}")
            lines.append("")

    episodes = friction.get("episodes") or []
    if episodes:
        lines.append("## 6. Behavioural Friction Evidence")
        lines.append(f"{friction.get('episode_count', len(episodes))} episode(s) detected deterministically from timestamped browser events.")
        lines.append("")
        lines.append("| Time Range | Severity | Signal Types | Summary |")
        lines.append("|---|---|---|---|")
        for ep in episodes:
            trange = f"{_fmt_ms(ep.get('start_time_ms'))}–{_fmt_ms(ep.get('end_time_ms'))}"
            sig_types = ", ".join(ep.get("signal_types", []))
            summary = "; ".join(ep.get("evidence_summary", []))
            lines.append(f"| {trange} | {ep.get('severity', '—')} | {_md_escape(sig_types)} | {_md_escape(summary)} |")
        lines.append("")

    recs = report.get("key_recommendations") or []
    if recs:
        lines.append("## 7. Recommendations")
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. **[{r.get('priority', 'n/a').upper()}] {_md_escape(r.get('title', ''))}** — {_md_escape(r.get('detail', ''))}")
        lines.append("")

    friction_points = scorecard.get("friction_points") or []
    if friction_points:
        lines.append("## 8. Supporting Evidence")
        lines.append("Evaluator-identified friction points (distinct from the deterministic Behavioural Friction Evidence above):")
        lines.append("")
        lines.append("| Step | Type | Severity | Description |")
        lines.append("|---|---|---|---|")
        for fp in friction_points:
            lines.append(f"| {fp.get('step', '—')} | {_md_escape(fp.get('type', ''))} | {fp.get('severity', '—')} | {_md_escape(fp.get('description', ''))} |")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# AGENT report -- PDF
# ============================================================================

def agent_report_to_pdf(payload: Dict[str, Any], report_id: Optional[str], generated_at: str) -> bytes:
    _ensure_fonts()
    S = _styles(BODY_FONT, BODY_FONT_BOLD)
    scorecard = payload.get("scorecard") or {}
    report = payload.get("report") or {}
    friction = payload.get("friction") or {}
    persona = payload.get("persona") or {}
    story: list = []

    story.append(Paragraph("ZYNX — Autonomous Agent UX Evaluation Report", S["title"]))
    story.append(Paragraph(
        f"Generated {generated_at} &nbsp;·&nbsp; Report ID {report_id or '(unsaved)'}"
        + (f" &nbsp;·&nbsp; Run ID {payload['run_id']}" if payload.get("run_id") else ""),
        S["subtitle"]))
    story.append(_hr())

    story.append(Paragraph("1. Evaluation Overview", S["h1"]))
    outcome = payload.get("task_completed")
    outcome_text = "Completed" if outcome is True else ("Not completed" if outcome is False else "Unknown")
    overview_rows = [["Field", "Value"],
                      ["Website", payload.get("target_url", "N/A")],
                      ["Task", payload.get("task_description", "N/A")],
                      ["Task Outcome", outcome_text]]
    if scorecard.get("overall_ux_score") is not None:
        overview_rows.append(["Overall UX Score", f"{scorecard['overall_ux_score']}/10"])
    story.append(_table(overview_rows, [40 * mm, 120 * mm], S))

    if persona:
        story.append(Paragraph("2. Website, Persona, and Task", S["h1"]))
        p_rows = [["Field", "Value"]] + [
            [k.replace("_", " ").title(), str(v)] for k, v in persona.items() if v not in (None, "")
        ]
        if len(p_rows) > 1:
            story.append(_table(p_rows, [50 * mm, 110 * mm], S))

    dims = [
        ("Effectiveness", scorecard.get("effectiveness")),
        ("Efficiency", scorecard.get("efficiency")),
        ("Learnability", scorecard.get("learnability")),
        ("Cognitive Load", scorecard.get("cognitive_load")),
    ]
    if any(d for _, d in dims):
        story.append(Paragraph("4. Existing UX Scorecard", S["h1"]))
        rows = [["Dimension", "Score", "Rationale"]]
        for label, dim in dims:
            if dim:
                rows.append([label, f"{dim.get('score', '—')}/10", dim.get("rationale", "")])
        if scorecard.get("system1_system2_ratio"):
            s = scorecard["system1_system2_ratio"]
            rows.append(["System 1 / System 2",
                         f"{s.get('system1_percent', '—')}% / {s.get('system2_percent', '—')}%",
                         s.get("rationale", "")])
        story.append(_table(rows, [34 * mm, 24 * mm, 112 * mm], S))

        if scorecard.get("psychological_biases"):
            story.append(Paragraph("Psychological Bias Analysis", S["h2"]))
            b_rows = [["Bias", "Score"]] + [
                [k.replace("_", " ").title(), f"{v}/10"] for k, v in scorecard["psychological_biases"].items()
            ]
            story.append(_table(b_rows, [120 * mm, 50 * mm], S))

    if report.get("executive_summary") or report.get("psychological_friction_analysis"):
        story.append(Paragraph("5. Key UX Findings", S["h1"]))
        if report.get("executive_summary"):
            story.append(Paragraph("<b>Executive Summary</b>", S["h2"]))
            story.append(Paragraph(report["executive_summary"], S["body"]))
        if report.get("psychological_friction_analysis"):
            story.append(Paragraph("<b>Psychological Friction Analysis</b>", S["h2"]))
            story.append(Paragraph(report["psychological_friction_analysis"], S["body"]))
        if scorecard.get("top_positives"):
            story.append(Paragraph("<b>Top Positives</b>", S["h2"]))
            for p in scorecard["top_positives"]:
                story.append(Paragraph(f"• {p}", S["body"]))
        if scorecard.get("top_issues"):
            story.append(Paragraph("<b>Top Issues</b>", S["h2"]))
            for p in scorecard["top_issues"]:
                story.append(Paragraph(f"• {p}", S["body"]))

    episodes = friction.get("episodes") or []
    if episodes:
        story.append(Paragraph("6. Behavioural Friction Evidence", S["h1"]))
        story.append(Paragraph(
            f"{friction.get('episode_count', len(episodes))} episode(s) detected deterministically "
            "from timestamped browser events (no LLM involved in detection).", S["body"]))
        rows = [["Time Range", "Severity", "Signal Types", "Summary"]]
        for ep in episodes:
            trange = f"{_fmt_ms(ep.get('start_time_ms'))}–{_fmt_ms(ep.get('end_time_ms'))}"
            rows.append([trange, ep.get("severity", "—"),
                         ", ".join(ep.get("signal_types", [])),
                         "; ".join(ep.get("evidence_summary", []))])
        story.append(_table(rows, [26 * mm, 20 * mm, 40 * mm, 74 * mm], S))

    recs = report.get("key_recommendations") or []
    if recs:
        story.append(Paragraph("7. Recommendations", S["h1"]))
        for r in recs:
            story.append(Paragraph(
                f"<b>[{r.get('priority', 'n/a').upper()}] {r.get('title', '')}</b> — {r.get('detail', '')}",
                S["body"]))

    friction_points = scorecard.get("friction_points") or []
    if friction_points:
        story.append(Paragraph("8. Supporting Evidence", S["h1"]))
        story.append(Paragraph(
            "Evaluator-identified friction points (distinct from the deterministic "
            "Behavioural Friction Evidence above).", S["body"]))
        rows = [["Step", "Type", "Severity", "Description"]]
        for fp in friction_points:
            rows.append([str(fp.get("step", "—")), fp.get("type", ""), fp.get("severity", "—"), fp.get("description", "")])
        story.append(_table(rows, [16 * mm, 28 * mm, 20 * mm, 96 * mm], S))

    return _render_pdf(story, "Autonomous Agent Report", "Autonomous Agent UX Evaluation", report_id)


# ============================================================================
# HUMAN THINK-ALOUD report -- JSON
# ============================================================================

def think_aloud_report_to_dict(
    report: Dict[str, Any], segments: List[Dict[str, Any]],
    researcher_notes: Optional[str], task_outcome_note: Optional[str],
    report_id: Optional[str], generated_at: str,
) -> Dict[str, Any]:
    """`report` is the exact output of human_ux_report.build_human_ux_report()
    (session/task_outcome/score/signal_summary/timeline/findings/
    expectation_experience/summary). `segments` is the full transcript
    segment list (raw_transcript, display_transcript, classification,
    timing_source, confidence) -- these are never recomputed here."""
    clean_segments = [
        {
            "segment_id": s.get("segment_id"),
            "raw_transcript": s.get("raw_transcript"),
            "display_transcript": s.get("display_transcript"),
            "start_time_ms": s.get("start_time_ms"),
            "end_time_ms": s.get("end_time_ms"),
            "timing_source": s.get("timing_source"),
            "language": s.get("language"),
            "confidence": s.get("confidence"),
            "classification": s.get("classification"),
        }
        for s in (segments or [])
    ]
    return {
        "report_id": report_id,
        "mode": "think_aloud",
        "generated_at": generated_at,
        "session": report.get("session"),
        "task_outcome": report.get("task_outcome"),
        "task_outcome_note": task_outcome_note,
        "researcher_notes": researcher_notes,
        "score": report.get("score"),
        "signal_summary": report.get("signal_summary"),
        "timeline": report.get("timeline"),
        "findings": report.get("findings"),
        "expectation_experience": report.get("expectation_experience"),
        "summary": report.get("summary"),
        "segments": clean_segments,
    }


# ============================================================================
# HUMAN THINK-ALOUD report -- Markdown
# ============================================================================

def think_aloud_report_to_markdown(
    report: Dict[str, Any], segments: List[Dict[str, Any]],
    researcher_notes: Optional[str], task_outcome_note: Optional[str],
    report_id: Optional[str], generated_at: str,
) -> str:
    session = report.get("session") or {}
    score = report.get("score") or {}
    dims = score.get("dimensions") or {}

    lines: List[str] = []
    lines.append("# ZYNX — Human Think-Aloud UX Evaluation Report")
    lines.append("")
    lines.append(f"**Report ID:** {report_id or '(unsaved)'}  ")
    lines.append(f"**Generated:** {generated_at}  ")
    lines.append("**Mode:** Human + Think-Aloud  ")
    lines.append("")

    # 1. Session and Task Overview
    lines.append("## 1. Session and Task Overview")
    lines.append(f"- **Website:** {session.get('target_url', 'N/A')}")
    lines.append(f"- **Task:** {session.get('task_description', 'N/A')}")
    if session.get("persona"):
        persona_str = "; ".join(f"{k}: {v}" for k, v in session["persona"].items() if v)
        lines.append(f"- **Persona:** {_md_escape(persona_str)}")
    lines.append(f"- **Language:** {session.get('language', 'unknown')}")
    lines.append(f"- **Audio Duration:** {_fmt_ms(session.get('audio_duration_ms'))}")
    lines.append(f"- **Segment Count:** {session.get('segment_count', 0)}")
    lines.append(f"- **Task Outcome:** {report.get('task_outcome', 'not_assessed')}")
    lines.append("")

    # 2 & 3. Task UX Score + Breakdown
    lines.append("## 2. Task UX Score")
    if score.get("fully_assessed"):
        lines.append(f"**{score.get('total')} / {score.get('max', 100)}**")
    else:
        lines.append(f"**Status:** {score.get('status', 'Not fully assessed')}")
    lines.append("")
    lines.append(f"_{score.get('disclaimer', '')}_")
    lines.append("")
    lines.append("## 3. Score Breakdown")
    lines.append("| Dimension | Score | Max |")
    lines.append("|---|---|---|")
    dim_labels = [
        ("task_completion", "Task Completion"), ("verbal_ux_signals", "Verbal UX Signals"),
        ("recovery", "Recovery"), ("evidence_completeness", "Evidence Completeness"),
    ]
    for key, label in dim_labels:
        d = dims.get(key) or {}
        score_val = d.get("score")
        lines.append(f"| {label} | {score_val if score_val is not None else (d.get('status') or 'Not fully assessed')} | {d.get('max', '—')} |")
    lines.append("")

    # 4. Signal Summary
    signal_summary = report.get("signal_summary") or {}
    if signal_summary:
        lines.append("## 4. Think-Aloud Signal Summary")
        lines.append("| Label | Count |")
        lines.append("|---|---|")
        for label, info in signal_summary.items():
            lines.append(f"| {label} | {info.get('count', 0)} |")
        lines.append("")

    # 5. Evidence Timeline
    timeline = report.get("timeline") or []
    if timeline:
        lines.append("## 5. Evidence Timeline")
        lines.append("| Time | Labels | Timing Source | Evidence |")
        lines.append("|---|---|---|---|")
        for t in timeline:
            trange = (f"{_fmt_ms(t.get('start_time_ms'))}–{_fmt_ms(t.get('end_time_ms'))}"
                      if t.get("start_time_ms") is not None else "untimed")
            lines.append(f"| {trange} | {', '.join(t.get('labels', []))} | {t.get('timing_source') or '—'} | \"{_md_escape(t.get('text', ''))}\" |")
        lines.append("")

    # 6. UX Findings (includes summary + expectation/experience, matching the
    # frontend's own merged "UX Findings" section)
    exp_exp = report.get("expectation_experience") or {}
    if report.get("summary") or any(exp_exp.values()):
        lines.append("## 6. UX Findings")
        if report.get("summary"):
            lines.append(report["summary"])
            lines.append("")
        slot_labels = [
            ("expectation", "Participant expected"),
            ("observed_experience", "Participant reported"),
            ("what_happened_next", "Participant considered"),
            ("confirmation", "Participant later confirmed"),
        ]
        for key, lead in slot_labels:
            v = exp_exp.get(key)
            if v:
                lines.append(f"- **{lead}:** \"{_md_escape(v.get('text', ''))}\"" +
                              (f" ({_fmt_ms(v.get('start_time_ms'))})" if v.get("start_time_ms") is not None else ""))
        lines.append("")

    # 7. Task Outcome and Researcher Notes
    lines.append("## 7. Task Outcome and Researcher Notes")
    lines.append(f"**Task Outcome:** {report.get('task_outcome', 'not_assessed')}")
    if task_outcome_note:
        lines.append(f"**Task Outcome Note:** {_md_escape(task_outcome_note)}")
    if researcher_notes:
        lines.append(f"**Researcher Notes:** {_md_escape(researcher_notes)}")
    lines.append("")

    # 8. Full Transcript
    if segments:
        lines.append("## 8. Full Think-Aloud Transcript")
        lines.append("| Time | Timing Source | Display (Romanized) | Raw | Labels |")
        lines.append("|---|---|---|---|---|")
        for s in sorted(segments, key=lambda x: (x.get("start_time_ms") is None, x.get("start_time_ms") or 0)):
            trange = (f"{_fmt_ms(s.get('start_time_ms'))}–{_fmt_ms(s.get('end_time_ms'))}"
                      if s.get("start_time_ms") is not None else "untimed")
            labels = ", ".join((s.get("classification") or {}).get("labels", []) or [])
            lines.append(
                f"| {trange} | {s.get('timing_source') or '—'} | {_md_escape(s.get('display_transcript') or s.get('transcript'))} "
                f"| {_md_escape(s.get('raw_transcript'))} | {labels} |"
            )
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# HUMAN THINK-ALOUD report -- PDF
# ============================================================================

def think_aloud_report_to_pdf(
    report: Dict[str, Any], segments: List[Dict[str, Any]],
    researcher_notes: Optional[str], task_outcome_note: Optional[str],
    report_id: Optional[str], generated_at: str,
) -> bytes:
    _ensure_fonts()
    S = _styles(BODY_FONT, BODY_FONT_BOLD)
    session = report.get("session") or {}
    score = report.get("score") or {}
    dims = score.get("dimensions") or {}
    story: list = []

    story.append(Paragraph("ZYNX — Human Think-Aloud UX Evaluation Report", S["title"]))
    story.append(Paragraph(f"Generated {generated_at} &nbsp;·&nbsp; Report ID {report_id or '(unsaved)'}", S["subtitle"]))
    story.append(_hr())

    # 1. Session and Task Overview
    story.append(Paragraph("1. Session and Task Overview", S["h1"]))
    persona_str = "; ".join(f"{k}: {v}" for k, v in (session.get("persona") or {}).items() if v) or "—"
    rows = [
        ["Field", "Value"],
        ["Website", session.get("target_url", "N/A")],
        ["Task", session.get("task_description", "N/A")],
        ["Persona", persona_str],
        ["Language", session.get("language") or "unknown"],
        ["Audio Duration", _fmt_ms(session.get("audio_duration_ms"))],
        ["Segment Count", str(session.get("segment_count", 0))],
        ["Task Outcome", report.get("task_outcome", "not_assessed")],
    ]
    story.append(_table(rows, [40 * mm, 120 * mm], S))

    # 2. Task UX Score
    story.append(Paragraph("2. Task UX Score", S["h1"]))
    if score.get("fully_assessed"):
        story.append(Paragraph(f'<font size="20" color="#0B1F3A"><b>{score.get("total")} / {score.get("max", 100)}</b></font>', S["body"]))
    else:
        story.append(Paragraph(f'<b>{score.get("status", "Not fully assessed")}</b>', S["body"]))
    story.append(Paragraph(score.get("disclaimer", ""), S["meta"]))

    # 3. Score Breakdown
    story.append(Paragraph("3. Score Breakdown", S["h1"]))
    dim_labels = [
        ("task_completion", "Task Completion"), ("verbal_ux_signals", "Verbal UX Signals"),
        ("recovery", "Recovery"), ("evidence_completeness", "Evidence Completeness"),
    ]
    rows = [["Dimension", "Score", "Max"]]
    for key, label in dim_labels:
        d = dims.get(key) or {}
        score_val = d.get("score")
        rows.append([label, str(score_val) if score_val is not None else (d.get("status") or "Not fully assessed"), str(d.get("max", "—"))])
    story.append(_table(rows, [80 * mm, 50 * mm, 30 * mm], S))

    # 4. Signal Summary
    signal_summary = report.get("signal_summary") or {}
    if signal_summary:
        story.append(Paragraph("4. Think-Aloud Signal Summary", S["h1"]))
        rows = [["Label", "Count"]] + [[label, str(info.get("count", 0))] for label, info in signal_summary.items()]
        story.append(_table(rows, [120 * mm, 40 * mm], S))

    # 5. Evidence Timeline
    timeline = report.get("timeline") or []
    if timeline:
        story.append(Paragraph("5. Evidence Timeline", S["h1"]))
        rows = [["Time", "Labels", "Source", "Evidence"]]
        for t in timeline:
            trange = (f"{_fmt_ms(t.get('start_time_ms'))}–{_fmt_ms(t.get('end_time_ms'))}"
                      if t.get("start_time_ms") is not None else "untimed")
            rows.append([trange, ", ".join(t.get("labels", [])), t.get("timing_source") or "—", t.get("text", "")])
        story.append(_table(rows, [22 * mm, 30 * mm, 20 * mm, 88 * mm], S))

    # 6. UX Findings
    exp_exp = report.get("expectation_experience") or {}
    if report.get("summary") or any(exp_exp.values()):
        story.append(Paragraph("6. UX Findings", S["h1"]))
        if report.get("summary"):
            story.append(Paragraph(report["summary"], S["body"]))
        slot_labels = [
            ("expectation", "Participant expected"),
            ("observed_experience", "Participant reported"),
            ("what_happened_next", "Participant considered"),
            ("confirmation", "Participant later confirmed"),
        ]
        for key, lead in slot_labels:
            v = exp_exp.get(key)
            if v:
                suffix = f" · {_fmt_ms(v.get('start_time_ms'))}" if v.get("start_time_ms") is not None else ""
                story.append(Paragraph(f"<b>{lead}:</b> \u201c{v.get('text', '')}\u201d{suffix}", S["quote"]))

    # 7. Task Outcome and Researcher Notes
    story.append(Paragraph("7. Task Outcome and Researcher Notes", S["h1"]))
    story.append(Paragraph(f"<b>Task Outcome:</b> {report.get('task_outcome', 'not_assessed')}", S["body"]))
    if task_outcome_note:
        story.append(Paragraph(f"<b>Task Outcome Note:</b> {task_outcome_note}", S["body"]))
    if researcher_notes:
        story.append(Paragraph(f"<b>Researcher Notes:</b> {researcher_notes}", S["body"]))

    # 8. Full Transcript -- may span multiple pages; table header repeats
    # automatically via repeatRows=1 in _table().
    if segments:
        story.append(Paragraph("8. Full Think-Aloud Transcript", S["h1"]))
        ordered = sorted(segments, key=lambda x: (x.get("start_time_ms") is None, x.get("start_time_ms") or 0))
        rows = [["Time", "Transcript", "Labels"]]
        for s in ordered:
            trange = (f"{_fmt_ms(s.get('start_time_ms'))}–{_fmt_ms(s.get('end_time_ms'))}"
                      if s.get("start_time_ms") is not None else "untimed")
            if s.get("timing_source") == "derived":
                trange += " (derived)"
            text = s.get("display_transcript") or s.get("transcript") or ""
            labels = ", ".join((s.get("classification") or {}).get("labels", []) or [])
            rows.append([trange, text, labels])
        story.append(_table(rows, [26 * mm, 104 * mm, 30 * mm], S))

        raw_present = [s for s in ordered if s.get("raw_transcript")]
        if raw_present:
            story.append(Paragraph("Original-script transcript (raw, as returned by Sarvam STT)", S["h2"]))
            col_widths = [26 * mm, 134 * mm]
            header_row = [Paragraph("Time", S["cell_head"]), Paragraph("Raw Transcript", S["cell_head"])]
            table_rows = [header_row]
            for s in raw_present:
                trange = (f"{_fmt_ms(s.get('start_time_ms'))}–{_fmt_ms(s.get('end_time_ms'))}"
                          if s.get("start_time_ms") is not None else "untimed")
                raw_original = s.get("raw_transcript")
                # Render in the Devanagari-capable font when the text actually
                # needs it and that font registered successfully; otherwise
                # fall back to a safe placeholder rather than risking
                # missing-glyph boxes in the body font.
                if _devanagari_safe_font(raw_original) == DEVANAGARI_FONT and DEVANAGARI_FONT:
                    cell_para = Paragraph(raw_original, S["cell_deva"])
                else:
                    cell_para = Paragraph(safe_script_text(raw_original), S["cell"])
                table_rows.append([Paragraph(trange, S["cell"]), cell_para])
            raw_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
            raw_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ]))
            story.append(raw_table)

    return _render_pdf(story, "Human Think-Aloud Report", "Human Think-Aloud UX Evaluation", report_id)
