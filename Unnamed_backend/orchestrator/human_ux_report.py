"""Human + Think-Aloud UX Evaluation Report.

Turns a completed Human + Think-Aloud result (transcript + Phase 3
classifications) into a practical, evidence-grounded report: a deterministic
"Task UX Score", a signal summary, a chronological evidence timeline, a
handful of evidence-grounded findings, an expectation-vs-experience view,
and an optional LLM-written summary that is constrained to restate only the
supplied evidence.

HARD CONSTRAINTS (see the PHASE spec this implements):
- The Task UX Score is a heuristic for THIS session only -- never presented
  as a validated, universal measure of "website quality." See
  TASK_UX_SCORE_DISCLAIMER below; the frontend must display it verbatim.
- The score is 100% deterministic Python arithmetic. No LLM is involved in
  producing any number here -- the only LLM use in this module is
  generate_session_summary(), which writes prose ABOUT already-computed,
  already-structured evidence, and does not affect the score in any way.
- Only data that actually exists in a Human + Think-Aloud session is used:
  transcript segments, their Phase 3 classifications, timestamps, task
  outcome (researcher-selected). No Phase 1 autonomous-agent friction data
  (clicks, scrolls, navigation loops, ...) is used or referenced anywhere in
  this module -- that data does not exist for a human-driven session and is
  never fabricated here.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger("ux-human-report")

TASK_UX_SCORE_DISCLAIMER = (
    "Task UX Score reflects evidence observed in this session and should not "
    "be interpreted as a universal website quality score."
)
TASK_UX_SCORE_METHOD_NOTE = "Evidence-based heuristic score for this session."

# ---- score weighting (kept as explicit constants so they can be re-tuned
# later without touching the calculation logic) --------------------------
TASK_COMPLETION_MAX = 40
VERBAL_SIGNALS_MAX = 25
RECOVERY_MAX = 20
EVIDENCE_COMPLETENESS_MAX = 15
TASK_UX_SCORE_MAX = TASK_COMPLETION_MAX + VERBAL_SIGNALS_MAX + RECOVERY_MAX + EVIDENCE_COMPLETENESS_MAX  # 100

TASK_OUTCOMES = ("completed", "partially_completed", "not_completed", "not_assessed")
DEFAULT_TASK_OUTCOME = "not_assessed"

_COMPLETION_POINTS = {
    "completed": TASK_COMPLETION_MAX,        # 40
    "partially_completed": TASK_COMPLETION_MAX // 2,  # 20
    "not_completed": 0,
    "not_assessed": None,  # never silently scored -- see calculate_task_completion_score
}

# Verbal UX signal deductions (from a 25-point starting value). Only these
# three labels are treated as friction signals; EXPECTATION, NAVIGATION_INTENT
# and CONFIRMATION are contextual/neutral and never deduct.
_FRICTION_DEDUCTIONS = {
    "CONFUSION": 2,
    "UNCERTAINTY": 1,
    "COMPLAINT": 2,
}
_NEUTRAL_LABELS = ("EXPECTATION", "NAVIGATION_INTENT", "CONFIRMATION")
_DIFFICULTY_LABELS = frozenset(_FRICTION_DEDUCTIONS.keys())

_NOT_FULLY_ASSESSED = "Not fully assessed"


# ------------------------------------------------------------ helpers ----

def _segment_labels(segment: Dict[str, Any]) -> List[str]:
    classification = segment.get("classification")
    if not isinstance(classification, dict):
        return []
    labels = classification.get("labels")
    return labels if isinstance(labels, list) else []


def _segment_sort_key(segment: Dict[str, Any]):
    """Sort by start_time_ms when available; untimed segments keep their
    original relative order, placed after any timed ones."""
    start = segment.get("start_time_ms")
    return (start is None, start if start is not None else 0.0)


def _ordered_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(segments, key=_segment_sort_key)


def _segment_text(segment: Dict[str, Any]) -> str:
    # display_transcript is the evidence text shown to researchers; raw_transcript
    # is preserved elsewhere untouched and is never used as report evidence text.
    return segment.get("display_transcript") or segment.get("transcript") or ""


# ---------------------------------------------------- A. task completion --

def calculate_task_completion_score(task_outcome: str) -> Optional[int]:
    """Returns the 0-40 completion score, or None if the researcher has not
    assessed the task yet ("not_assessed") -- callers must never convert
    None into a guessed number."""
    return _COMPLETION_POINTS.get(task_outcome, _COMPLETION_POINTS["not_assessed"])


# ----------------------------------------------- B. verbal UX signals -----
#
# DELIBERATE DESIGN DECISION (documented per the refinement spec, not a
# silent change): this scoring function operates on whichever `segments`
# list it is given -- i.e. on EVIDENCE segments, not on Sarvam's original,
# possibly much coarser source chunks. When think_aloud.py splits one large
# Sarvam chunk (e.g. an entire 24s recording returned as a single "word") into
# several sentence-level evidence segments for finer classification
# granularity, the verbal-signals deduction is computed per EVIDENCE segment.
#
# This is intentional, not an accidental score change from finer
# segmentation: the "per segment" counting rule has always been meant to
# approximate "per distinct verbal instance". Two genuinely separate
# complaints transcribed as two sentences deserve two deductions; the old
# behavior of bundling them into one coarse chunk (and thus one deduction)
# was an artifact of poor segmentation, not a deliberate research choice to
# under-count repeated friction. See test_score_operates_on_evidence_segments
# in tests/test_human_ux_report.py, which pins this behavior explicitly so it
# cannot drift silently in the future.

def calculate_verbal_signals_score(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """25 points minus deterministic deductions for explicit friction labels
    (CONFUSION/UNCERTAINTY/COMPLAINT), each counted at most once per segment.
    Neutral labels (EXPECTATION/NAVIGATION_INTENT/CONFIRMATION) never deduct.
    """
    score = VERBAL_SIGNALS_MAX
    evidence: Dict[str, List[str]] = {label: [] for label in _FRICTION_DEDUCTIONS}
    for segment in segments:
        labels = set(_segment_labels(segment))  # dedup: count each label once per segment
        segment_id = segment.get("segment_id")
        for label, deduction in _FRICTION_DEDUCTIONS.items():
            if label in labels:
                score -= deduction
                evidence[label].append(segment_id)
    score = max(0, score)
    return {"score": score, "max": VERBAL_SIGNALS_MAX, "evidence_segment_ids": evidence}


# ------------------------------------------------------------ C. recovery -

def calculate_recovery_score(segments: List[Dict[str, Any]], task_outcome: str) -> Dict[str, Any]:
    """Recovery: does the session show movement from an explicitly stated
    difficulty (CONFUSION/UNCERTAINTY/COMPLAINT) toward a later CONFIRMATION,
    combined with the researcher-selected task outcome? Uses ONLY explicit
    Think-Aloud evidence and the task outcome -- never inferred from silence.

    Explicitly specified anchor points (from the Phase spec):
      - task_outcome == "not_assessed"            -> not fully assessed
      - no difficulty signal anywhere in session   -> 15/20 (no episode to judge)
      - difficulty -> later CONFIRMATION -> completed            -> 20/20
      - difficulty -> later CONFIRMATION -> partially_completed  -> 12/20
      - difficulty -> no later CONFIRMATION -> not_completed     -> 0/20

    The remaining combinations (difficulty + later confirmation + not_completed;
    difficulty + no later confirmation + completed/partially_completed) are not
    given explicit numbers in the spec. They are extrapolated here, monotonically
    consistent with the anchors above (never exceeding the "no difficulty" 15
    baseline when recovery wasn't actually evidenced, never exceeding the
    matching anchor when the outcome is worse) -- documented inline, not hidden.
    """
    if task_outcome == "not_assessed":
        return {"score": None, "max": RECOVERY_MAX, "status": _NOT_FULLY_ASSESSED,
                "rationale": "Task outcome has not been assessed yet."}

    ordered = _ordered_segments(segments)
    difficulty_index = None
    for i, segment in enumerate(ordered):
        if _DIFFICULTY_LABELS.intersection(_segment_labels(segment)):
            difficulty_index = i
            break

    if difficulty_index is None:
        return {"score": 15, "max": RECOVERY_MAX,
                "rationale": "No explicit difficulty signal (CONFUSION/UNCERTAINTY/COMPLAINT) "
                             "was observed, so there is no recovery episode to evaluate."}

    has_later_confirmation = any(
        "CONFIRMATION" in _segment_labels(seg) for seg in ordered[difficulty_index + 1:]
    )

    if has_later_confirmation:
        if task_outcome == "completed":
            return {"score": 20, "max": RECOVERY_MAX,
                    "rationale": "An explicit difficulty signal was followed by a later "
                                 "confirmation, and the task was marked completed."}
        if task_outcome == "partially_completed":
            return {"score": 12, "max": RECOVERY_MAX,
                    "rationale": "An explicit difficulty signal was followed by a later "
                                 "confirmation, and the task was marked partially completed."}
        # not_completed, despite a later confirmation -- extrapolated, not an explicit anchor.
        return {"score": 6, "max": RECOVERY_MAX,
                "rationale": "An explicit difficulty signal was followed by a later confirmation, "
                             "but the task was ultimately marked not completed (extrapolated rule)."}

    if task_outcome == "not_completed":
        return {"score": 0, "max": RECOVERY_MAX,
                "rationale": "An explicit difficulty signal was observed with no later "
                             "confirmation, and the task was marked not completed."}
    # completed/partially_completed despite no later confirmation -- extrapolated,
    # kept below the no-difficulty baseline (15) since recovery wasn't evidenced.
    return {"score": 8, "max": RECOVERY_MAX,
            "rationale": "An explicit difficulty signal was observed with no later confirmation, "
                         "even though the task outcome was not marked not-completed (extrapolated rule)."}


# ------------------------------------------------ D. evidence completeness

def calculate_evidence_completeness_score(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Data-quality dimension: is there enough structured Think-Aloud
    evidence for this report to be interpretable? NOT a judgement of the
    participant -- a quiet session with clean data still scores well here.
    """
    has_transcript = any(_segment_text(s).strip() for s in segments)
    if not has_transcript:
        return {"score": 0, "max": EVIDENCE_COMPLETENESS_MAX, "rationale": "No usable transcript."}

    has_classification = any(isinstance(s.get("classification"), dict) for s in segments)
    if not has_classification:
        return {"score": 5, "max": EVIDENCE_COMPLETENESS_MAX,
                "rationale": "Transcript exists but classification is unavailable."}

    all_timed = all(s.get("start_time_ms") is not None and s.get("end_time_ms") is not None
                     for s in segments)
    if all_timed:
        return {"score": 15, "max": EVIDENCE_COMPLETENESS_MAX,
                "rationale": "Transcript, timestamps, and classification are all available."}
    return {"score": 10, "max": EVIDENCE_COMPLETENESS_MAX,
            "rationale": "Transcript and classification are available, but timing is incomplete."}


# ------------------------------------------------------ overall UX score --

def calculate_task_ux_score(segments: List[Dict[str, Any]], task_outcome: str) -> Dict[str, Any]:
    """Combines all four dimensions. If task_outcome is "not_assessed", the
    overall total is never a misleading full number -- it's reported as
    "Not fully assessed" alongside whichever component scores ARE available.
    """
    completion = calculate_task_completion_score(task_outcome)
    verbal = calculate_verbal_signals_score(segments)
    recovery = calculate_recovery_score(segments, task_outcome)
    evidence = calculate_evidence_completeness_score(segments)

    fully_assessed = completion is not None and recovery.get("score") is not None
    total = (completion + verbal["score"] + recovery["score"] + evidence["score"]) if fully_assessed else None

    return {
        "total": total,
        "max": TASK_UX_SCORE_MAX,
        "fully_assessed": fully_assessed,
        "status": None if fully_assessed else _NOT_FULLY_ASSESSED,
        "disclaimer": TASK_UX_SCORE_DISCLAIMER,
        "method_note": TASK_UX_SCORE_METHOD_NOTE,
        "dimensions": {
            "task_completion": {
                "score": completion, "max": TASK_COMPLETION_MAX,
                "status": _NOT_FULLY_ASSESSED if completion is None else None,
            },
            "verbal_ux_signals": verbal,
            "recovery": recovery,
            "evidence_completeness": evidence,
        },
    }


# ------------------------------------------------------- signal summary --

def build_signal_summary(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic counts of each of the six labels across all segments,
    with the supporting segment_ids so the frontend can make each count
    expandable to its evidence without a second computation."""
    all_labels = list(_NEUTRAL_LABELS) + list(_FRICTION_DEDUCTIONS.keys())
    counts = {label: {"count": 0, "segment_ids": []} for label in all_labels}
    for segment in segments:
        segment_id = segment.get("segment_id")
        for label in set(_segment_labels(segment)):
            if label in counts:
                counts[label]["count"] += 1
                counts[label]["segment_ids"].append(segment_id)
    return counts


# ------------------------------------------------------------ timeline ---

def build_evidence_timeline(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chronological "Think-Aloud Evidence Timeline" -- explicitly NOT a
    "Common UX Timeline" (that belongs to a later phase that fuses browser
    events with this transcript; no browser events exist for a Human
    session's Think-Aloud data, so none are referenced here)."""
    timeline = []
    for segment in _ordered_segments(segments):
        timeline.append({
            "segment_id": segment.get("segment_id"),
            "start_time_ms": segment.get("start_time_ms"),
            "end_time_ms": segment.get("end_time_ms"),
            # "source" = Sarvam's own timestamp for exactly this text;
            # "derived" = this text is one sentence split out of a larger
            # Sarvam chunk and start/end are that PARENT chunk's shared
            # range, not this sentence's own exact timing; "untimed" = no
            # timestamp at all. The frontend uses this to honestly label
            # shared-range entries rather than implying per-sentence timing
            # Sarvam never provided.
            "timing_source": segment.get("timing_source"),
            "labels": _segment_labels(segment),
            "text": _segment_text(segment),
            "language": segment.get("language"),
        })
    return timeline


# ------------------------------------------------------------- findings --

_FINDING_TITLES = {
    frozenset(["EXPECTATION"]): "Stated expectation",
    frozenset(["CONFUSION"]): "Reported confusion",
    frozenset(["NAVIGATION_INTENT"]): "Navigation decision",
    frozenset(["UNCERTAINTY"]): "Expressed uncertainty",
    frozenset(["CONFIRMATION"]): "Confirmed finding",
    frozenset(["COMPLAINT"]): "Usability complaint",
}


def _finding_title(labels: List[str]) -> str:
    key = frozenset(labels)
    if key in _FINDING_TITLES:
        return _FINDING_TITLES[key]
    # Multi-label combination not in the single-label map above -- build a
    # deterministic compound title rather than inventing subject-matter
    # (e.g. "Confusion + Usability complaint" for CONFUSION+COMPLAINT).
    singles = [_FINDING_TITLES.get(frozenset([l]), l.replace("_", " ").title()) for l in labels]
    return " + ".join(singles)


def build_findings(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One finding per segment that has at least one classified label --
    never fabricated: a segment with zero labels produces zero findings.
    Titles are generated deterministically from the label combination
    (never invented subject-matter/topic text), while evidence_text is
    always the segment's own verbatim display_transcript."""
    findings = []
    for segment in _ordered_segments(segments):
        labels = _segment_labels(segment)
        if not labels:
            continue
        findings.append({
            "title": _finding_title(labels),
            "evidence_text": _segment_text(segment),
            "labels": labels,
            "segment_id": segment.get("segment_id"),
            "start_time_ms": segment.get("start_time_ms"),
            "end_time_ms": segment.get("end_time_ms"),
        })
    return findings


# ------------------------------------------------- expectation vs experience

_EXPECTATION_LABELS = frozenset(["EXPECTATION"])
_EXPERIENCE_LABELS = frozenset(["COMPLAINT", "CONFUSION"])
_NEXT_LABELS = frozenset(["NAVIGATION_INTENT", "UNCERTAINTY"])
_CONFIRMATION_LABELS = frozenset(["CONFIRMATION"])


def _segment_view(segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segment_id": segment.get("segment_id"),
        "text": _segment_text(segment),
        "labels": _segment_labels(segment),
        "start_time_ms": segment.get("start_time_ms"),
        "end_time_ms": segment.get("end_time_ms"),
    }


def build_expectation_vs_experience(segments: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Four evidence-linked slots (expectation / observed_experience /
    what_happened_next / confirmation). Each is either a real segment's
    evidence or None -- never a generic/invented diagnosis. Slots are
    searched in chronological order and, past the first, only after the
    previous slot's position, so the narrative reads forward in time."""
    ordered = _ordered_segments(segments)

    def first_after(label_set: frozenset, after_index: int):
        for i in range(after_index + 1, len(ordered)):
            if label_set.intersection(_segment_labels(ordered[i])):
                return i, ordered[i]
        return None, None

    exp_i, exp_seg = first_after(_EXPECTATION_LABELS, -1)
    obs_i, obs_seg = first_after(_EXPERIENCE_LABELS, exp_i if exp_i is not None else -1)
    next_i, next_seg = first_after(_NEXT_LABELS, obs_i if obs_i is not None else (exp_i if exp_i is not None else -1))
    conf_i, conf_seg = first_after(_CONFIRMATION_LABELS,
                                    next_i if next_i is not None else
                                    (obs_i if obs_i is not None else (exp_i if exp_i is not None else -1)))

    return {
        "expectation": _segment_view(exp_seg) if exp_seg else None,
        "observed_experience": _segment_view(obs_seg) if obs_seg else None,
        "what_happened_next": _segment_view(next_seg) if next_seg else None,
        "confirmation": _segment_view(conf_seg) if conf_seg else None,
    }


# -------------------------------------------------------- session summary

_SUMMARY_PROMPT = """\
You are writing a short, factual summary of a UX research Think-Aloud session, \
for a researcher's report. You are given the task, persona, the researcher's \
task outcome assessment, and the EXACT structured evidence collected (transcript \
segments with their explicit classification labels).

Summarize ONLY what is supported by the supplied evidence. Do not infer emotions \
or psychological states (e.g. never say "frustrated", "confused" as a feeling, \
"stressed"). Do not introduce facts not present in the evidence. Do not create \
new UX problems or observations beyond what the evidence already states. Write \
2-4 plain sentences, no headings, no JSON, no markdown.

=== TASK ===
{task_description}

=== PERSONA ===
{persona_summary}

=== TASK OUTCOME (researcher-assessed) ===
{task_outcome}

=== STRUCTURED EVIDENCE (chronological) ===
{evidence_lines}
"""


def _persona_summary_line(persona: Optional[Dict[str, Any]]) -> str:
    if not persona:
        return "Not specified."
    parts = [f"{k}: {v}" for k, v in persona.items() if v]
    return "; ".join(parts) if parts else "Not specified."


def generate_session_summary(
    llm, task_description: str, persona: Optional[Dict[str, Any]],
    segments: List[Dict[str, Any]], task_outcome: str,
) -> str:
    """Optional LLM-written prose summary, constrained to restate only the
    supplied structured evidence -- it does not add new observations and it
    never affects the score. Falls back to a deterministic, template-built
    summary if no LLM is configured or the call fails, so the report always
    has a summary rather than a blank/crashed section."""
    ordered = _ordered_segments(segments)
    evidence_lines = "\n".join(
        f"- [{', '.join(_segment_labels(s)) or 'no label'}] \"{_segment_text(s)}\""
        for s in ordered if _segment_text(s).strip()
    ) or "(no transcript evidence)"

    if llm is not None:
        try:
            prompt = _SUMMARY_PROMPT.format(
                task_description=task_description or "Not specified.",
                persona_summary=_persona_summary_line(persona),
                task_outcome=task_outcome,
                evidence_lines=evidence_lines,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.error(f"Session summary LLM call failed, using deterministic fallback: {e}")

    # Deterministic fallback: a plain, evidence-grounded template.
    labeled = [s for s in ordered if _segment_labels(s)]
    if not labeled:
        return "No explicit UX signals were classified in this session's transcript."
    fragments = [f'"{_segment_text(s)}" ({", ".join(_segment_labels(s))})' for s in labeled[:4]]
    return "Observed evidence in this session included: " + "; ".join(fragments) + "."


# --------------------------------------------------------- full report ---

def build_human_ux_report(
    target_url: str,
    task_description: str,
    persona: Optional[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    audio_duration_ms: Optional[float],
    language: Optional[str],
    task_outcome: str = DEFAULT_TASK_OUTCOME,
    llm=None,
) -> Dict[str, Any]:
    """Assembles the full HumanUXReport. Pure function of its inputs (plus
    one optional LLM call for prose summary only) -- safe to call repeatedly
    with the same session data and always get the same score/breakdown."""
    if task_outcome not in TASK_OUTCOMES:
        task_outcome = DEFAULT_TASK_OUTCOME

    segments = segments or []
    score = calculate_task_ux_score(segments, task_outcome)
    signal_summary = build_signal_summary(segments)
    timeline = build_evidence_timeline(segments)
    findings = build_findings(segments)
    expectation_experience = build_expectation_vs_experience(segments)
    summary = generate_session_summary(llm, task_description, persona, segments, task_outcome)

    return {
        "session": {
            "target_url": target_url,
            "task_description": task_description,
            "persona": persona,
            "audio_duration_ms": audio_duration_ms,
            "language": language,
            "segment_count": len(segments),
        },
        "task_outcome": task_outcome,
        "score": score,
        "signal_summary": signal_summary,
        "timeline": timeline,
        "findings": findings,
        "expectation_experience": expectation_experience,
        "summary": summary,
    }
