"""Phase 3: Think-Aloud UX Classification.

Classifies the EXPLICIT verbal UX content of each ThinkAloudSegment (Phase 2)
into zero or more of exactly six labels. This is a baseline LLM classifier,
deliberately kept behind a small abstraction (ThinkAloudClassifier) so a
future supervised/ML classifier can replace it without touching any caller.

Hard scope boundaries (see PHASE 3 spec):
- Classifies only what the participant explicitly SAID. Never infers
  emotion, confusion-from-pauses, frustration-from-clicks, stress, or any
  other psychological/behavioural state -- those are separate signal
  channels (Phase 1 friction, and later multimodal fusion), not this
  classifier's job.
- Reasons only over `display_transcript` (the romanized text). Never sees
  audio, screenshots, or browser events.
- Never rewrites transcript text; evidence_text must be grounded in (copied
  from) the transcript, never paraphrased or invented.
- Never modifies any existing ThinkAloudSegment field (raw_transcript,
  display_transcript, transcript, timestamps, language, ...). Classification
  is attached as separate metadata alongside the segment, never merged into
  or replacing transcript fields.

This module is independent of FastAPI routes, Playwright, browser events,
friction analysis, and Sarvam STT -- it only knows about segment dicts in,
classification dicts out.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

try:
    from ..agents.json_utils import extract_json_object
except ImportError:
    from agents.json_utils import extract_json_object

logger = logging.getLogger("ux-think-aloud-classifier")

ALLOWED_LABELS = (
    "EXPECTATION", "CONFUSION", "NAVIGATION_INTENT",
    "UNCERTAINTY", "CONFIRMATION", "COMPLAINT",
)
_ALLOWED_LABELS_SET = frozenset(ALLOWED_LABELS)
ALLOWED_CONFIDENCE = ("high", "medium", "low")
_ALLOWED_CONFIDENCE_SET = frozenset(ALLOWED_CONFIDENCE)

_LABEL_DEFINITIONS = """\
- EXPECTATION: the participant explicitly states what they expected to find, see, or happen.
- CONFUSION: the participant explicitly states difficulty understanding where to go, what
  something means, or how something works. Only assign this from what they SAY -- you are
  not given pauses, click counts, or any other behavioural signal, so never infer this from
  hesitation or timing; there is none in your input.
- NAVIGATION_INTENT: the participant explicitly states an intended next navigation or interaction
  (e.g. deciding to check a different link/section/page next). This is a forward-looking plan,
  not itself a problem report -- assign NAVIGATION_INTENT alone unless the same statement
  ALSO separately names something wrong with the interface.
- UNCERTAINTY: the participant explicitly expresses uncertainty or lack of confidence
  (e.g. "maybe", "shayad", "I think", "perhaps").
- CONFIRMATION: the participant explicitly confirms that something is correct, relevant,
  or what they were looking for.
- COMPLAINT: the participant explicitly identifies a usability problem, obstacle, missing
  information, poor visibility, or an unclear/hard-to-find interface element -- i.e. any
  explicit negative observation about the interface itself, stated as a fact about the
  interface (not merely "I expected X elsewhere" -- that alone is EXPECTATION). This
  includes statements like "X is not visible", "X nahi dikh raha/rahi", "X samajh nahi aa
  raha" (about an interface element), "X easily visible nahi hai", "X available nahi hai",
  or "X missing/broken/hidden/hard to find". A statement can be BOTH COMPLAINT and another
  label at once -- e.g. "button samajh nahi aa raha" is both CONFUSION (difficulty
  understanding) and COMPLAINT (identifies the button as the problem).

IMPORTANT: a single transcript segment often contains several sentences/clauses (e.g. one
sentence stating an expectation, another separately stating something was not visible,
another stating a decision to check elsewhere). Evaluate EACH clause independently against
EVERY label above -- do not let one clause's label (e.g. NAVIGATION_INTENT for "I'll check
other links") stop you from also assigning a different label (e.g. COMPLAINT) to a SEPARATE
clause in the same segment that explicitly names an interface problem.
"""

_PROMPT_TEMPLATE = """\
You are classifying ONE spoken think-aloud statement from a UX research \
participant who was manually using a website. You are given ONLY the \
transcript text below -- no audio, no screenshots, no click/scroll data, no \
timing information. Classify strictly what the participant explicitly SAID.

Do NOT infer emotion, confusion, frustration, stress, or cognitive load. Do \
NOT assume a psychological state. Do NOT translate or rewrite the statement \
-- it may be in English, Hindi, or Hinglish/code-mixed Roman script; treat \
it exactly as written.

=== TRANSCRIPT SEGMENT ===
{display_transcript}

=== ALLOWED LABELS (assign zero, one, or several; use ONLY these six) ===
{label_definitions}

RULES:
- A segment may have zero labels, one label, or multiple labels. Do not force exactly one.
- Only assign a label when the statement EXPLICITLY supports it under its definition above.
  When in doubt, leave it out.
- evidence_text MUST be copied verbatim (an exact substring) from the transcript above --
  never paraphrase, translate, summarize, or invent it.
- rationale must be one short, concise sentence, grounded only in the words said -- never a
  claim about the participant's emotional or mental state.
- Respond with ONLY valid JSON, no prose, no markdown fences, matching exactly this shape:
{{"labels": ["<LABEL>", "..."], "evidence_text": "<verbatim quote from the transcript>", \
"confidence": "high|medium|low", "rationale": "<one concise, evidence-grounded sentence>"}}
"""


def _safe_fallback(segment: Dict[str, Any], rationale: str = "Classification unavailable.") -> Dict[str, Any]:
    """Never invents a label. Used whenever the LLM is unavailable, the
    response can't be parsed, or nothing about it can be trusted."""
    return {
        "segment_id": segment.get("segment_id"),
        "labels": [],
        "evidence_text": segment.get("display_transcript") or "",
        "confidence": "low",
        "rationale": rationale,
    }


def _validate_classification(result: Any, segment: Dict[str, Any]) -> Dict[str, Any]:
    """LLM output is never trusted blindly. Invalid labels are dropped (not
    fatal on their own); an invalid/missing confidence is coerced to "low";
    a missing/empty evidence_text falls back to the segment's own transcript
    rather than being invented."""
    if not isinstance(result, dict):
        return _safe_fallback(segment)

    raw_labels = result.get("labels")
    labels: List[str] = []
    if isinstance(raw_labels, list):
        for label in raw_labels:
            if isinstance(label, str):
                candidate = label.strip().upper()
                if candidate in _ALLOWED_LABELS_SET and candidate not in labels:
                    labels.append(candidate)
                elif candidate and candidate not in _ALLOWED_LABELS_SET:
                    logger.warning(f"Dropping invalid think-aloud label from LLM output: {label!r}")

    confidence = result.get("confidence")
    if isinstance(confidence, str) and confidence.strip().lower() in _ALLOWED_CONFIDENCE_SET:
        confidence = confidence.strip().lower()
    else:
        confidence = "low"

    evidence_text = result.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        evidence_text = segment.get("display_transcript") or ""

    rationale = result.get("rationale")
    rationale = rationale.strip() if isinstance(rationale, str) else ""

    return {
        "segment_id": segment.get("segment_id"),
        "labels": labels,
        "evidence_text": evidence_text,
        "confidence": confidence,
        "rationale": rationale[:400],
    }


class ThinkAloudClassifier:
    """Abstraction so a future supervised/ML classifier can replace the LLM
    baseline without any caller change. Only LLMThinkAloudClassifier is
    implemented in Phase 3."""

    def classify_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def classify_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.classify_segment(s) for s in segments]


class LLMThinkAloudClassifier(ThinkAloudClassifier):
    """Phase 3 baseline: uses the project's existing configured LLM (same
    Groq/OpenAI client already built for evaluator/reporter analysis --
    no new provider, no new API key, no second LLM client)."""

    def __init__(self, llm):
        self.llm = llm

    def classify_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        text = (segment.get("display_transcript") or "").strip()

        if not text:
            return _safe_fallback(segment, rationale="Empty transcript segment.")
        if self.llm is None:
            return _safe_fallback(segment, rationale="No LLM configured.")

        prompt = _PROMPT_TEMPLATE.format(display_transcript=text, label_definitions=_LABEL_DEFINITIONS)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_content = response.content
            finish_reason = None
            meta = getattr(response, "response_metadata", None)
            if isinstance(meta, dict):
                finish_reason = meta.get("finish_reason")

            # Debug logging around the real LLM response, per Phase 3 debug
            # requirements: segment id, response type, content length, and
            # the full untruncated content -- never the API key or any
            # request header/secret. Left at DEBUG level (already visible
            # since this project runs logging at INFO by default in main.py,
            # so bump to INFO deliberately while this is actively debugged).
            logger.info(f"[ThinkAloudClassifier] segment={segment.get('segment_id')} "
                        f"response_type={type(response).__name__} "
                        f"content_type={type(raw_content).__name__} "
                        f"content_length={len(raw_content) if raw_content else 0} "
                        f"finish_reason={finish_reason}")
            # log_raw_response() logs at DEBUG, which main.py's default INFO
            # level would silently suppress -- log the full, untruncated raw
            # content at INFO explicitly so it's actually visible while this
            # is being debugged. Never logs the API key or any request header.
            logger.info(f"[ThinkAloudClassifier] segment={segment.get('segment_id')} "
                        f"raw LLM content:\n{raw_content!r}")

            if finish_reason == "length":
                # The model's own token budget cut the response off mid-JSON --
                # this is NOT a generic parse failure, it's a truncation, and
                # is worth distinguishing so it's never mistaken for malformed
                # model output in future debugging.
                logger.error(f"[ThinkAloudClassifier] segment={segment.get('segment_id')} "
                             f"LLM response was TRUNCATED by max_tokens (finish_reason=length); "
                             f"content_length={len(raw_content) if raw_content else 0}")
                return _safe_fallback(segment, rationale="Classification unavailable: LLM response was truncated.")

            result = extract_json_object(raw_content)
            return _validate_classification(result, segment)
        except Exception as e:
            logger.error(f"[ThinkAloudClassifier] classification failed for segment "
                         f"{segment.get('segment_id')}: {type(e).__name__}: {e}")
            return _safe_fallback(segment)


def classify_result(result: Dict[str, Any], classifier: Optional[ThinkAloudClassifier]) -> Dict[str, Any]:
    """Attach a `classification` dict to each segment in a ThinkAloudResult,
    in place, without touching any existing segment field. No-op (segments
    left exactly as they are) if the result isn't available (transcription
    failed) or no classifier was supplied.

    Classification failures never crash the caller -- classify_segment
    already falls back safely per segment, and this function additionally
    guards the whole batch defensively.
    """
    if not result or not result.get("available") or classifier is None:
        return result
    try:
        for segment in result.get("segments", []):
            segment["classification"] = classifier.classify_segment(segment)
    except Exception:
        logger.exception("Think-aloud classification batch failed; segments left unclassified")
    return result
