import { useState, useEffect, useRef } from "react";

/* ── helpers ──────────────────────────────────────────────────────────────── */
const clamp = (v, min = 0, max = 1) => Math.min(max, Math.max(min, v ?? 0));

/* ── Logo SVG ─────────────────────────────────────────────────────────────── */
function LogoIcon() {
  return (
    <svg viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg" width="34" height="34">
      <circle cx="17" cy="17" r="15" className="logo-icon-ring" />
      <path d="M17 4 A13 13 0 0 1 30 17" className="logo-icon-arc" />
      <circle cx="17" cy="17" r="2.5" className="logo-icon-dot" />
    </svg>
  );
}

/* ── Score ring (artsy) ───────────────────────────────────────────────────── */
function ScoreRing({ value, max = 10, label, color = "#4B5694", rationale }) {
  const sz = 90;
  const r1 = 36; // outer track
  const r2 = 28; // inner dash ring
  const c = sz / 2;
  const circ = 2 * Math.PI * r1;
  const fraction = clamp((value ?? 0) / max, 0, 1);
  const offset = circ * (1 - fraction);
  return (
    <div className="score-ring-wrap">
      <svg width={sz} height={sz} className="score-ring-svg">
        {/* Outer track */}
        <circle cx={c} cy={c} r={r1} className="ring-bg" />
        {/* Dashed inner ring */}
        <circle cx={c} cy={c} r={r2} className="ring-dash" />
        {/* Filled arc */}
        <circle
          cx={c} cy={c} r={r1}
          className="ring-arc"
          style={{
            stroke: color,
            strokeDasharray: circ,
            strokeDashoffset: offset,
          }}
        />
        {/* Number */}
        <text x={c} y={c + 1} className="ring-num">{value ?? "—"}</text>
      </svg>
      <div className="ring-label">{label}</div>
      {rationale && <div className="ring-rationale">{rationale}</div>}
    </div>
  );
}

/* ── Phase helper ─────────────────────────────────────────────────────────── */
const PHASE_ORDER = ["running", "evaluating", "reporting", "done"];
function phaseDone(current, target) {
  return PHASE_ORDER.indexOf(current) > PHASE_ORDER.indexOf(target);
}

const PHASE_LABELS = {
  idle: "Idle",
  running: "Navigating",
  evaluating: "Evaluating",
  reporting: "Writing report",
  done: "Complete",
  error: "Error",
};

const LOG_TYPE = {
  navigate: "navigate",
  click: "click",
  scroll: "scroll",
  complete: "complete",
  error: "error",
  evaluating: "eval",
  reporting: "eval",
};

/* ── Human + Think-Aloud panel (Phase 2) ─────────────────────────────────────
   Fully separate from the Autonomous Agent flow: no run_evaluation_stream,
   no navigator, no SSE -- just session lifecycle calls + a real microphone
   recording sent to Sarvam STT for transcription. */
function fmtMMSS(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function ThinkAloudPanel({ url, setUrl, task, setTask, persona, setPersona }) {
  const [stage, setStage] = useState("setup");       // setup | recording | stopped | processing | completed | error
  const [micError, setMicError] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioDurationMs, setAudioDurationMs] = useState(null);
  const [segments, setSegments] = useState([]);
  const [timingStatus, setTimingStatus] = useState(null);
  const [language, setLanguage] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const recordingStartedAtRef = useRef(null);
  // The real target website, opened as its own genuine browser window/tab --
  // NOT an iframe. mygov.in (the site this whole project is tested against)
  // sends X-Frame-Options: SAMEORIGIN, so browsers refuse to render it in an
  // iframe on our origin; window.open() is a real top-level navigation, so
  // it is never subject to framing restrictions.
  const websiteWindowRef = useRef(null);
  const [websiteWindowClosed, setWebsiteWindowClosed] = useState(false);

  const openWebsiteWindow = () => {
    // NOTE: intentionally NOT using "noopener" -- per spec, noopener makes
    // window.open() always return null (it deliberately severs the handle),
    // which would make it impossible to detect whether the user later closed
    // the tab. "noreferrer" alone is kept for privacy and still returns a
    // usable window reference.
    const w = window.open(url, "zynx-think-aloud-target", "noreferrer");
    if (!w) {
      setWebsiteWindowClosed(true);
      return null;
    }
    websiteWindowRef.current = w;
    setWebsiteWindowClosed(false);
    return w;
  };

  const resetForRetry = () => {
    setStage("setup");
    setMicError("");
    setSessionId(null);
    setElapsedSec(0);
    setAudioBlob(null);
    setAudioDurationMs(null);
    setSegments([]);
    setTimingStatus(null);
    setLanguage(null);
    setWebsiteWindowClosed(false);
  };

  const startSession = async () => {
    setMicError("");

    if (!navigator.mediaDevices?.getUserMedia || typeof window.MediaRecorder === "undefined") {
      setStage("error");
      setMicError("This browser does not support audio recording (MediaRecorder API unavailable). Try a recent Chrome, Edge, or Firefox.");
      return;
    }

    // Open the REAL target website as its own genuine browser window/tab --
    // done first, synchronously within this click handler, so browsers don't
    // treat it as an unrequested popup (most popup blockers only allow
    // window.open when it happens directly inside a user gesture, before any
    // `await`). This is a real top-level navigation to the actual site, not
    // an iframe -- it is never subject to X-Frame-Options/CSP framing rules.
    openWebsiteWindow();

    let newSessionId;
    try {
      const res = await fetch("/api/think-aloud/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_url: url, task_description: task, persona }),
      });
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || "Failed to start session");
      newSessionId = d.session_id;
      setSessionId(newSessionId);
    } catch (e) {
      setStage("error");
      setMicError(`Could not start session: ${e.message}`);
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setStage("error");
      if (e.name === "NotAllowedError" || e.name === "PermissionDeniedError") {
        setMicError("Microphone permission was denied. Please allow microphone access and try again.");
      } else if (e.name === "NotFoundError" || e.name === "DevicesNotFoundError") {
        setMicError("No microphone was found on this device.");
      } else {
        setMicError(`Could not access the microphone: ${e.message}`);
      }
      return;
    }

    streamRef.current = stream;
    chunksRef.current = [];
    let recorder;
    try {
      recorder = new MediaRecorder(stream);
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      setStage("error");
      setMicError(`Failed to initialize the recorder: ${e.message}`);
      return;
    }
    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data); };
    recorder.onerror = (e) => {
      setStage("error");
      setMicError(`Recording failed: ${e.error?.message || "unknown recorder error"}`);
    };
    recorder.start();
    recordingStartedAtRef.current = performance.now();

    try {
      await fetch("/api/think-aloud/recording/start", {
        method: "POST",
        body: new URLSearchParams({ session_id: newSessionId }),
      });
    } catch (_) { /* non-fatal for the local recording itself */ }

    setStage("recording");
    setElapsedSec(0);
    timerRef.current = setInterval(() => {
      setElapsedSec((s) => s + 1);
      // Recording itself is completely independent of this check -- it only
      // updates the "Reopen Website" affordance if the user closed that tab.
      // Wrapped defensively: a site with a strict Cross-Origin-Opener-Policy
      // could in principle make `.closed` throw rather than just read false.
      try {
        if (websiteWindowRef.current?.closed) setWebsiteWindowClosed(true);
      } catch (_) { /* cross-origin isolation quirk -- ignore, non-fatal */ }
    }, 1000);
  };

  const stopSession = async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;

    const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
    recorder.stop();
    await stopped;

    clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());

    const durationMs = recordingStartedAtRef.current != null
      ? performance.now() - recordingStartedAtRef.current : null;
    setAudioDurationMs(durationMs);

    try {
      await fetch("/api/think-aloud/recording/stop", {
        method: "POST",
        body: new URLSearchParams({ session_id: sessionId }),
      });
    } catch (_) { /* non-fatal */ }

    const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
    if (!blob || blob.size === 0) {
      setStage("error");
      setMicError("The recording was empty -- no audio was captured. Please try again.");
      return;
    }
    setAudioBlob(blob);
    setStage("stopped");
  };

  const processRecording = async () => {
    if (!audioBlob || !sessionId) return;
    setStage("processing");
    setMicError("");
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      if (audioDurationMs != null) form.append("audio_duration_ms", String(audioDurationMs));
      form.append("file", audioBlob, "recording.webm");
      const res = await fetch("/api/think-aloud/transcribe", { method: "POST", body: form });
      const result = await res.json();
      if (!result.available) {
        setStage("error");
        setMicError(result.error || "Transcription failed for an unknown reason.");
        return;
      }
      setSegments(result.segments || []);
      setTimingStatus(result.timing_status);
      setLanguage(result.language);
      setStage("completed");
    } catch (e) {
      setStage("error");
      setMicError(`Could not reach the transcription service: ${e.message}`);
    }
  };

  const editSegmentText = (segmentId, newText) => {
    // Only the displayed (Romanized) text is editable. raw_transcript --
    // Sarvam's original, untouched output -- is never modified by this.
    setSegments((prev) => prev.map((s) =>
      s.segment_id === segmentId ? { ...s, display_transcript: newText, transcript: newText } : s
    ));
  };
  const [showRawTranscript, setShowRawTranscript] = useState(false);

  useEffect(() => () => { clearInterval(timerRef.current); streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

  const STAGE_LABEL = {
    setup: "Ready", recording: "Recording", stopped: "Recording complete",
    processing: "Processing", completed: "Completed", error: "Error",
  };
  const dotClass = stage === "error" ? "error" : stage === "recording" || stage === "processing" ? "running" : stage === "setup" ? "idle" : "";
  const canEditSetup = stage === "setup";

  return (
    <>
      {/* ══════════════════════════════════════════════════════
          LEFT — SESSION SETUP
      ══════════════════════════════════════════════════════ */}
      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="section-num">B</span>
            <span className="section-title">Session Setup</span>
          </div>
          <div className="form-group">
            <label className="form-label">Website URL</label>
            <input className="input" value={url} onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com" disabled={!canEditSetup} />
          </div>
          <div className="form-group">
            <label className="form-label">Task Description</label>
            <textarea className="textarea" value={task} onChange={(e) => setTask(e.target.value)} rows={3}
              placeholder="What should you try to accomplish?" disabled={!canEditSetup} />
          </div>
          <div className="form-group">
            <label className="form-label">Persona Type</label>
            <div className="persona-grid">
              {["novice", "intermediate", "expert"].map((t) => (
                <button key={t} type="button" disabled={!canEditSetup}
                  className={`persona-pill ${persona.persona_type === t ? "active" : ""}`}
                  onClick={() => setPersona({ ...persona, persona_type: t })}>{t}</button>
              ))}
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="form-label" style={{marginBottom: 10}}>Session Status</div>
          <div className="header-status" style={{fontSize: 12}}>
            <span className={`status-dot ${dotClass}`} />
            {STAGE_LABEL[stage]}
          </div>
        </div>

        <div className="sidebar-footer">
          {stage === "setup" ? (
            <button id="ta-start-btn" className="btn-run" onClick={startSession}>Start Think-Aloud Session →</button>
          ) : (
            <p className="form-hint" style={{textAlign: "center"}}>
              Session actions are in the workspace on the right →
            </p>
          )}
        </div>
      </aside>

      {/* ══════════════════════════════════════════════════════
          RIGHT — THINK-ALOUD WORKSPACE
      ══════════════════════════════════════════════════════ */}
      <section className="panel">
        <div className="live-content">

          {stage === "setup" && (
            <div className="report-content">
              <div className="report-block">
                <div className="report-block-title">
                  <em>Think-Aloud Session</em>
                </div>
                <p className="report-para">
                  Navigate the website naturally and speak your thoughts aloud.
                </p>
              </div>
              <div className="report-block">
                <div className="rec-list">
                  {[
                    "Start the session -- the website opens in a new browser tab/window.",
                    "Allow microphone access when prompted.",
                    "Arrange the website tab and this panel side by side.",
                    "Perform the task yourself in the website tab, saying what you expect, notice, or are trying to find.",
                    "Come back to this panel and stop when the task is complete.",
                  ].map((step, i) => (
                    <div key={i} className="rec-item">
                      <span className="rec-num">{i + 1}</span>
                      <div className="rec-body"><div className="rec-detail">{step}</div></div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="report-block">
                <div className="stat-wall" style={{gridTemplateColumns: "1fr 1fr", border: "1px solid var(--cream-ghost)", borderRadius: "var(--radius-sm)"}}>
                  <div className="stat-item">
                    <span className="stat-label">Microphone</span>
                    <span className="rec-detail" style={{marginTop: 4}}>Ready to request access</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Website</span>
                    <span className="rec-detail" style={{marginTop: 4}}>Opens in a new tab on start</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {stage === "recording" && (
            <div className="empty-state">
              <div className="header-status" style={{fontSize: 11}}>
                <span className="status-dot running" />
                Recording
              </div>
              <h2 className="empty-title" style={{fontSize: 56}}>{fmtMMSS(elapsedSec)}</h2>
              <p className="empty-sub">Microphone: Active</p>
              <p className="form-hint" style={{textAlign: "center", maxWidth: 300}}>
                The website is open in a separate tab. Switch to it, use the site
                normally, and speak aloud -- recording continues regardless.
              </p>
              {websiteWindowClosed && (
                <button className="tab" onClick={openWebsiteWindow}>Reopen Website Tab</button>
              )}
              <div className="report-block" style={{textAlign: "left", maxWidth: 360}}>
                <p className="report-para" style={{marginBottom: 12}}>Speak naturally while completing the task. You can describe:</p>
                <div className="sum-col">
                  {["what you expect to find", "what you are looking for", "what seems unclear", "why you are choosing an option"].map((t, i) => (
                    <div key={i} className="sum-item"><span className="sum-bullet">•</span>{t}</div>
                  ))}
                </div>
              </div>
              <button className="btn-run" style={{maxWidth: 260}} onClick={stopSession}>Stop Session</button>
            </div>
          )}

          {stage === "stopped" && (
            <div className="empty-state">
              <div className="empty-glyph">✓</div>
              <h2 className="empty-title">Recording Complete</h2>
              <div className="stat-wall" style={{gridTemplateColumns: "1fr 1fr", border: "1px solid var(--cream-ghost)", borderRadius: "var(--radius-sm)", maxWidth: 320}}>
                <div className="stat-item">
                  <span className="stat-label">Duration</span>
                  <span className="stat-num" style={{fontSize: 22}}>{fmtMMSS((audioDurationMs || 0) / 1000)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Status</span>
                  <span className="rec-detail" style={{marginTop: 4}}>Ready to process</span>
                </div>
              </div>
              <div style={{display: "flex", flexDirection: "column", gap: 10, width: "100%", maxWidth: 260}}>
                <button className="btn-run" onClick={processRecording}>Process Recording</button>
                <button className="tab" style={{width: "100%", justifyContent: "center"}} onClick={resetForRetry}>Record Again</button>
              </div>
            </div>
          )}

          {stage === "processing" && (
            <div className="empty-state">
              <div className="header-status" style={{fontSize: 11}}>
                <span className="status-dot running" />
                Processing
              </div>
              <div className="spinner-lg" />
              <h2 className="empty-title">Transcribing your session…</h2>
              <p className="empty-sub">Sending audio to Sarvam for multilingual speech-to-text.</p>
            </div>
          )}

          {stage === "error" && (
            <div className="empty-state">
              <div className="empty-glyph" style={{color: "#c85c6e"}}>!</div>
              <h2 className="empty-title">Something went wrong</h2>
              <p className="empty-sub">{micError}</p>
              <button className="btn-run" style={{maxWidth: 200}} onClick={resetForRetry}>Try Again</button>
            </div>
          )}

          {stage === "completed" && (
            <div className="report-content">
              <div className="report-block">
                <div className="report-block-title">
                  <em>Think-Aloud Transcript</em>
                  <span className="report-block-title-en">
                    {language || "unknown"} · {segments.length} segment{segments.length === 1 ? "" : "s"} · {timingStatus}
                  </span>
                </div>
                <label className="form-hint" style={{display: "flex", alignItems: "center", gap: 6, cursor: "pointer", marginBottom: 14}}>
                  <input type="checkbox" checked={showRawTranscript} onChange={(e) => setShowRawTranscript(e.target.checked)} />
                  Show raw transcript (original script, as returned by Sarvam)
                </label>
                <div className="friction-list">
                  {segments.map((seg) => (
                    <div key={seg.segment_id} className="friction-card">
                      <div className="friction-meta">
                        <span className="friction-step-num">
                          {seg.start_time_ms != null && seg.end_time_ms != null
                            ? `${fmtMMSS(seg.start_time_ms / 1000)} – ${fmtMMSS(seg.end_time_ms / 1000)}`
                            : "untimed"}
                        </span>
                        <span className="friction-type">{seg.language || "unknown"}</span>
                      </div>
                      {/* Romanized display_transcript is the primary, editable text. */}
                      <textarea
                        className="textarea"
                        value={seg.display_transcript ?? seg.transcript}
                        onChange={(e) => editSegmentText(seg.segment_id, e.target.value)}
                        rows={2}
                        style={{fontSize: 13, lineHeight: 1.6, minHeight: 44}}
                      />
                      {/* Optional, subtle raw-transcript view -- original script, read-only evidence. */}
                      {showRawTranscript && seg.raw_transcript && (
                        <p className="rec-detail" style={{marginTop: 6, opacity: 0.6}} lang={seg.language || undefined}>
                          Raw: {seg.raw_transcript}
                        </p>
                      )}
                      {/* Phase 3: explicit UX classification -- visually secondary to the transcript itself. */}
                      {seg.classification?.labels?.length > 0 && (
                        <p className="form-hint" style={{marginTop: 8, letterSpacing: "1.5px"}}>
                          {seg.classification.labels.join(" · ")} · {seg.classification.confidence?.toUpperCase()}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
                <button className="tab" style={{marginTop: 16}} onClick={resetForRetry}>Record Again</button>
              </div>
            </div>
          )}

        </div>
      </section>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  /* Evaluation mode: "agent" (existing autonomous behaviour) or "human"
     (Phase 2: Human + Think-Aloud). These are kept strictly separate --
     the autonomous navigator is never invoked in "human" mode. */
  const [mode, setMode] = useState("agent");

  /* Setup */
  const [url, setUrl] = useState("https://example.com");
  const [task, setTask] = useState("Find the pricing section and understand the cost structure.");
  const [persona, setPersona] = useState({
    persona_type: "novice",
    name: "",
    age_range: "",
    tech_literacy: 4,
    primary_goal: "",
    device: "desktop",
    domain_familiarity: "first time",
    frustration_tolerance: "medium",
  });

  /* Run state */
  const [events, setEvents] = useState([]);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("live");
  const [report, setReport] = useState(null);
  const [scorecard, setScorecard] = useState(null);
  const [friction, setFriction] = useState(null);
  const [phase, setPhase] = useState("idle");
  const terminalRef = useRef(null);

  /* Derived */
  const steps = events.filter((e) => e.event === "node_complete");
  const latest = steps[steps.length - 1];
  const lastShot = [...steps].reverse().find((e) => e.screenshot_base64);

  useEffect(() => {
    if (terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [steps.length]);

  useEffect(() => {
    if (report) setActiveTab("report");
  }, [report]);

  const startEvaluation = async () => {
    setRunning(true); setEvents([]);
    setReport(null); setScorecard(null); setFriction(null);
    setPhase("running"); setActiveTab("live");

    try {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_url: url,
          task_description: task,
          persona: { ...persona, tech_literacy: parseInt(persona.tech_literacy, 10) },
        }),
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: dr } = await reader.read();
        done = dr;
        if (value) {
          const lines = dec.decode(value).split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const d = JSON.parse(line.slice(6));
                setEvents((p) => [...p, d]);
                if (d.event === "evaluating") setPhase("evaluating");
                if (d.event === "reporting") setPhase("reporting");
                if (d.event === "scorecard") setScorecard(d.scorecard);
                if (d.event === "friction") setFriction(d);
                if (d.event === "report") {
                  setReport(d.report);
                  if (d.scorecard) setScorecard(d.scorecard);
                  if (d.friction) setFriction({ episodes: d.friction.episodes, episode_count: d.friction.episode_count, signal_count: d.friction.signal_count });
                  setPhase("done");
                }
                if (d.event === "done" && phase !== "done") setPhase("done");
                if (d.event === "error") setPhase("error");
              } catch (_) {}
            }
          }
        }
      }
    } catch (e) {
      console.error(e); setPhase("error");
    } finally {
      setRunning(false);
    }
  };

  const cogLoad = latest?.cognitive_load ?? 0;
  const frustration = latest?.emotional_frustration ?? 0;
  const clickCount = steps.filter(e => e.action?.type === "click").length;
  const scrollCount = steps.filter(e => e.action?.type === "scroll").length;
  const errorCount = latest?.errors?.length ?? 0;

  /* ══════════════════════════════════════════════════════════
     RENDER
  ══════════════════════════════════════════════════════════ */
  return (
    <div className="app">

      {/* ── HEADER ── */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon"><LogoIcon /></div>
          <div className="logo-text-wrap">
            <span className="logo-name">Zynx</span>
            <span className="logo-sub">UX Intelligence Platform</span>
          </div>
        </div>

        <div className="header-center">
          {phase === "idle" && (
            <div className="persona-grid" style={{marginRight: "16px"}}>
              <button type="button"
                className={`persona-pill ${mode === "agent" ? "active" : ""}`}
                onClick={() => setMode("agent")} title="The system navigates the website automatically.">
                Autonomous Agent
              </button>
              <button type="button"
                className={`persona-pill ${mode === "human" ? "active" : ""}`}
                onClick={() => setMode("human")} title="You navigate the website manually while speaking aloud.">
                Human + Think-Aloud
              </button>
            </div>
          )}
          {phase !== "idle" && (
            <div className="phase-chips">
              {[
                { key: "running",   label: "Navigate" },
                { key: "evaluating",label: "Evaluate" },
                { key: "reporting", label: "Report" },
                { key: "done",      label: "Done" },
              ].map(({ key, label }) => (
                <div key={key} className={`phase-chip ${phase === key ? "active" : phaseDone(phase, key) ? "done" : ""}`}>
                  <span className="phase-chip-dot" />
                  {label}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="header-status">
          <span className={`status-dot ${phase === "idle" ? "idle" : phase === "error" ? "error" : phase === "done" ? "" : "running"}`} />
          {PHASE_LABELS[phase] || phase}
        </div>
      </header>

      <main className="main">
      {mode === "agent" ? (<>

        {/* ══════════════════════════════════════════════════════
            SIDEBAR
        ══════════════════════════════════════════════════════ */}
        <aside className="sidebar">

          {/* 01 — Target */}
          <div className="sidebar-section">
            <div className="sidebar-section-header">
              <span className="section-num">01</span>
              <span className="section-title">Target</span>
            </div>
            <div className="form-group">
              <label className="form-label">Website URL</label>
              <input className="input" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://example.com" />
            </div>
            <div className="form-group">
              <label className="form-label">Task Description</label>
              <textarea className="textarea" value={task} onChange={e => setTask(e.target.value)} rows={3}
                placeholder="What should the user try to accomplish?" />
              <span className="form-hint">Describe the goal from the user's perspective</span>
            </div>
          </div>

          {/* 02 — Persona */}
          <div className="sidebar-section">
            <div className="sidebar-section-header">
              <span className="section-num">02</span>
              <span className="section-title">Persona</span>
            </div>

            <div className="form-group">
              <label className="form-label">Name <span style={{color:"var(--steel-dim)"}}>optional</span></label>
              <input className="input" value={persona.name}
                onChange={e => setPersona({...persona, name: e.target.value})} placeholder="e.g. Priya" />
            </div>

            <div className="form-group">
              <label className="form-label">Expertise Level</label>
              <div className="persona-grid">
                {["novice","intermediate","expert"].map(t => (
                  <button key={t} type="button"
                    className={`persona-pill ${persona.persona_type === t ? "active" : ""}`}
                    onClick={() => setPersona({...persona, persona_type: t})}>
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Age Range</label>
              <input className="input" value={persona.age_range}
                onChange={e => setPersona({...persona, age_range: e.target.value})} placeholder="e.g. 25–34" />
            </div>

            <div className="form-group">
              <label className="form-label">
                Tech Literacy
                <span className="literacy-val">{persona.tech_literacy}</span>
              </label>
              <div className="range-wrap">
                <input type="range" className="range-input" min={1} max={10}
                  value={persona.tech_literacy}
                  onChange={e => setPersona({...persona, tech_literacy: e.target.value})} />
                <div className="range-labels"><span>Beginner</span><span>Expert</span></div>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Primary Goal</label>
              <input className="input" value={persona.primary_goal}
                onChange={e => setPersona({...persona, primary_goal: e.target.value})}
                placeholder="e.g. Find and compare pricing plans" />
            </div>
          </div>

          {/* 03 — Context */}
          <div className="sidebar-section">
            <div className="sidebar-section-header">
              <span className="section-num">03</span>
              <span className="section-title">Context</span>
            </div>

            <div className="form-group">
              <label className="form-label">Device</label>
              <div className="persona-grid">
                {[["desktop","🖥"], ["mobile","📱"], ["tablet","📟"]].map(([d, icon]) => (
                  <button key={d} type="button"
                    className={`persona-pill ${persona.device === d ? "active" : ""}`}
                    onClick={() => setPersona({...persona, device: d})}>
                    {icon} {d}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Domain Familiarity</label>
              <select className="select" value={persona.domain_familiarity}
                onChange={e => setPersona({...persona, domain_familiarity: e.target.value})}>
                {["first time","occasional user","regular user","power user"].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Frustration Tolerance</label>
              <div className="persona-grid">
                {["low","medium","high"].map(t => (
                  <button key={t} type="button"
                    className={`persona-pill ${persona.frustration_tolerance === t ? "active" : ""}`}
                    onClick={() => setPersona({...persona, frustration_tolerance: t})}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="sidebar-footer">
            <button id="run-btn" className="btn-run" onClick={startEvaluation} disabled={running}>
              {running ? <><div className="spinner" />{PHASE_LABELS[phase]}</> : "Run Evaluation →"}
            </button>
          </div>
        </aside>

        {/* ══════════════════════════════════════════════════════
            MAIN PANEL
        ══════════════════════════════════════════════════════ */}
        <section className="panel">

          {/* Tab bar */}
          <div className="tab-bar">
            <button className={`tab ${activeTab === "live" ? "active" : ""}`} onClick={() => setActiveTab("live")}>
              Live Run
              {steps.length > 0 && <span className="tab-badge">{steps.length}</span>}
            </button>
            <button className={`tab ${activeTab === "report" ? "active" : ""}`}
              onClick={() => setActiveTab("report")} disabled={!report && !scorecard}>
              Report
              {scorecard && <span className="tab-badge done-badge">✓</span>}
            </button>
          </div>

          {/* ── LIVE TAB ── */}
          {activeTab === "live" && (
            <div className="live-content">
              {steps.length === 0 && !running ? (
                <div className="empty-state">
                  <div className="empty-glyph">UX</div>
                  <h2 className="empty-title">Ready to evaluate.</h2>
                  <p className="empty-sub">Configure your persona on the left and run the evaluation to begin.</p>
                </div>
              ) : (
                <>
                  <div className="viewport-area">
                    {/* Screenshot */}
                    <div className="screenshot-pane">
                      <div className="screenshot-header">
                        <span className="url-bar">
                          <span className="url-scheme">https://</span>
                          {(latest?.current_url || url).replace(/^https?:\/\//, "")}
                        </span>
                        {steps.length > 0 && <span className="step-badge">step {steps.length}</span>}
                      </div>
                      <div className="screenshot-body">
                        {!lastShot ? (
                          <div className="screenshot-placeholder">
                            {running ? <><div className="spinner-lg" /><p>Browser loading…</p></> : <p>No screenshot yet.</p>}
                          </div>
                        ) : (
                          <img src={`data:image/png;base64,${lastShot.screenshot_base64}`}
                            className="screenshot-img" alt="live view" />
                        )}
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="metrics-pane">
                      {/* Big stat numbers */}
                      <div className="stat-wall">
                        <div className="stat-item">
                          <span className="stat-num">{steps.length}</span>
                          <span className="stat-label">Steps</span>
                        </div>
                        <div className="stat-item">
                          <span className={`stat-num ${latest?.task_completed ? "success" : "accent"}`}>
                            {latest?.task_completed ? "✓" : "…"}
                          </span>
                          <span className="stat-label">Task</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-num">{clickCount}</span>
                          <span className="stat-label">Clicks</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-num accent">{scrollCount}</span>
                          <span className="stat-label">Scrolls</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-num">{latest?.dom_count ?? "—"}</span>
                          <span className="stat-label">DOM</span>
                        </div>
                        <div className="stat-item">
                          <span className={`stat-num ${errorCount > 0 ? "warn" : "accent"}`}>
                            {errorCount}
                          </span>
                          <span className="stat-label">Errors</span>
                        </div>
                      </div>

                      {/* Gauges */}
                      <div className="gauges-section">
                        <div className="gauge-row">
                          <div className="gauge-header">
                            <span className="gauge-lbl">Cognitive Load</span>
                            <span className="gauge-pct">{(cogLoad * 100).toFixed(0)}%</span>
                          </div>
                          <div className="gauge-track">
                            <div className="gauge-fill cog" style={{width:`${cogLoad*100}%`}} />
                          </div>
                        </div>
                        <div className="gauge-row">
                          <div className="gauge-header">
                            <span className="gauge-lbl">Frustration</span>
                            <span className="gauge-pct">{(frustration * 100).toFixed(0)}%</span>
                          </div>
                          <div className="gauge-track">
                            <div className="gauge-fill frus" style={{width:`${frustration*100}%`}} />
                          </div>
                        </div>
                      </div>

                      {/* Phase banner during eval/report */}
                      {(phase === "evaluating" || phase === "reporting") && (
                        <div className="phase-banner">
                          <div className="spinner" />
                          {PHASE_LABELS[phase]}…
                        </div>
                      )}

                      {/* Working memory */}
                      {latest?.working_memory?.length > 0 && (
                        <div className="memory-section">
                          <div className="memory-title">Agent Memory</div>
                          {latest.working_memory.map((m, i) => (
                            <div key={i} className="memory-item">
                              <span className="memory-arrow">›</span>
                              <span>{m}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Terminal */}
                  <div className="terminal">
                    <div className="terminal-header">
                      <div className="terminal-dot red" />
                      <div className="terminal-dot yellow" />
                      <div className="terminal-dot green" />
                      <span className="terminal-title">Agent Thoughts</span>
                    </div>
                    <div className="terminal-body" ref={terminalRef}>
                      {steps.map((e, i) => {
                        const atype = e.action?.type || e.node || "navigate";
                        return (
                          <div key={i} className="log-line">
                            <span className="log-step">{String(i+1).padStart(2,"0")}</span>
                            <span className={`log-type ${LOG_TYPE[atype] || "navigate"}`}>{atype}</span>
                            <span className="log-thought">{e.thought_trace || ""}</span>
                          </div>
                        );
                      })}
                      {(phase === "evaluating" || phase === "reporting") && (
                        <div className="log-line">
                          <span className="log-step">⏳</span>
                          <span className="log-type eval">{phase}</span>
                          <span className="log-thought">{PHASE_LABELS[phase]}…</span>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── REPORT TAB ── */}
          {activeTab === "report" && (
            <div className="report-content">
              {!scorecard && !report ? (
                <div className="empty-state">
                  <div className="empty-glyph">—</div>
                  <h2 className="empty-title">No report yet.</h2>
                  <p className="empty-sub">Run an evaluation first to generate the full UX report.</p>
                </div>
              ) : (
                <>
                  {/* Overall banner */}
                  {scorecard && (
                    <div className="overall-banner">
                      <div>
                        <div className="overall-score-wrap">
                          <span className="overall-score-num">{scorecard.overall_ux_score ?? "—"}</span>
                          <span className="overall-score-denom">/10</span>
                        </div>
                        <div className="overall-score-label">Overall UX Score</div>
                      </div>
                      <div className="overall-meta">
                        <div className="overall-url">{url.replace(/^https?:\/\//, "")}</div>
                        <div className="overall-tags">
                          {persona.persona_type && <span className="overall-tag">{persona.persona_type}</span>}
                          {persona.device && <span className="overall-tag">{persona.device}</span>}
                          {persona.name && <span className="overall-tag">{persona.name}</span>}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Score rings */}
                  {scorecard && (
                    <div className="report-block">
                      <span className="report-block-num">01</span>
                      <div className="report-block-title">
                        <em>Scorecard</em>
                        <span className="report-block-title-en">UX Dimensions</span>
                      </div>
                      <div className="rings-row">
                        {[
                          { key: "effectiveness", label: "Effectiveness", color: "#4B5694" },
                          { key: "efficiency",    label: "Efficiency",    color: "#7288AE" },
                          { key: "learnability",  label: "Learnability",  color: "#6875B8" },
                          { key: "cognitive_load",label: "Cog. Load",     color: "#8B7AAE" },
                        ].map(({ key, label, color }) => (
                          <ScoreRing key={key}
                            value={scorecard[key]?.score}
                            label={label} color={color}
                            rationale={scorecard[key]?.rationale}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* System 1 / System 2 */}
                  {scorecard?.system1_system2_ratio && (
                    <div className="report-block">
                      <span className="report-block-num">02</span>
                      <div className="report-block-title">
                        <em>Thinking Mode</em>
                        <span className="report-block-title-en">System 1 vs System 2</span>
                      </div>
                      <div className="s1s2-wrap">
                        <div className="s1s2-bar">
                          <div className="s1-fill" style={{width:`${scorecard.system1_system2_ratio.system1_percent}%`}}>
                            System 1 · {scorecard.system1_system2_ratio.system1_percent}%
                          </div>
                          <div className="s2-fill">
                            {scorecard.system1_system2_ratio.system2_percent}% · System 2
                          </div>
                        </div>
                        {scorecard.system1_system2_ratio.rationale && (
                          <p className="s1s2-rationale">{scorecard.system1_system2_ratio.rationale}</p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Biases */}
                  {scorecard?.psychological_biases && (
                    <div className="report-block">
                      <span className="report-block-num">03</span>
                      <div className="report-block-title">
                        <em>Psychological Biases</em>
                        <span className="report-block-title-en">Cognitive Patterns</span>
                      </div>
                      <div className="bias-grid">
                        {Object.entries(scorecard.psychological_biases).map(([key, val]) => (
                          <div key={key} className="bias-row">
                            <span className="bias-key">{key.replace(/_/g, " ")}</span>
                            <div className="bias-track">
                              <div className="bias-fill" style={{width:`${(val/10)*100}%`}} />
                            </div>
                            <span className="bias-val">{val}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Friction points */}
                  {scorecard?.friction_points?.length > 0 && (
                    <div className="report-block">
                      <span className="report-block-num">04</span>
                      <div className="report-block-title">
                        <em>Friction Points</em>
                        <span className="report-block-title-en">Where users struggle</span>
                      </div>
                      <div className="friction-list">
                        {scorecard.friction_points.map((fp, i) => {
                          const sevColor = {low:"#82c4a0", medium:"#c8a06e", high:"#c85c6e"};
                          return (
                            <div key={i} className="friction-card">
                              <div className="friction-meta">
                                <span className="friction-step-num">
                                  {fp.step != null ? `step ${fp.step}` : "—"}
                                </span>
                                <span className="friction-type">{fp.type}</span>
                                <div className="friction-sev-dot" style={{background: sevColor[fp.severity] || "#7288AE"}} />
                                <span style={{fontSize:"9px", color: sevColor[fp.severity] || "#7288AE", fontWeight:700, textTransform:"uppercase", letterSpacing:"1px"}}>{fp.severity}</span>
                              </div>
                              <p className="friction-desc">{fp.description}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Behavioural Friction (Phase 1) */}
                  {friction?.episodes?.length > 0 && (
                    <div className="report-block">
                      <span className="report-block-num">05</span>
                      <div className="report-block-title">
                        <em>Behavioural Friction</em>
                        <span className="report-block-title-en">
                          {friction.episode_count} episode{friction.episode_count === 1 ? "" : "s"} · {friction.signal_count} signal{friction.signal_count === 1 ? "" : "s"} detected
                        </span>
                      </div>
                      <div className="friction-list">
                        {friction.episodes.map((ep, i) => {
                          const sevColor = {low:"#82c4a0", medium:"#c8a06e", high:"#c85c6e"};
                          const fmt = (ms) => {
                            const s = Math.round(ms / 1000);
                            return `${String(Math.floor(s / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
                          };
                          return (
                            <div key={ep.episode_id || i} className="friction-card">
                              <div className="friction-meta">
                                <span className="friction-step-num">Episode {String(i+1).padStart(2,"0")}</span>
                                <span className="friction-type">
                                  {fmt(ep.start_time_ms)} – {fmt(ep.end_time_ms)} · {Math.round(ep.duration_ms/1000)}s
                                </span>
                                <div className="friction-sev-dot" style={{background: sevColor[ep.severity] || "#7288AE"}} />
                                <span style={{fontSize:"9px", color: sevColor[ep.severity] || "#7288AE", fontWeight:700, textTransform:"uppercase", letterSpacing:"1px"}}>{ep.severity}</span>
                              </div>
                              <p className="friction-desc">{ep.signal_types.join(", ").replace(/_/g, " ")}</p>
                              {ep.evidence_summary?.map((line, j) => (
                                <p key={j} className="friction-desc" style={{opacity:0.75, fontSize:"12px"}}>• {line}</p>
                              ))}
                              {ep.affected_elements?.length > 0 && (
                                <p className="friction-desc" style={{opacity:0.6, fontSize:"11px"}}>
                                  Affected: {ep.affected_elements.join(", ")}
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Report narrative */}
                  {report?.executive_summary && (
                    <div className="report-block">
                      <span className="report-block-num">06</span>
                      <div className="report-block-title">
                        <em>Executive Summary</em>
                      </div>
                      <p className="report-para">{report.executive_summary}</p>
                    </div>
                  )}

                  {report?.psychological_friction_analysis && (
                    <div className="report-block">
                      <span className="report-block-num">07</span>
                      <div className="report-block-title">
                        <em>Friction Analysis</em>
                        <span className="report-block-title-en">Cognitive Psychology</span>
                      </div>
                      <p className="report-para">{report.psychological_friction_analysis}</p>
                    </div>
                  )}

                  {/* Recommendations */}
                  {report?.key_recommendations?.length > 0 && (
                    <div className="report-block">
                      <span className="report-block-num">08</span>
                      <div className="report-block-title">
                        <em>Recommendations</em>
                        <span className="report-block-title-en">Actionable Fixes</span>
                      </div>
                      <div className="rec-list">
                        {report.key_recommendations.map((r, i) => (
                          <div key={i} className="rec-item">
                            <span className="rec-num">{String(i+1).padStart(2,"0")}</span>
                            <div className="rec-body">
                              <div className={`rec-priority ${r.priority}`}>{r.priority} priority</div>
                              <div className="rec-title">{r.title}</div>
                              <div className="rec-detail">{r.detail}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* What worked / Issues */}
                  {scorecard && (scorecard.top_positives?.length > 0 || scorecard.top_issues?.length > 0) && (
                    <div className="report-block">
                      <span className="report-block-num">09</span>
                      <div className="report-block-title">
                        <em>Summary</em>
                        <span className="report-block-title-en">Findings</span>
                      </div>
                      <div className="summary-two-col">
                        {scorecard.top_positives?.length > 0 && (
                          <div className="sum-col">
                            <div className="sum-col-title">What worked</div>
                            {scorecard.top_positives.map((p, i) => (
                              <div key={i} className="sum-item">
                                <span className="sum-bullet" style={{color:"#82c4a0"}}>+</span>
                                {p}
                              </div>
                            ))}
                          </div>
                        )}
                        {scorecard.top_issues?.length > 0 && (
                          <div className="sum-col">
                            <div className="sum-col-title">Issues found</div>
                            {scorecard.top_issues.map((p, i) => (
                              <div key={i} className="sum-item">
                                <span className="sum-bullet" style={{color:"#c8a06e"}}>–</span>
                                {p}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

        </section>
      </>) : (
        <ThinkAloudPanel url={url} setUrl={setUrl} task={task} setTask={setTask} persona={persona} setPersona={setPersona} />
      )}
      </main>
    </div>
  );
}
