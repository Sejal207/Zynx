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

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
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
    setReport(null); setScorecard(null);
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
                if (d.event === "report") {
                  setReport(d.report);
                  if (d.scorecard) setScorecard(d.scorecard);
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

                  {/* Report narrative */}
                  {report?.executive_summary && (
                    <div className="report-block">
                      <span className="report-block-num">05</span>
                      <div className="report-block-title">
                        <em>Executive Summary</em>
                      </div>
                      <p className="report-para">{report.executive_summary}</p>
                    </div>
                  )}

                  {report?.psychological_friction_analysis && (
                    <div className="report-block">
                      <span className="report-block-num">06</span>
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
                      <span className="report-block-num">07</span>
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
                      <span className="report-block-num">08</span>
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
      </main>
    </div>
  );
}
