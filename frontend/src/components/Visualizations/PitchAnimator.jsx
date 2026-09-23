import { useEffect, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

// The "process" view for Module 5: reveals the spectrum bars left to right,
// then zooms in on the detected peak bin, then fades in the note readout —
// tracing exactly the module's stages (Spectrum → Find peak → Estimate
// pitch). Plays once, then freezes on the finished note card (a
// parent-supplied `key` forces a fresh replay).
export default function PitchAnimator({ pitch, onComplete, loop = false }) {
    const bars = pitch?.magnitude || [];
    const maxFrequency = pitch?.maxFrequency || 0;
    const peakFrequency = pitch?.peakFrequency || 0;
    const peakBinIndex = pitch?.peakBinIndex ?? 0;
    const note = pitch?.note;
    const count = bars.length;

    const [progress, setProgress] = useState(0);
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    useEffect(() => {
        if (!count) return undefined;
        let raf;
        let start = performance.now();
        let completedOnce = false;
        const duration = 6400;

        const tick = (now) => {
            const p = clamp((now - start) / duration);
            setProgress(p);

            if (p >= 1) {
                if (!completedOnce) {
                    completedOnce = true;
                    onCompleteRef.current?.();
                }
                if (loop) {
                    start = now + 900;
                    completedOnce = false;
                    setProgress(0);
                    raf = requestAnimationFrame(tick);
                }
                return;
            }
            raf = requestAnimationFrame(tick);
        };

        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [count, loop]);

    if (!count) {
        return <div className="spectrum-empty">Run the detection first — the spectrum is calculated by the Python backend.</div>;
    }

    // Three phases: reveal the spectrum (0-0.55), highlight the peak bin
    // (0.55-0.75), fade in the note card (0.75-1.0).
    const phase = progress < 0.55 ? "spectrum" : progress < 0.75 ? "peak" : "pitch";
    const revealFrac = phase === "spectrum" ? progress / 0.55 : 1;
    const revealed = Math.max(1, Math.floor(clamp(revealFrac) * count));
    const peakHighlight = phase !== "spectrum";
    const showNote = phase === "pitch";
    const noteOpacity = phase === "pitch" ? clamp((progress - 0.75) / 0.25) : 0;

    const sharedMax = bars.reduce((m, v) => Math.max(m, Number(v) || 0), 1e-9);
    const cents = note?.cents ?? 0;
    const centsLabel = Math.abs(cents) < 1 ? "in tune" : `${cents > 0 ? "+" : ""}${cents.toFixed(0)} cents`;

    return (
        <div className="frequency-animator">
            <div className="frequency-explanation">
                <div>
                    <span className="control-kicker">BACKEND FFT ANALYSIS</span>
                    <strong>f₀ = argmax |X(f)|  (searched within the selected range)</strong>
                </div>
                <span className="frequency-status">
                    {phase === "spectrum" && "Reading spectral bins…"}
                    {phase === "peak" && "Searching for the strongest bin…"}
                    {phase === "pitch" && "Mapping frequency to the nearest musical note…"}
                </span>
            </div>

            <div className="frequency-stage-grid">
                <div className="spectrum-stage input" style={{ gridColumn: "1 / -1" }}>
                    <div className="spectrum-stage-head">
                        <span>MAGNITUDE SPECTRUM |X(f)|</span>
                        <small>MAGNITUDE</small>
                    </div>
                    <div className="spectrum-plot">
                        <div className="spectrum-zero" />
                        {Array.from({ length: count }, (_, i) => {
                            const value = Number(bars[i] || 0);
                            const height = clamp(value / sharedMax) * 88;
                            const visible = i < revealed;
                            const isPeak = i === peakBinIndex;
                            const active = isPeak && peakHighlight;
                            return (
                                <span
                                    key={i}
                                    className={`spectrum-stem ${visible ? "is-visible" : ""} ${active ? "is-active" : ""}`}
                                    style={{
                                        left: `${((i + 0.5) / count) * 100}%`,
                                        height: `${Math.max(2, height)}%`,
                                        "--spectrum-color": active ? "#f7b801" : "#4ea1ff",
                                    }}
                                >
                                    <i />
                                </span>
                            );
                        })}
                        {peakHighlight && (
                            <span
                                className="spectrum-cursor"
                                style={{
                                    left: `${((peakBinIndex + 0.5) / count) * 100}%`,
                                    "--spectrum-color": "#f7b801",
                                }}
                            />
                        )}
                    </div>
                    <div className="spectrum-axis">
                        <span>0 Hz</span>
                        <span>{Math.round(maxFrequency).toLocaleString()} Hz</span>
                    </div>
                </div>
            </div>

            <div
                className="math-visual"
                style={{
                    marginTop: "18px",
                    opacity: showNote ? noteOpacity : 0,
                    transition: "opacity .3s ease",
                    pointerEvents: showNote ? "auto" : "none",
                }}
            >
                <div className="math-formula" style={{ flexDirection: "column", gap: "6px" }}>
                    <span style={{ fontSize: "13px", letterSpacing: "2px", color: "rgba(255,255,255,.55)" }}>
                        DOMINANT FREQUENCY
                    </span>
                    <span style={{ fontSize: "34px", fontWeight: 800 }}>
                        {peakFrequency.toFixed(1)} Hz
                    </span>
                    <span style={{ fontSize: "22px", color: "#f7b801", fontWeight: 800 }}>
                        ≈ {note?.note ?? "-"}{note?.octave ?? ""}
                        <span style={{ fontSize: "13px", color: "rgba(255,255,255,.55)", marginLeft: "10px" }}>
                            {centsLabel}
                        </span>
                    </span>
                </div>
            </div>

            <div className="frequency-progress">
                <div>
                    <span>
                        {phase === "spectrum" && "Building the magnitude spectrum…"}
                        {phase === "peak" && "Locking onto the loudest frequency bin…"}
                        {phase === "pitch" && "Estimating the musical pitch…"}
                    </span>
                    <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="frequency-progress-track">
                    <span style={{ width: `${progress * 100}%` }} />
                </div>
            </div>
        </div>
    );
}
