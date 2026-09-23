import React, { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

// Formats a plain amplitude number for the axis labels (never dB — this is
// linear amplitude vs time, so it can legitimately be negative).
function formatAmp(v) {
    if (Math.abs(v) < 0.0005) return "0";
    return `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
}

export default function EffectAnimator({ visualization, mode = "echo", onComplete, loop = true, delayMs, decay, repeats, formula, paramLabel, operationLabel, outputCaption, progressLabel }) {
    const input = visualization?.time?.input || [];
    const output = visualization?.time?.output || [];
    const [progress, setProgress] = useState(0);
    const completed = useRef(false);
    // A ref, not a dependency — onComplete is typically a fresh inline
    // function every parent render, and re-running this effect on that
    // change would restart the whole animation from 0 right after it
    // naturally finishes (the completion itself often causes that re-render).
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    useEffect(() => {
        if (!input.length || !output.length) return undefined;
        let raf;
        let start = performance.now();
        const duration = 7600;

        const tick = (now) => {
            const p = clamp((now - start) / duration);
            setProgress(p);

            if (p >= 1) {
                if (!completed.current) {
                    completed.current = true;
                    onCompleteRef.current?.();
                }
                if (loop) {
                    start = now + 900;
                    setProgress(0);
                    raf = requestAnimationFrame(tick);
                }
                return;
            }
            raf = requestAnimationFrame(tick);
        };

        completed.current = false;
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [input.length, output.length, loop]);

    // Plot geometry — a real axis (0 line + labeled +/- range), not just a
    // bare centerline, so the natural positive/negative swing of a genuine
    // amplitude-vs-time waveform reads as normal rather than "wrong".
    const width = 900;
    const height = 260;
    const pad = { left: 46, right: 20, top: 16, bottom: 16 };
    const plotTop = pad.top;
    const plotBottom = height - pad.bottom;
    const midY = (plotTop + plotBottom) / 2;
    const halfH = (plotBottom - plotTop) / 2;

    const max = useMemo(() => Math.max(
        ...input.map(v => Math.abs(v)),
        ...output.map(v => Math.abs(v)),
        1e-6
    ), [input, output]);

    const toX = (i, len) => pad.left + (i / Math.max(1, len - 1)) * (width - pad.left - pad.right);
    const toY = (v) => midY - (v / max) * halfH * 0.92;

    const polyline = (values, end) => values
        .slice(0, Math.max(2, Math.floor(end * values.length)))
        .map((v, i) => `${toX(i, values.length).toFixed(1)},${toY(v).toFixed(1)}`)
        .join(" ");

    const reveal = progress < .5 ? progress * 2 : 1;
    const outputReveal = progress < .35 ? 0 : (progress - .35) / .65;
    const marker = toX(reveal, 1);
    const markerOut = toX(outputReveal, 1);

    const renderAxis = () => (
        <g>
            {/* zero-amplitude reference line */}
            <line x1={pad.left} x2={width - pad.right} y1={midY} y2={midY} className="conv-axis" />
            {/* labeled +/- ticks so the natural bipolar swing is legible */}
            {[1, 0.5, 0, -0.5, -1].map((f) => {
                const y = midY - f * halfH * 0.92;
                return (
                    <g key={f}>
                        <line x1={pad.left - 4} x2={pad.left} y1={y} y2={y} stroke="rgba(255,255,255,.35)" />
                        <text x={pad.left - 8} y={y} fill="rgba(255,255,255,.45)" fontSize="10" fontFamily="ui-monospace, monospace" textAnchor="end" dominantBaseline="middle">
                            {formatAmp(f * max)}
                        </text>
                    </g>
                );
            })}
        </g>
    );

    const resolvedParamLabel = paramLabel ?? (mode === "echo"
        ? `D = ${Math.round(delayMs ?? 280)} ms   α = ${(decay ?? 0.55).toFixed(2)}   repeats = ${repeats ?? 5}`
        : `D = ${Math.round(delayMs ?? 350)} ms   mix = ${(decay ?? 0.85).toFixed(2)}`);

    return (
        <div className="effect-animator">
            <div className="effect-stage">
                <div className="effect-stage-label">
                    <span>INPUT x[n]</span>
                    <small>amplitude vs time · original signal</small>
                </div>
                <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                    {renderAxis()}
                    <polyline points={polyline(input, reveal)} fill="none" stroke="#4ea1ff" strokeWidth="3" vectorEffect="non-scaling-stroke" />
                    <line x1={marker} x2={marker} y1={plotTop} y2={plotBottom} className="conv-playhead" />
                    <circle cx={marker} cy={midY} r="6" fill="#4ea1ff" stroke="#fff" strokeWidth="2" />
                </svg>
            </div>

            <div className="effect-operation">
                <span>{operationLabel ?? (mode === "echo" ? "FEEDBACK DELAY (IIR)" : "SINGLE-TAP DELAY (FIR)")}</span>
                <strong>{formula ?? (mode === "echo" ? "y[n] = x[n] + α·y[n−D]" : "y[n] = x[n] + α·x[n−D]")}</strong>
                <small style={{ color: "rgba(255,255,255,.5)", fontSize: "10px", letterSpacing: ".5px" }}>{resolvedParamLabel}</small>
            </div>

            <div className="effect-stage output">
                <div className="effect-stage-label">
                    <span>OUTPUT y[n]</span>
                    <small>amplitude vs time · {outputCaption ?? (mode === "echo" ? "decaying repeats fed back in" : "original + one delayed copy")}</small>
                </div>
                <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                    {renderAxis()}
                    <polyline points={polyline(output, outputReveal)} fill="none" stroke="#ff5d6c" strokeWidth="3" vectorEffect="non-scaling-stroke" />
                    {outputReveal > 0 && (
                        <>
                            <line x1={markerOut} x2={markerOut} y1={plotTop} y2={plotBottom} className="conv-playhead" />
                            <circle cx={markerOut} cy={midY} r="6" fill="#ff5d6c" stroke="#fff" strokeWidth="2" />
                        </>
                    )}
                </svg>
            </div>

            <div className="frequency-progress">
                <div>
                    <span>{progressLabel ?? (mode === "echo" ? "Feeding the delayed output back into itself…" : "Moving samples through the delay line…")}</span>
                    <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="frequency-progress-track"><span style={{ width: `${progress * 100}%` }} /></div>
            </div>
        </div>
    );
}