import { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

// The "process" view for Module 4: sweeps left to right revealing the
// backend's per-frame short-time energy curve against its threshold,
// coloring each revealed frame green (active) or grey (quiet) — the same
// classification the backend used to gate the audio. Plays once, then
// freezes fully revealed (a parent-supplied `key` forces a fresh replay).
export default function ActivityAnimator({ activity, onComplete, loop = false }) {
    const energy = activity?.energy || [];
    const active = activity?.active || [];
    const threshold = activity?.threshold || 0;
    const peak = activity?.peak || 0;

    const [progress, setProgress] = useState(0);
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    const count = energy.length;

    useEffect(() => {
        if (!count) return undefined;
        let raf;
        let start = performance.now();
        let completedOnce = false;
        const duration = 6200;

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

    const width = 900;
    const height = 260;
    const pad = { left: 46, right: 20, top: 20, bottom: 30 };
    const plotTop = pad.top;
    const plotBottom = height - pad.bottom;
    const plotLeft = pad.left;
    const plotRight = width - pad.right;

    const maxEnergy = useMemo(() => Math.max(peak, ...energy, 1e-9), [energy, peak]);

    const toX = (i) => plotLeft + (i / Math.max(1, count - 1)) * (plotRight - plotLeft);
    const toY = (e) => plotBottom - clamp(e / maxEnergy) * (plotBottom - plotTop);

    const revealed = Math.max(1, Math.floor(progress * count));
    const cursorX = toX(Math.min(count - 1, revealed - 1));

    const thresholdY = toY(threshold);

    if (!count) {
        return <div className="spectrum-empty">Run the analysis first — activity is calculated by the Python backend.</div>;
    }

    const barWidth = Math.max(1.5, (plotRight - plotLeft) / count - 1);

    return (
        <div className="activity-animator">
            <div className="effect-stage" style={{ gridColumn: "1 / -1" }}>
                <div className="effect-stage-label">
                    <span>SHORT-TIME ENERGY vs. TIME</span>
                    <small>each frame's RMS energy, framed then measured — green = active, grey = quiet</small>
                </div>
                <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                    <line x1={plotLeft} x2={plotRight} y1={plotBottom} y2={plotBottom} className="conv-axis" />

                    {/* Threshold reference line */}
                    <line
                        x1={plotLeft} x2={plotRight} y1={thresholdY} y2={thresholdY}
                        stroke="rgba(255,93,108,.55)" strokeDasharray="5 5" vectorEffect="non-scaling-stroke"
                    />
                    <text x={plotRight} y={thresholdY - 6} fill="rgba(255,93,108,.75)" fontSize="10" fontFamily="ui-monospace, monospace" textAnchor="end">
                        THRESHOLD
                    </text>

                    {Array.from({ length: count }, (_, i) => {
                        if (i >= revealed) return null;
                        const x = toX(i);
                        const y = toY(energy[i]);
                        const isActive = !!active[i];
                        return (
                            <rect
                                key={i}
                                x={x - barWidth / 2}
                                y={y}
                                width={barWidth}
                                height={Math.max(1, plotBottom - y)}
                                fill={isActive ? "#3ddc84" : "rgba(255,255,255,.18)"}
                                opacity={isActive ? 0.85 : 0.6}
                            />
                        );
                    })}

                    {progress < 1 && (
                        <line x1={cursorX} x2={cursorX} y1={plotTop} y2={plotBottom} className="conv-playhead" />
                    )}
                </svg>
            </div>

            <div className="effect-operation">
                <span>SHORT-TIME ENERGY (STE) VOICE/ACTIVITY DETECTION</span>
                <strong>active[n] = STE[n] &gt; threshold</strong>
                <small style={{ color: "rgba(255,255,255,.5)", fontSize: "10px", letterSpacing: ".5px" }}>
                    {count} frames · {(active.filter(Boolean).length / count * 100).toFixed(0)}% marked active
                </small>
            </div>

            <div className="frequency-progress">
                <div>
                    <span>Framing the signal and measuring energy…</span>
                    <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="frequency-progress-track"><span style={{ width: `${progress * 100}%` }} /></div>
            </div>
        </div>
    );
}
