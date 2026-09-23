import { useEffect, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

const STAGE_TEXT = {
    pitch: {
        operation: "SHIFT BINS",
        detail: "Phase-vocoder time-stretch, then resample — the magnitude bins move to new frequencies while the duration stays put.",
    },
    stretch: {
        operation: "RESPACE FRAMES",
        detail: "The same phase vocoder without the resampling step — frames are respaced in time, so the duration changes but every bin stays at its own frequency.",
    },
    robot: {
        operation: "ZERO THE PHASE",
        detail: "Every bin keeps its magnitude and has its phase set to 0. Nothing about |X(f)| changes, yet the voice becomes a flat monotone buzz — that difference is what phase was carrying.",
    },
    whisper: {
        operation: "RANDOMIZE THE PHASE",
        detail: "Every bin keeps its magnitude and gets a random phase. Same |X(f)| again, but the result is breathy and unvoiced.",
    },
};

// The "process" view for Module 6. Two stacked rows — magnitude and phase —
// each showing input beside output, so the thing the module is actually
// about (magnitude and phase carry different information, and can be
// manipulated independently) is visible directly: for robot/whisper the top
// row is identical on both sides while the bottom row is completely rewritten.
export default function MorphAnimator({ morph, onComplete, loop = false }) {
    const mode = morph?.mode || "pitch";
    const inMag = morph?.magnitude?.input || [];
    const outMag = morph?.magnitude?.output || [];
    const inPhase = morph?.phase?.input || [];
    const outPhase = morph?.phase?.output || [];
    const maxFrequency = morph?.maxFrequency || 0;
    const displayMax = morph?.displayMax || 1e-9;
    const count = Math.max(inMag.length, outMag.length);

    const [progress, setProgress] = useState(0);
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    useEffect(() => {
        if (!count) return undefined;
        let raf;
        let start = performance.now();
        let completedOnce = false;
        const duration = 7000;

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
        return <div className="spectrum-empty">Run the morph first — the spectrum is calculated by the Python backend.</div>;
    }

    // Four phases mirroring the module's own stages: frame the signal,
    // read spectrum + phase, apply the morph, reconstruct.
    const phase =
        progress < 0.2 ? "frame" :
        progress < 0.5 ? "analyze" :
        progress < 0.78 ? "morph" : "reconstruct";

    const analyzeReveal = phase === "frame" ? 0 : phase === "analyze" ? clamp((progress - 0.2) / 0.3) : 1;
    const outputReveal = phase === "morph" ? clamp((progress - 0.5) / 0.28) : phase === "reconstruct" ? 1 : 0;

    const inputRevealed = Math.floor(analyzeReveal * count);
    const outputRevealed = Math.floor(outputReveal * count);

    const stage = STAGE_TEXT[mode] || STAGE_TEXT.pitch;

    // Phase is drawn as a bar growing up or down from a centre line, since
    // it is signed and wraps within [-pi, +pi].
    const phaseRow = (values, revealedCount, color) => (
        <div className="spectrum-plot" style={{ height: "110px" }}>
            <div className="spectrum-zero" style={{ top: "50%", bottom: "auto" }} />
            {Array.from({ length: count }, (_, i) => {
                const value = Number(values[i] || 0);
                const frac = clamp(Math.abs(value) / Math.PI, 0, 1);
                const height = Math.max(1.5, frac * 44);
                const visible = i < revealedCount;
                return (
                    <span
                        key={i}
                        className={`spectrum-stem ${visible ? "is-visible" : ""}`}
                        style={{
                            left: `${((i + 0.5) / count) * 100}%`,
                            height: `${height}%`,
                            bottom: value >= 0 ? "50%" : "auto",
                            top: value >= 0 ? "auto" : "50%",
                            "--spectrum-color": color,
                        }}
                    >
                        <i />
                    </span>
                );
            })}
        </div>
    );

    const magRow = (values, revealedCount, color) => (
        <div className="spectrum-plot" style={{ height: "110px" }}>
            <div className="spectrum-zero" />
            {Array.from({ length: count }, (_, i) => {
                const value = Number(values[i] || 0);
                const height = clamp(value / displayMax) * 88;
                const visible = i < revealedCount;
                return (
                    <span
                        key={i}
                        className={`spectrum-stem ${visible ? "is-visible" : ""}`}
                        style={{
                            left: `${((i + 0.5) / count) * 100}%`,
                            height: `${Math.max(2, height)}%`,
                            "--spectrum-color": color,
                        }}
                    >
                        <i />
                    </span>
                );
            })}
        </div>
    );

    return (
        <div className="frequency-animator">
            <div className="frequency-explanation">
                <div>
                    <span className="control-kicker">PHASE VOCODER · STFT FRAME</span>
                    <strong>X(f) = |X(f)| · e^{"{ j∠X(f) }"}</strong>
                </div>
                <span className="frequency-status">
                    {phase === "frame" && "Framing the signal…"}
                    {phase === "analyze" && "Reading magnitude and phase…"}
                    {phase === "morph" && `${stage.operation}…`}
                    {phase === "reconstruct" && "Reconstructing with inverse STFT…"}
                </span>
            </div>

            <p className="frequency-note">{stage.detail}</p>

            <div className="frequency-stage-grid two-col">
                <div className="spectrum-stage input">
                    <div className="spectrum-stage-head">
                        <span>INPUT MAGNITUDE |X(f)|</span>
                        <small>MAGNITUDE</small>
                    </div>
                    {magRow(inMag, inputRevealed, "#4ea1ff")}
                    <div className="spectrum-stage-head" style={{ marginTop: "14px" }}>
                        <span>INPUT PHASE ∠X(f)</span>
                        <small>−π … +π</small>
                    </div>
                    {phaseRow(inPhase, inputRevealed, "#b66cff")}
                    <div className="spectrum-axis">
                        <span>0 Hz</span>
                        <span>{Math.round(maxFrequency).toLocaleString()} Hz</span>
                    </div>
                </div>

                <div className="spectrum-stage output">
                    <div className="spectrum-stage-head">
                        <span>OUTPUT MAGNITUDE |Y(f)|</span>
                        <small>MAGNITUDE</small>
                    </div>
                    {magRow(outMag, outputRevealed, "#ff5d6c")}
                    <div className="spectrum-stage-head" style={{ marginTop: "14px" }}>
                        <span>OUTPUT PHASE ∠Y(f)</span>
                        <small>−π … +π</small>
                    </div>
                    {phaseRow(outPhase, outputRevealed, "#f7b801")}
                    <div className="spectrum-axis">
                        <span>0 Hz</span>
                        <span>{Math.round(maxFrequency).toLocaleString()} Hz</span>
                    </div>
                </div>
            </div>

            <div className="effect-operation" style={{ marginTop: "16px" }}>
                <span>STAGE {phase === "frame" ? "1 / 4 · FRAME SIGNAL" : phase === "analyze" ? "2 / 4 · SPECTRUM + PHASE" : phase === "morph" ? `3 / 4 · ${stage.operation}` : "4 / 4 · RECONSTRUCT"}</span>
                <strong>
                    {mode === "robot" && "∠Y(f) = 0   ·   |Y(f)| = |X(f)|"}
                    {mode === "whisper" && "∠Y(f) = random   ·   |Y(f)| = |X(f)|"}
                    {mode === "pitch" && "bins scaled in frequency, duration preserved"}
                    {mode === "stretch" && "frames respaced in time, frequencies preserved"}
                </strong>
            </div>

            <div className="frequency-progress">
                <div>
                    <span>Frame → Spectrum + phase → {stage.operation} → Reconstruct</span>
                    <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="frequency-progress-track">
                    <span style={{ width: `${progress * 100}%` }} />
                </div>
            </div>
        </div>
    );
}
