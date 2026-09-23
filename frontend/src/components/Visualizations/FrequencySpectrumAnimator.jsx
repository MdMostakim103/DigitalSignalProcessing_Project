import React, { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

// Audio spectra cover an enormous range — a 20 Hz-18 kHz sweep through a
// 10th-order low-pass spans ~160 dB between its loudest and quietest bin. Map
// that linearly onto bar height and essentially every bar lands on the chart
// floor, so the input and output panels look identical no matter what the
// processing did. Heights are therefore logarithmic, clipped at a visible
// noise floor, which is also simply how spectra are read in practice.
const DB_FLOOR = -80;

function dbHeight(value, reference) {
    if (!(value > 0) || !(reference > 0)) return 0;
    const db = 20 * Math.log10(value / reference);
    return clamp((db - DB_FLOOR) / -DB_FLOOR) * 88;
}

export default function FrequencySpectrumAnimator({
    visualization,
    mode = "amplify",
    loop = true,
    onComplete,
    gain = 1,
}) {
    const isTwoPanel = mode === "echo" || mode === "delay" || mode === "activity" || mode === "pitch" || mode === "morph";

    const source = visualization?.convolution?.frequency || visualization?.frequency;
    const input = source?.input;
    const middle = source?.filterResponse || source?.impulseResponse || source?.input;
    const output = source?.output;

    const inputBars = input?.magnitude || [];
    const middleBars = middle?.magnitude || [];
    const outputBars = output?.magnitude || [];
    const multipliedBars = useMemo(
        () => inputBars.map(value => value * Math.max(0.01, Number(gain) || 1)),
        [inputBars, gain]
    );
    const count = Math.max(inputBars.length, middleBars.length, outputBars.length, multipliedBars.length);

    const [progress, setProgress] = useState(0);
    const TWO_PANEL_SPLIT = 0.45;
    // A ref, not a dependency — onComplete is typically a fresh inline
    // function every parent render, and re-running this effect on that
    // change would restart the animation right after it naturally finishes.
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    useEffect(() => {
        if (!count) return undefined;
        let raf;
        let start = performance.now();
        let completedOnce = false;

        const duration = mode === "convolution" ? 7600 : isTwoPanel ? 4200 : 5200;

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
    }, [count, mode, loop, isTwoPanel]);

    const phase = isTwoPanel
        ? (progress < TWO_PANEL_SPLIT ? "input" : "output")
        : (progress < 0.28 ? "input" : progress < 0.62 ? "multiply" : "output");
    const local = isTwoPanel
        ? (phase === "input" ? progress / TWO_PANEL_SPLIT : (progress - TWO_PANEL_SPLIT) / (1 - TWO_PANEL_SPLIT))
        : (phase === "input"
            ? progress / 0.28
            : phase === "multiply"
                ? (progress - 0.28) / 0.34
                : (progress - 0.62) / 0.38);

    const revealed = Math.max(1, Math.floor(clamp(local) * count));
    const cursor = Math.min(count - 1, Math.floor(progress * count));

    const panels = useMemo(() => {
        if (isTwoPanel) {
            return [
                { key: "input", title: "INPUT SPECTRUM X(f)", color: "#4ea1ff", values: inputBars, scale: "db" },
                { key: "output", title: "OUTPUT SPECTRUM Y(f)", color: "#ff5d6c", values: outputBars, scale: "db" },
            ];
        }
        return [
            {
                key: "input",
                title: "INPUT SPECTRUM X(f)",
                color: "#4ea1ff",
                values: inputBars,
                scale: "db",
            },
            {
                key: "middle",
                title: mode === "convolution" ? "IMPULSE RESPONSE H(f)" : mode === "filter" ? "FILTER RESPONSE H(f)" : "MULTIPLY BY GAIN",
                color: mode === "convolution" ? "#f7b801" : mode === "filter" ? "#f7b801" : "#b66cff",
                values: mode === "convolution" || mode === "filter" ? middleBars : multipliedBars,
                // The filter response is a gain, not a level — 0..1 linear is
                // how you read "multiply by this much".
                scale: mode === "filter" ? "gain" : "db",
            },
            {
                key: "output",
                title: "OUTPUT SPECTRUM Y(f)",
                color: "#ff5d6c",
                values: outputBars,
                scale: "db",
            },
        ];
    }, [mode, isTwoPanel, inputBars, middleBars, outputBars, multipliedBars]);

    if (!count) {
        return (
            <div className="spectrum-empty">
                Run the backend processing first to load the real FFT spectrum.
            </div>
        );
    }

    const multipliedMax = multipliedBars.reduce((m, value) => Math.max(m, Math.abs(value)), 0);
    // Reference level for the dB panels: audio magnitudes only. A filter
    // response is a 0..1 gain — a different quantity — and including it here
    // would drag the reference up by orders of magnitude and flatten the
    // audio panels, which is exactly the bug this replaced.
    const audioMax = Math.max(
        input?.displayMax || 0,
        output?.displayMax || 0,
        mode === "convolution" ? (middle?.displayMax || 0) : 0,
        multipliedMax,
        1e-12
    );
    const gainMax = Math.max(...middleBars.map((v) => Math.abs(Number(v) || 0)), 1e-9);

    return (
        <div className="frequency-animator">
            <div className="frequency-explanation">
                <div>
                    <span className="control-kicker">BACKEND FFT ANALYSIS</span>
                    <strong>
                        {mode === "convolution" && "Y(f) = X(f) · H(f)"}
                        {mode === "filter" && "Y(f) = X(f) · H(f)   (H(f) = filter's frequency response)"}
                        {mode === "amplify" && `Y(f) = ${Number(gain).toFixed(2)} · X(f)`}
                        {mode === "delay" && "Y(f) = X(f) · e^{-jωD}  →  |Y(f)| = |X(f)|"}
                        {mode === "echo" && "Y(f) = X(f) · (1 + α·e^{-jωD} + α²·e^{-j2ωD} + …)"}
                        {mode === "activity" && "Y(f): X(f) with quiet-region energy gated toward silence"}
                        {mode === "pitch" && "Y(f): pure tone synthesized at the detected dominant frequency f₀"}
                        {mode === "morph" && "Y(f) = |X(f)| · e^{ j∠Y(f) }   (phase vocoder resynthesis)"}
                    </strong>
                </div>
                <span className="frequency-status">
                    {phase === "input" && "Reading spectral bins…"}
                    {phase === "multiply" && (mode === "convolution" || mode === "filter" ? "Multiplying frequency bins…" : "Scaling every magnitude…")}
                    {phase === "output" && (isTwoPanel ? "Comparing magnitude spectra…" : "Writing output spectrum…")}
                </span>
            </div>

            {isTwoPanel && (
                <p className="frequency-note">
                    {mode === "delay"
                        ? "A pure delay only shifts every frequency's phase — it does not touch the magnitude, so X(f) and Y(f) look almost identical here."
                        : mode === "activity"
                            ? "Gating quiet frames toward silence removes their broadband contribution, so the output spectrum's overall energy drops without any one frequency being targeted."
                            : mode === "pitch"
                                ? "A real signal's energy is usually spread across a fundamental and several overtones (X(f)); the synthesized tone (Y(f)) concentrates all its energy into one sharp spike at f₀ — that spike is the frequency the detector picked."
                                : mode === "morph"
                                    ? "A pitch shift slides the whole harmonic pattern to new frequencies. A phase-only morph (robot/whisper) leaves this magnitude picture essentially untouched — which is the point: the audible change came entirely from the phase."
                                    : "Echo feeds decaying delayed copies back in, which reshapes the magnitude spectrum into a rippled “comb filter” pattern."}
                </p>
            )}

            <div className={`frequency-stage-grid ${isTwoPanel ? "two-col" : ""}`}>
                {panels.map((panel, panelIndex) => {
                    const revealCount =
                        panel.key === "input" ? (phase === "input" ? revealed : count) :
                        panel.key === "middle" ? count :
                        phase === "output" ? revealed : 0;

                    return (
                        <div className={`spectrum-stage ${panel.key}`} key={panel.key}>
                            <div className="spectrum-stage-head">
                                <span>{panel.title}</span>
                                <small>{panel.scale === "gain" ? "GAIN ×" : `dB · floor ${DB_FLOOR}`}</small>
                            </div>
                            <div className="spectrum-plot">
                                <div className="spectrum-zero" />
                                {Array.from({ length: count }, (_, i) => {
                                    const value = Number(panel.values[i] || 0);
                                    const height = panel.scale === "gain"
                                        ? clamp(value / gainMax) * 88
                                        : dbHeight(value, audioMax);
                                    const visible = i < revealCount;
                                    const active = i === cursor && visible;
                                    return (
                                        <span
                                            key={i}
                                            className={`spectrum-stem ${visible ? "is-visible" : ""} ${active ? "is-active" : ""}`}
                                            style={{
                                                left: `${((i + 0.5) / count) * 100}%`,
                                                height: `${Math.max(2, height)}%`,
                                                "--spectrum-color": panel.color,
                                            }}
                                        >
                                            <i />
                                        </span>
                                    );
                                })}
                                {/* Parked at the last bin once finished, the
                                    playhead reads as a tall bar of data. Hide
                                    it when the sweep is done. */}
                                {progress < 1 && (
                                    <span
                                        className="spectrum-cursor"
                                        style={{
                                            left: `${((cursor + 0.5) / count) * 100}%`,
                                            "--spectrum-color": panel.color,
                                        }}
                                    />
                                )}
                            </div>
                            <div className="spectrum-axis">
                                <span>0 Hz</span>
                                <span>{Math.round(input?.maxFrequency || 0).toLocaleString()} Hz</span>
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="frequency-progress">
                <div>
                    <span>
                        {mode === "convolution" && "X(f) × H(f) → Y(f)"}
                        {mode === "amplify" && "Gain applied to every frequency bin"}
                        {isTwoPanel && "Comparing the spectrum before and after"}
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
