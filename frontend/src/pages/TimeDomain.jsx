import React, { useEffect, useRef, useState } from "react";
import "../styles/time-domain.css";
import { processAudio } from "../services/api";
import ConvolutionAnimator from "../components/Visualizations/ConvolutionAnimator";
import EffectAnimator from "../components/Visualizations/EffectAnimator";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";

const MODE_CONTENT = {
    convolution: {
        titleWord: "CONVOLUTION",
        desc: "Watch the signal move through time: samples are delayed, multiplied by an impulse response, and accumulated to build the final output.",
        buttonText: "RUN CONVOLUTION",
        requiresIR: true,
        formula: "y[n] = Σ x[k] · h[n − k]",
    },
    echo: {
        titleWord: "ECHO",
        desc: "Visualize how a signal repeats over time, blending with previous delayed copies to create a decaying, continuous reflection.",
        buttonText: "PROCESS ECHO",
        requiresIR: false,
        formula: "y[n] = x[n] + α·y[n−D]   (feedback → decaying repeats)",
    },
    delay: {
        titleWord: "DELAY",
        desc: "See how a signal is buffered and played back after a precise time interval, creating a distinct, single repetition.",
        buttonText: "PROCESS DELAY",
        requiresIR: false,
        formula: "y[n] = x[n] + α·x[n−D]   (one repeat, no feedback)",
    },
};

// The user gets a bounded slider, never raw control — these ranges (and the
// backend's own clamping) keep D/decay/repeats inside sensible, audible
// territory no matter what they drag to.
const PARAM_RANGES = {
    delayMs: { min: 50, max: 800, step: 10 },
    decay: { min: 0.1, max: 0.9, step: 0.01 },
    repeats: { min: 2, max: 8, step: 1 },
};

const PARAM_DEFAULTS = {
    echo: { delayMs: 280, decay: 0.55, repeats: 5 },
    delay: { delayMs: 350, decay: 0.85, repeats: 1 },
};

function formatBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Compact, dependency-free SVG line for the "before / after" comparison
// graphs in the completion modal. The arrays it draws are already the
// backend's binned, smoothed amplitude-vs-time samples — plain linear
// amplitude, never dB, so the axis can legitimately dip below zero.
function MiniWaveGraph({ data, color, sharedMax }) {
    const width = 600;
    const height = 140;
    const midY = height / 2;
    const usable = midY - 10;

    if (!data || data.length < 2) {
        return (
            <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                <line x1="0" y1={midY} x2={width} y2={midY} className="result-axis" />
            </svg>
        );
    }

    const max = sharedMax || data.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6);
    const step = width / (data.length - 1);
    const path = data
        .map((v, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(2)} ${(midY - (v / max) * usable).toFixed(2)}`)
        .join(" ");

    return (
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            {[0.25, 0.5, 0.75].map((f) => (
                <line key={f} x1="0" y1={height * f} x2={width} y2={height * f} className="result-grid" />
            ))}
            <line x1="0" y1={midY} x2={width} y2={midY} className="result-axis" />
            <path d={path} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

export default function TimeDomain() {
    const [mode, setMode] = useState("convolution");
    const [domain, setDomain] = useState("time");

    const [inputBuffer, setInputBuffer] = useState(null);
    const [irBuffer, setIrBuffer] = useState(null);
    const [inputFile, setInputFile] = useState(null);
    const [irFile, setIrFile] = useState(null);
    const [inputName, setInputName] = useState("");
    const [irName, setIrName] = useState("");
    const [inputUrl, setInputUrl] = useState("");
    const [irUrl, setIrUrl] = useState("");

    const [isProcessing, setIsProcessing] = useState(false);
    const [warning, setWarning] = useState("");
    const [showVisualizer, setShowVisualizer] = useState(false);
    const [animationDone, setAnimationDone] = useState(false);

    const [backendResult, setBackendResult] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [modalDomain, setModalDomain] = useState("time");

    const [delayMs, setDelayMs] = useState(PARAM_DEFAULTS.echo.delayMs);
    const [decay, setDecay] = useState(PARAM_DEFAULTS.echo.decay);
    const [repeats, setRepeats] = useState(PARAM_DEFAULTS.echo.repeats);
    // Snapshot of the params actually sent to the backend for the run in
    // progress — the sliders above stay live/editable, but the animation and
    // labels should reflect what was really processed, not what the sliders
    // happen to say right now.
    const [runParams, setRunParams] = useState(PARAM_DEFAULTS.echo);

    const inputRef = useRef(null);
    const irRef = useRef(null);
    const audioCtxRef = useRef(null);
    const inputUrlRef = useRef("");
    const irUrlRef = useRef("");

    const currentContent = MODE_CONTENT[mode];

    useEffect(() => {
        return () => {
            if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
            if (irUrlRef.current) URL.revokeObjectURL(irUrlRef.current);
        };
    }, []);

    const initAudio = () => {
        if (!audioCtxRef.current) {
            audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        return audioCtxRef.current;
    };

    const handleFileUpload = async (event, type) => {
        const file = event.target.files?.[0];
        event.target.value = "";
        if (!file) return;

        try {
            const context = initAudio();
            const arrayBuffer = await file.arrayBuffer();
            const decoded = await context.decodeAudioData(arrayBuffer);
            const channelData = decoded.getChannelData(0);
            const objectUrl = URL.createObjectURL(file);

            if (type === "input") {
                if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
                inputUrlRef.current = objectUrl;
                setInputBuffer(channelData);
                setInputFile(file);
                setInputName(file.name);
                setInputUrl(objectUrl);
            } else {
                if (irUrlRef.current) URL.revokeObjectURL(irUrlRef.current);
                irUrlRef.current = objectUrl;
                setIrBuffer(channelData);
                setIrFile(file);
                setIrName(file.name);
                setIrUrl(objectUrl);
            }
        } catch (err) {
            console.error(err);
            setWarning("That file could not be decoded as audio in the browser.");
            return;
        }

        setShowVisualizer(false);
        setAnimationDone(false);
        setBackendResult(null);
        setShowModal(false);
        setWarning("");
    };

    const switchMode = (nextMode) => {
        if (isProcessing) return;
        setMode(nextMode);
        setShowVisualizer(false);
        setAnimationDone(false);
        setBackendResult(null);
        setShowModal(false);
        setDomain("time");
        setModalDomain("time");
        if (PARAM_DEFAULTS[nextMode]) {
            const d = PARAM_DEFAULTS[nextMode];
            setDelayMs(d.delayMs);
            setDecay(d.decay);
            setRepeats(d.repeats);
        }
    };

    // Convolution animates immediately with a handful of real samples (it's
    // the exact algorithm — reverse h, slide, multiply, accumulate — just on
    // 80/40 points instead of the whole file, because doing that for every
    // real sample would be far too slow to watch). The backend computes the
    // real, full-resolution audio via FFT convolution in parallel. Echo and
    // delay are cheap enough that we just animate the backend's own binned
    // result once it comes back.
    const startProcessing = async () => {
        if (!inputFile) return;
        if (currentContent.requiresIR && !irFile) return;

        setIsProcessing(true);
        setWarning("");
        setBackendResult(null);
        setShowModal(false);
        setAnimationDone(false);
        setDomain("time");
        setModalDomain("time");

        const isConvolution = mode === "convolution";
        if (isConvolution) setShowVisualizer(true);
        setRunParams({ delayMs, decay, repeats });

        try {
            const result = await processAudio(
                inputFile,
                mode,
                2.0,
                5, 5, 5,
                currentContent.requiresIR ? irFile : null,
                mode === "echo"
                    ? { delay_ms: delayMs, decay, repeats }
                    : mode === "delay"
                        ? { delay_ms: delayMs, decay }
                        : {}
            );
            setBackendResult(result);
            if (!isConvolution) setShowVisualizer(true);
        } catch (err) {
            console.error(err);
            setWarning("Failed to connect to the FastAPI backend on port 8000.");
            if (isConvolution) setShowVisualizer(false);
        } finally {
            setIsProcessing(false);
        }
    };

    // Open the results popup only once the on-screen animation has actually
    // finished AND the backend result has arrived (whichever comes second).
    useEffect(() => {
        if (animationDone && backendResult) setShowModal(true);
    }, [animationDone, backendResult]);

    const isRunDisabled = isProcessing || !inputFile || (currentContent.requiresIR && !irFile);

    const timeInput = backendResult?.visualization?.time?.input || [];
    const timeOutput = backendResult?.visualization?.time?.output || [];
    const sharedTimeMax = Math.max(
        ...timeInput.map((v) => Math.abs(v)),
        ...timeOutput.map((v) => Math.abs(v)),
        1e-6
    );

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 03 / TIME-DOMAIN PROCESSING</span>
            </div>

            <div className="module-intro">
                <h1>
                    EXPLORE THE <span>{currentContent.titleWord}</span><br />
                    OF YOUR SIGNAL.
                </h1>
                <p>{currentContent.desc}</p>

                <div className="mode-selector-row">
                    <button className={mode === "echo" ? "selected" : ""} onClick={() => switchMode("echo")}>ECHO</button>
                    <button className={mode === "convolution" ? "selected" : ""} onClick={() => switchMode("convolution")}>CONVOLUTION</button>
                    <button className={mode === "delay" ? "selected" : ""} onClick={() => switchMode("delay")}>DELAY</button>
                </div>
            </div>

            <div className="module-workspace">
                <div className="module-controls" style={{ marginBottom: currentContent.requiresIR ? "16px" : "40px" }}>
                    <div>
                        <span className="control-kicker">01 / INPUT SIGNAL</span>
                        <h2>Bring in a WAV signal</h2>
                        <p>Choose an audio file for x[n].</p>
                    </div>
                    <div className="upload-group">
                        {inputBuffer && (
                            <div className="file-chip">
                                <strong title={inputName}>{inputName}</strong>
                                <small>{inputBuffer.length.toLocaleString()} SAMPLES · {formatBytes(inputFile?.size)}</small>
                            </div>
                        )}
                        <label className={`upload-module-button ${isProcessing ? "is-disabled" : ""}`}>
                            {inputBuffer ? "CHANGE WAV" : "CHOOSE WAV"}
                            <input ref={inputRef} type="file" accept="audio/*" disabled={isProcessing} onChange={(e) => handleFileUpload(e, "input")} />
                        </label>
                    </div>
                </div>

                {currentContent.requiresIR && (
                    <div className="module-controls" style={{ marginBottom: "40px" }}>
                        <div>
                            <span className="control-kicker">02 / IMPULSE RESPONSE</span>
                            <h2>Upload an IR file</h2>
                            <p>Choose an impulse response h[n]. It will be convolved with x[n] on the backend.</p>
                        </div>
                        <div className="upload-group">
                            {irBuffer && (
                                <div className="file-chip">
                                    <strong title={irName}>{irName}</strong>
                                    <small>{irBuffer.length.toLocaleString()} SAMPLES · {formatBytes(irFile?.size)}</small>
                                </div>
                            )}
                            <label className={`upload-module-button ${isProcessing ? "is-disabled" : ""}`}>
                                {irBuffer ? "CHANGE IR" : "CHOOSE IR"}
                                <input ref={irRef} type="file" accept="audio/*" disabled={isProcessing} onChange={(e) => handleFileUpload(e, "ir")} />
                            </label>
                        </div>
                    </div>
                )}

                {(mode === "echo" || mode === "delay") && (
                    <div className={`param-control-row ${isProcessing ? "is-disabled" : ""}`}>
                        <div className="param-control">
                            <div className="param-control-head">
                                <span>DELAY TIME (D)</span>
                                <strong>{Math.round(delayMs)} ms</strong>
                            </div>
                            <input
                                type="range"
                                className="range-input"
                                min={PARAM_RANGES.delayMs.min}
                                max={PARAM_RANGES.delayMs.max}
                                step={PARAM_RANGES.delayMs.step}
                                value={delayMs}
                                disabled={isProcessing}
                                style={{ "--pos": (delayMs - PARAM_RANGES.delayMs.min) / (PARAM_RANGES.delayMs.max - PARAM_RANGES.delayMs.min) }}
                                onChange={(e) => setDelayMs(Number(e.target.value))}
                            />
                            <small>How far back in time the repeat is copied from.</small>
                        </div>

                        <div className="param-control">
                            <div className="param-control-head">
                                <span>{mode === "echo" ? "DECAY (α)" : "MIX (α)"}</span>
                                <strong>{decay.toFixed(2)}</strong>
                            </div>
                            <input
                                type="range"
                                className="range-input"
                                min={PARAM_RANGES.decay.min}
                                max={PARAM_RANGES.decay.max}
                                step={PARAM_RANGES.decay.step}
                                value={decay}
                                disabled={isProcessing}
                                style={{ "--pos": (decay - PARAM_RANGES.decay.min) / (PARAM_RANGES.decay.max - PARAM_RANGES.decay.min) }}
                                onChange={(e) => setDecay(Number(e.target.value))}
                            />
                            <small>{mode === "echo" ? "How much quieter each repeat is than the last." : "How loud the single repeat is."}</small>
                        </div>

                        {mode === "echo" && (
                            <div className="param-control">
                                <div className="param-control-head">
                                    <span>REPEATS</span>
                                    <strong>{repeats}</strong>
                                </div>
                                <input
                                    type="range"
                                    className="range-input"
                                    min={PARAM_RANGES.repeats.min}
                                    max={PARAM_RANGES.repeats.max}
                                    step={PARAM_RANGES.repeats.step}
                                    value={repeats}
                                    disabled={isProcessing}
                                    style={{ "--pos": (repeats - PARAM_RANGES.repeats.min) / (PARAM_RANGES.repeats.max - PARAM_RANGES.repeats.min) }}
                                    onChange={(e) => setRepeats(Number(e.target.value))}
                                />
                                <small>How many decaying reflections are kept before the tail is cut off.</small>
                            </div>
                        )}
                    </div>
                )}

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                <div className="math-visual" style={{ marginBottom: "24px" }}>
                    <div className="math-formula"><span>{currentContent.formula}</span></div>
                </div>

                <div className="module-action-row" style={{ marginBottom: "8px" }}>
                    <button className="process-main-button" onClick={startProcessing} disabled={isRunDisabled}>
                        {isProcessing ? "FETCHING BACKEND…" : currentContent.buttonText}
                    </button>
                </div>

                <div className="domain-switch-row">
                    <button
                        type="button"
                        className={domain === "time" ? "is-selected" : ""}
                        onClick={() => setDomain("time")}
                    >
                        TIME DOMAIN
                    </button>
                    <button
                        type="button"
                        className={domain === "frequency" ? "is-selected" : ""}
                        onClick={() => setDomain("frequency")}
                        disabled={!backendResult}
                        title={!backendResult ? "Run the effect first" : "Show the backend FFT spectrum"}
                    >
                        FREQUENCY SPECTRUM
                    </button>
                </div>

                {showVisualizer && (
                    <div className="signal-sections">
                        <div className="processing-panel is-processing" style={{ gridColumn: "1 / -1", width: "100%" }}>
                            <div className="panel-heading">
                                <div>
                                    <span className="control-kicker">ANIMATION ENGINE ({domain.toUpperCase()})</span>
                                    <h3>{domain === "time" ? "Sliding Window Analysis" : "Frequency Spectrum Multiplication"}</h3>
                                </div>
                                {backendResult && (
                                    <button className="secondary-button" onClick={() => setShowModal(true)}>
                                        OPEN RESULTS ↗
                                    </button>
                                )}
                            </div>

                            {domain === "time" ? (
                                mode === "convolution" ? (
                                    <ConvolutionAnimator
                                        inputBuffer={inputBuffer}
                                        irBuffer={irBuffer}
                                        onComplete={() => setAnimationDone(true)}
                                    />
                                ) : (
                                    backendResult ? (
                                        <EffectAnimator
                                            visualization={backendResult.visualization}
                                            mode={mode}
                                            delayMs={runParams.delayMs}
                                            decay={runParams.decay}
                                            repeats={runParams.repeats}
                                            onComplete={() => setAnimationDone(true)}
                                        />
                                    ) : (
                                        <div className="spectrum-empty">Waiting for the backend…</div>
                                    )
                                )
                            ) : (
                                backendResult ? (
                                    <FrequencySpectrumAnimator
                                        visualization={backendResult.visualization}
                                        mode={mode}
                                        loop
                                    />
                                ) : (
                                    <div className="spectrum-empty">
                                        Run the effect once — the frequency spectrum is calculated by the Python backend.
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                )}

                {!showVisualizer && !isProcessing && (
                    <div className="empty-module-state">
                        Upload {currentContent.requiresIR ? "a signal and an impulse response" : "a signal"}, then press
                        &nbsp;<strong style={{ color: "#fff" }}>{currentContent.buttonText}</strong>.
                    </div>
                )}

                {showModal && backendResult && (
                    <div className="module-modal-backdrop" onClick={() => setShowModal(false)}>
                        <div className="completion-modal" onClick={(e) => e.stopPropagation()}>
                            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
                            <span className="modal-kicker">BACKEND DSP COMPLETE</span>
                            <h2>Input vs. Output</h2>
                            <p>
                                The animation used a reduced set of points to stay fast; the audio below is the real,
                                full-resolution result computed on the backend.
                            </p>

                            <div className="modal-domain-toggle">
                                <button className={modalDomain === "time" ? "is-selected" : ""} onClick={() => setModalDomain("time")}>
                                    TIME DOMAIN
                                </button>
                                <button className={modalDomain === "frequency" ? "is-selected" : ""} onClick={() => setModalDomain("frequency")}>
                                    FREQUENCY SPECTRUM
                                </button>
                            </div>

                            <div className="modal-audio-buttons">
                                <span className="file-chip" style={{ borderColor: "rgba(78,161,255,.4)" }}>
                                    <strong style={{ color: "#4ea1ff" }}>INPUT x[n]</strong>
                                    <audio controls src={inputUrl} style={{ width: "220px", marginTop: "6px" }} />
                                </span>
                                {currentContent.requiresIR && (
                                    <span className="file-chip" style={{ borderColor: "rgba(247,184,1,.4)" }}>
                                        <strong style={{ color: "#f7b801" }}>IMPULSE RESPONSE h[n]</strong>
                                        <audio controls src={irUrl} style={{ width: "220px", marginTop: "6px" }} />
                                    </span>
                                )}
                                <span className="file-chip" style={{ borderColor: "rgba(255,93,108,.4)" }}>
                                    <strong style={{ color: "#ff5d6c" }}>OUTPUT y[n]</strong>
                                    <audio controls src={backendResult.audio_url} style={{ width: "220px", marginTop: "6px" }} />
                                </span>
                            </div>

                            {modalDomain === "time" ? (
                                <div className="result-graphs">
                                    <div className="result-graph-card">
                                        <div className="result-graph-head">
                                            <span>x[n] · INPUT SIGNAL</span>
                                            <small>{backendResult.input_duration_seconds}s · amplitude vs time</small>
                                        </div>
                                        <div className="result-graph-body">
                                            <MiniWaveGraph data={timeInput} color="#4ea1ff" sharedMax={sharedTimeMax} />
                                        </div>
                                    </div>
                                    <div className="result-graph-card">
                                        <div className="result-graph-head">
                                            <span>y[n] · OUTPUT SIGNAL</span>
                                            <small>{backendResult.processed_duration_seconds}s · amplitude vs time</small>
                                        </div>
                                        <div className="result-graph-body">
                                            <MiniWaveGraph data={timeOutput} color="#ff5d6c" sharedMax={sharedTimeMax} />
                                        </div>
                                    </div>
                                    <p className="result-graph-hint">
                                        Both curves share one amplitude scale and are binned + smoothed for display — the underlying
                                        audio keeps its full sample rate.
                                    </p>
                                </div>
                            ) : (
                                <FrequencySpectrumAnimator visualization={backendResult.visualization} mode={mode} loop />
                            )}

                            <div className="result-modal-actions">
                                <button className="process-main-button" onClick={() => setShowModal(false)}>
                                    CLOSE &amp; REPLAY ANIMATION
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}