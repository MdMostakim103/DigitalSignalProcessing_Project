import { useEffect, useRef, useState } from "react";
import "../styles/time-domain.css";
import { processAudio } from "../services/api";
import MorphAnimator from "../components/Visualizations/MorphAnimator";
import EffectAnimator from "../components/Visualizations/EffectAnimator";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";

// Two morphs that change pitch/time while preserving the other, and two
// that change *only* the phase — which is how this module shows that
// magnitude and phase carry independent information.
const MORPH_CONTENT = {
    pitch: {
        label: "PITCH SHIFT",
        titleWord: "PITCH",
        desc: "Move the pitch up or down while the duration stays exactly the same — the phase vocoder stretches time, then resampling undoes the length change.",
        buttonText: "RUN PITCH SHIFT",
        formula: "bins scaled in frequency · duration preserved",
        usesSteps: true,
        usesRate: false,
        outputCaption: "pitch-shifted signal",
    },
    stretch: {
        label: "TIME STRETCH",
        titleWord: "TIMING",
        desc: "Change how long the signal lasts while every frequency stays exactly where it was — the same phase vocoder, without the resampling step.",
        buttonText: "RUN TIME STRETCH",
        formula: "frames respaced in time · frequencies preserved",
        usesSteps: false,
        usesRate: true,
        outputCaption: "time-stretched signal",
    },
    robot: {
        label: "ROBOT (ZERO PHASE)",
        titleWord: "PHASE",
        desc: "Keep every magnitude exactly as it was and set all phase to zero. Nothing about |X(f)| changes, yet the voice becomes a flat monotone buzz — that difference is what the phase was carrying.",
        buttonText: "RUN ROBOT MORPH",
        formula: "∠Y(f) = 0   ·   |Y(f)| = |X(f)|",
        usesSteps: false,
        usesRate: false,
        outputCaption: "zero-phase (robotized) signal",
    },
    whisper: {
        label: "WHISPER (RANDOM PHASE)",
        titleWord: "PHASE",
        desc: "Keep every magnitude again, but randomize the phase instead of zeroing it. Same |X(f)|, yet the result turns breathy and unvoiced.",
        buttonText: "RUN WHISPER MORPH",
        formula: "∠Y(f) = random   ·   |Y(f)| = |X(f)|",
        usesSteps: false,
        usesRate: false,
        outputCaption: "randomized-phase (whispered) signal",
    },
};

const MORPH_ORDER = ["pitch", "stretch", "robot", "whisper"];

const STEPS_RANGE = { min: -12, max: 12, step: 1 };
const RATE_RANGE = { min: 0.5, max: 2, step: 0.05 };

function formatBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Same compact, dependency-free SVG line used by the other modules' result
// modal — plain linear amplitude vs time, auto-scaled to the shared peak so
// the curve never reads as "mostly negative" just because it is bipolar.
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

export default function VoiceMorphing() {
    const [morphMode, setMorphMode] = useState("pitch");
    const [domain, setDomain] = useState("process");

    const [inputBuffer, setInputBuffer] = useState(null);
    const [inputFile, setInputFile] = useState(null);
    const [inputName, setInputName] = useState("");
    const [inputUrl, setInputUrl] = useState("");
    const [sampleRate, setSampleRate] = useState(44100);

    const [isProcessing, setIsProcessing] = useState(false);
    const [warning, setWarning] = useState("");
    const [showVisualizer, setShowVisualizer] = useState(false);
    const [animationDone, setAnimationDone] = useState(false);

    const [backendResult, setBackendResult] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [modalDomain, setModalDomain] = useState("time");

    const [nSteps, setNSteps] = useState(4);
    const [rate, setRate] = useState(1.5);
    const [runParams, setRunParams] = useState({ mode: "pitch", nSteps: 4, rate: 1.5 });

    // Each animation plays once on its first mount, then freezes fully
    // revealed. Bumping these forces a remount (a fresh play-through) only
    // when the user explicitly presses a replay button.
    const [processRunId, setProcessRunId] = useState(0);
    const [timeRunId, setTimeRunId] = useState(0);
    const [freqRunId, setFreqRunId] = useState(0);
    const [modalFreqRunId, setModalFreqRunId] = useState(0);

    const inputRef = useRef(null);
    const audioCtxRef = useRef(null);
    const inputUrlRef = useRef("");

    const content = MORPH_CONTENT[morphMode];

    useEffect(() => {
        return () => {
            if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
        };
    }, []);

    const initAudio = () => {
        if (!audioCtxRef.current) {
            audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        return audioCtxRef.current;
    };

    const resetVisualizerState = () => {
        setShowVisualizer(false);
        setAnimationDone(false);
        setBackendResult(null);
        setShowModal(false);
        setDomain("process");
        setModalDomain("time");
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = "";
        if (!file) return;

        try {
            const context = initAudio();
            const arrayBuffer = await file.arrayBuffer();
            const decoded = await context.decodeAudioData(arrayBuffer);
            const channelData = decoded.getChannelData(0);
            const objectUrl = URL.createObjectURL(file);

            if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
            inputUrlRef.current = objectUrl;
            setInputBuffer(channelData);
            setInputFile(file);
            setInputName(file.name);
            setInputUrl(objectUrl);
            setSampleRate(decoded.sampleRate);
        } catch (err) {
            console.error(err);
            setWarning("That file could not be decoded as audio in the browser.");
            return;
        }

        setWarning("");
        resetVisualizerState();
    };

    const switchMorph = (nextMode) => {
        if (isProcessing) return;
        setMorphMode(nextMode);
        resetVisualizerState();
    };

    // A phase vocoder runs over every STFT frame of the whole file — not
    // something to hand-roll in the browser. Same approach as the other
    // modules: the animation plays back the backend's own representative
    // frame (magnitude + phase, before and after) once the real,
    // full-resolution audio comes back from FastAPI.
    const startProcessing = async () => {
        if (!inputFile) return;

        setIsProcessing(true);
        setWarning("");
        setBackendResult(null);
        setShowModal(false);
        setAnimationDone(false);
        setDomain("process");
        setModalDomain("time");
        setShowVisualizer(true);
        setRunParams({ mode: morphMode, nSteps, rate });

        try {
            const result = await processAudio(
                inputFile,
                "morph",
                2.0,
                5, 5, 5,
                null,
                {
                    morph_mode: morphMode,
                    n_steps: content.usesSteps ? nSteps : undefined,
                    rate: content.usesRate ? rate : undefined,
                }
            );
            setBackendResult(result);
        } catch (err) {
            console.error(err);
            setWarning("Failed to connect to the FastAPI backend on port 8000.");
            setShowVisualizer(false);
        } finally {
            setIsProcessing(false);
        }
    };

    useEffect(() => {
        if (animationDone && backendResult) setShowModal(true);
    }, [animationDone, backendResult]);

    const isRunDisabled = isProcessing || !inputFile;

    const timeInput = backendResult?.visualization?.time?.input || [];
    const timeOutput = backendResult?.visualization?.time?.output || [];
    const sharedTimeMax = Math.max(
        ...timeInput.map((v) => Math.abs(v)),
        ...timeOutput.map((v) => Math.abs(v)),
        1e-6
    );

    const runContent = MORPH_CONTENT[runParams.mode] || content;
    const paramLabel = runParams.mode === "pitch"
        ? `${runParams.nSteps > 0 ? "+" : ""}${runParams.nSteps} semitones · duration preserved`
        : runParams.mode === "stretch"
            ? `rate = ${runParams.rate.toFixed(2)}× · pitch preserved`
            : runParams.mode === "robot"
                ? "phase set to 0 · magnitude untouched"
                : "phase randomized · magnitude untouched";

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 06 / VOICE MORPHING</span>
            </div>

            <div className="module-intro">
                <h1>
                    MANIPULATE THE <span>{content.titleWord}</span><br />
                    OF YOUR SIGNAL.
                </h1>
                <p>{content.desc}</p>

                <div className="mode-selector-row">
                    {MORPH_ORDER.map((key) => (
                        <button
                            key={key}
                            className={morphMode === key ? "selected" : ""}
                            onClick={() => switchMorph(key)}
                        >
                            {MORPH_CONTENT[key].label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="module-workspace">
                <div className="module-controls" style={{ marginBottom: "40px" }}>
                    <div>
                        <span className="control-kicker">01 / INPUT SIGNAL</span>
                        <h2>Bring in a WAV signal</h2>
                        <p>Choose an audio file for x[n] — a voice recording shows this off best.</p>
                    </div>
                    <div className="upload-group">
                        {inputBuffer && (
                            <div className="file-chip">
                                <strong title={inputName}>{inputName}</strong>
                                <small>{inputBuffer.length.toLocaleString()} SAMPLES · {formatBytes(inputFile?.size)} · {sampleRate.toLocaleString()} Hz</small>
                            </div>
                        )}
                        <label className={`upload-module-button ${isProcessing ? "is-disabled" : ""}`}>
                            {inputBuffer ? "CHANGE WAV" : "CHOOSE WAV"}
                            <input ref={inputRef} type="file" accept="audio/*" disabled={isProcessing} onChange={handleFileUpload} />
                        </label>
                    </div>
                </div>

                {(content.usesSteps || content.usesRate) && (
                    <div className={`param-control-row ${isProcessing ? "is-disabled" : ""}`}>
                        {content.usesSteps && (
                            <div className="param-control">
                                <div className="param-control-head">
                                    <span>PITCH SHIFT</span>
                                    <strong>{nSteps > 0 ? "+" : ""}{nSteps} semitones</strong>
                                </div>
                                <input
                                    type="range"
                                    className="range-input"
                                    min={STEPS_RANGE.min}
                                    max={STEPS_RANGE.max}
                                    step={STEPS_RANGE.step}
                                    value={nSteps}
                                    disabled={isProcessing}
                                    style={{ "--pos": (nSteps - STEPS_RANGE.min) / (STEPS_RANGE.max - STEPS_RANGE.min) }}
                                    onChange={(e) => setNSteps(Number(e.target.value))}
                                />
                                <small>12 semitones = one octave. Positive shifts up, negative shifts down.</small>
                            </div>
                        )}
                        {content.usesRate && (
                            <div className="param-control">
                                <div className="param-control-head">
                                    <span>STRETCH RATE</span>
                                    <strong>{rate.toFixed(2)}×</strong>
                                </div>
                                <input
                                    type="range"
                                    className="range-input"
                                    min={RATE_RANGE.min}
                                    max={RATE_RANGE.max}
                                    step={RATE_RANGE.step}
                                    value={rate}
                                    disabled={isProcessing}
                                    style={{ "--pos": (rate - RATE_RANGE.min) / (RATE_RANGE.max - RATE_RANGE.min) }}
                                    onChange={(e) => setRate(Number(e.target.value))}
                                />
                                <small>Above 1× plays faster and shorter; below 1× plays slower and longer. The pitch does not move either way.</small>
                            </div>
                        )}
                    </div>
                )}

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                <div className="math-visual" style={{ marginBottom: "24px" }}>
                    <div className="math-formula"><span>{content.formula}</span></div>
                </div>

                <div className="module-action-row" style={{ marginBottom: "8px" }}>
                    <button className="process-main-button" onClick={startProcessing} disabled={isRunDisabled}>
                        {isProcessing ? "FETCHING BACKEND…" : content.buttonText}
                    </button>
                </div>

                <div className="domain-switch-row">
                    <button
                        type="button"
                        className={domain === "process" ? "is-selected" : ""}
                        onClick={() => setDomain("process")}
                    >
                        PROCESS
                    </button>
                    <button
                        type="button"
                        className={domain === "time" ? "is-selected" : ""}
                        onClick={() => setDomain("time")}
                        disabled={!backendResult}
                        title={!backendResult ? "Run the morph first" : undefined}
                    >
                        TIME DOMAIN
                    </button>
                    <button
                        type="button"
                        className={domain === "frequency" ? "is-selected" : ""}
                        onClick={() => setDomain("frequency")}
                        disabled={!backendResult}
                        title={!backendResult ? "Run the morph first" : "Show the backend FFT spectrum"}
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
                                    <h3>
                                        {domain === "process" && "Frame → Spectrum + Phase → Morph → Reconstruct"}
                                        {domain === "time" && "Original vs. Morphed Signal"}
                                        {domain === "frequency" && "Frequency Spectrum Comparison"}
                                    </h3>
                                    {backendResult && (
                                        <small style={{ color: "rgba(255,255,255,.5)" }}>{paramLabel}</small>
                                    )}
                                </div>
                                {backendResult && (
                                    <div style={{ display: "flex", gap: "10px" }}>
                                        <button
                                            className="secondary-button"
                                            onClick={() => {
                                                if (domain === "process") setProcessRunId((id) => id + 1);
                                                else if (domain === "time") setTimeRunId((id) => id + 1);
                                                else setFreqRunId((id) => id + 1);
                                            }}
                                        >
                                            ⟳ REPLAY ANIMATION
                                        </button>
                                        <button className="secondary-button" onClick={() => setShowModal(true)}>
                                            OPEN RESULTS ↗
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div style={{ display: domain === "process" ? "block" : "none" }}>
                                {backendResult ? (
                                    <MorphAnimator
                                        key={`process-${processRunId}`}
                                        morph={backendResult.morph}
                                        loop={false}
                                        onComplete={() => setAnimationDone(true)}
                                    />
                                ) : (
                                    <div className="spectrum-empty">Waiting for the backend…</div>
                                )}
                            </div>

                            <div style={{ display: domain === "time" ? "block" : "none" }}>
                                {backendResult ? (
                                    <EffectAnimator
                                        key={`time-${timeRunId}`}
                                        visualization={backendResult.visualization}
                                        mode="morph"
                                        loop={false}
                                        formula={runContent.formula}
                                        paramLabel={paramLabel}
                                        operationLabel={runContent.label}
                                        outputCaption={runContent.outputCaption}
                                        progressLabel="Reconstructing the morphed signal…"
                                    />
                                ) : (
                                    <div className="spectrum-empty">Waiting for the backend…</div>
                                )}
                            </div>

                            <div style={{ display: domain === "frequency" ? "block" : "none" }}>
                                {backendResult ? (
                                    <FrequencySpectrumAnimator
                                        key={`freq-${freqRunId}`}
                                        visualization={backendResult.visualization}
                                        mode="morph"
                                        loop={false}
                                    />
                                ) : (
                                    <div className="spectrum-empty">
                                        Run the morph once — the frequency spectrum is calculated by the Python backend.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {!showVisualizer && !isProcessing && (
                    <div className="empty-module-state">
                        Upload a signal, then press&nbsp;<strong style={{ color: "#fff" }}>{content.buttonText}</strong>.
                    </div>
                )}

                {showModal && backendResult && (
                    <div className="module-modal-backdrop" onClick={() => setShowModal(false)}>
                        <div className="completion-modal" onClick={(e) => e.stopPropagation()}>
                            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
                            <span className="modal-kicker">BACKEND DSP COMPLETE</span>
                            <h2>Input vs. Output</h2>
                            <p>
                                The animation used one representative STFT frame to stay readable; the audio below is the
                                real, full-resolution result computed on the backend — play both to hear what changed.
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
                                <span className="file-chip" style={{ borderColor: "rgba(255,93,108,.4)" }}>
                                    <strong style={{ color: "#ff5d6c" }}>MORPHED y[n]</strong>
                                    <audio controls src={backendResult.audio_url} style={{ width: "220px", marginTop: "6px" }} />
                                </span>
                            </div>

                            <div style={{ display: modalDomain === "time" ? "block" : "none" }}>
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
                                            <span>y[n] · MORPHED SIGNAL</span>
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
                            </div>

                            <div style={{ display: modalDomain === "frequency" ? "block" : "none" }}>
                                <div className="module-action-row" style={{ marginBottom: "10px" }}>
                                    <button className="secondary-button" onClick={() => setModalFreqRunId((id) => id + 1)}>
                                        ⟳ REPLAY ANIMATION
                                    </button>
                                </div>
                                <FrequencySpectrumAnimator
                                    key={`modal-freq-${modalFreqRunId}`}
                                    visualization={backendResult.visualization}
                                    mode="morph"
                                    loop={false}
                                />
                            </div>

                            <div className="result-modal-actions">
                                <button className="process-main-button" onClick={() => setShowModal(false)}>
                                    CLOSE
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}
