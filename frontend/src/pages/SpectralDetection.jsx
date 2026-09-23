import { useEffect, useRef, useState } from "react";
import "../styles/time-domain.css";
import { processAudio } from "../services/api";
import PitchAnimator from "../components/Visualizations/PitchAnimator";
import EffectAnimator from "../components/Visualizations/EffectAnimator";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";

const RANGE_LIMIT = { min: 20, max: 8000, step: 10 };

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

export default function SpectralDetection() {
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

    const [fmin, setFmin] = useState(50);
    const [fmax, setFmax] = useState(2000);

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

    const nyquist = Math.max(RANGE_LIMIT.min + 100, Math.floor(sampleRate / 2) - 50);
    const rangeMax = Math.min(RANGE_LIMIT.max, nyquist);

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

            const maxHz = Math.min(RANGE_LIMIT.max, Math.max(RANGE_LIMIT.min + 100, Math.floor(decoded.sampleRate / 2) - 50));
            setFmin((v) => Math.min(v, maxHz - 100));
            setFmax((v) => Math.min(v, maxHz));
        } catch (err) {
            console.error(err);
            setWarning("That file could not be decoded as audio in the browser.");
            return;
        }

        setWarning("");
        resetVisualizerState();
    };

    // Peak-picking on a full FFT isn't something you can meaningfully
    // hand-animate sample-by-sample in the browser. So, same approach as the
    // other modules: the animation plays back the backend's own binned
    // spectrum + detected peak once the real result comes back from FastAPI.
    const startProcessing = async () => {
        if (!inputFile) return;
        if (fmin >= fmax) return;

        setIsProcessing(true);
        setWarning("");
        setBackendResult(null);
        setShowModal(false);
        setAnimationDone(false);
        setDomain("process");
        setModalDomain("time");
        setShowVisualizer(true);

        try {
            const result = await processAudio(
                inputFile,
                "pitch",
                2.0,
                5, 5, 5,
                null,
                { pitch_fmin: fmin, pitch_fmax: fmax }
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

    const isRunDisabled = isProcessing || !inputFile || fmin >= fmax;

    const timeInput = backendResult?.visualization?.time?.input || [];
    const timeOutput = backendResult?.visualization?.time?.output || [];
    const sharedTimeMax = Math.max(
        ...timeInput.map((v) => Math.abs(v)),
        ...timeOutput.map((v) => Math.abs(v)),
        1e-6
    );

    const pitch = backendResult?.pitch;
    const note = pitch?.note;
    const noteLabel = note?.note ? `${note.note}${note.octave ?? ""}` : null;
    const paramLabel = pitch
        ? `f₀ = ${pitch.peakFrequency.toFixed(1)} Hz${noteLabel ? `   ≈ ${noteLabel}` : ""}`
        : "";

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 05 / SPECTRAL DETECTION</span>
            </div>

            <div className="module-intro">
                <h1>
                    FIND THE <span>DOMINANT PITCH</span><br />
                    OF YOUR SIGNAL.
                </h1>
                <p>
                    Transform the signal to the frequency domain, find its strongest spectral peak, and map that
                    frequency to the nearest musical note — then hear a pure tone synthesized at exactly that
                    frequency, to check the detector's answer by ear.
                </p>
            </div>

            <div className="module-workspace">
                <div className="module-controls" style={{ marginBottom: "40px" }}>
                    <div>
                        <span className="control-kicker">01 / INPUT SIGNAL</span>
                        <h2>Bring in a WAV signal</h2>
                        <p>Choose an audio file for x[n].</p>
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

                <div className={`param-control-row ${isProcessing ? "is-disabled" : ""}`}>
                    <div className="param-control">
                        <div className="param-control-head">
                            <span>SEARCH RANGE MIN</span>
                            <strong>{Math.round(fmin)} Hz</strong>
                        </div>
                        <input
                            type="range"
                            className="range-input"
                            min={RANGE_LIMIT.min}
                            max={rangeMax}
                            step={RANGE_LIMIT.step}
                            value={fmin}
                            disabled={isProcessing}
                            style={{ "--pos": (fmin - RANGE_LIMIT.min) / (rangeMax - RANGE_LIMIT.min) }}
                            onChange={(e) => setFmin(Number(e.target.value))}
                        />
                        <small>Peaks below this frequency are ignored (filters out DC/hum/rumble).</small>
                    </div>
                    <div className="param-control">
                        <div className="param-control-head">
                            <span>SEARCH RANGE MAX</span>
                            <strong>{Math.round(fmax)} Hz</strong>
                        </div>
                        <input
                            type="range"
                            className="range-input"
                            min={RANGE_LIMIT.min}
                            max={rangeMax}
                            step={RANGE_LIMIT.step}
                            value={fmax}
                            disabled={isProcessing}
                            style={{ "--pos": (fmax - RANGE_LIMIT.min) / (rangeMax - RANGE_LIMIT.min) }}
                            onChange={(e) => setFmax(Number(e.target.value))}
                        />
                        <small>Peaks above this frequency are ignored.</small>
                    </div>
                </div>

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}
                {fmin >= fmax && (
                    <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> The search range minimum must be below the maximum.</div>
                )}

                <div className="math-visual" style={{ marginBottom: "24px" }}>
                    <div className="math-formula"><span>f₀ = argmax |X(f)|   for {fmin} Hz ≤ f ≤ {fmax} Hz</span></div>
                </div>

                <div className="module-action-row" style={{ marginBottom: "8px" }}>
                    <button className="process-main-button" onClick={startProcessing} disabled={isRunDisabled}>
                        {isProcessing ? "FETCHING BACKEND…" : "RUN PITCH DETECTION"}
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
                        title={!backendResult ? "Run the detection first" : undefined}
                    >
                        TIME DOMAIN
                    </button>
                    <button
                        type="button"
                        className={domain === "frequency" ? "is-selected" : ""}
                        onClick={() => setDomain("frequency")}
                        disabled={!backendResult}
                        title={!backendResult ? "Run the detection first" : "Show the backend FFT spectrum"}
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
                                        {domain === "process" && "Spectrum → Find Peak → Estimate Pitch"}
                                        {domain === "time" && "Original vs. Synthesized Tone"}
                                        {domain === "frequency" && "Frequency Spectrum Comparison"}
                                    </h3>
                                    {noteLabel && (
                                        <small style={{ color: "rgba(255,255,255,.5)" }}>detected: {pitch.peakFrequency.toFixed(1)} Hz ≈ {noteLabel}</small>
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
                                    <PitchAnimator
                                        key={`process-${processRunId}`}
                                        pitch={backendResult.pitch}
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
                                        mode="pitch"
                                        loop={false}
                                        formula="y[n] = A·sin(2π·f₀·n/sr)   (f₀ = detected dominant frequency)"
                                        paramLabel={paramLabel}
                                        operationLabel="PITCH DETECTION"
                                        outputCaption="pure tone synthesized at f₀"
                                        progressLabel="Synthesizing a pure tone at the detected frequency…"
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
                                        mode="pitch"
                                        loop={false}
                                    />
                                ) : (
                                    <div className="spectrum-empty">
                                        Run the detection once — the frequency spectrum is calculated by the Python backend.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {!showVisualizer && !isProcessing && (
                    <div className="empty-module-state">
                        Upload a signal, then press&nbsp;<strong style={{ color: "#fff" }}>RUN PITCH DETECTION</strong>.
                    </div>
                )}

                {showModal && backendResult && (
                    <div className="module-modal-backdrop" onClick={() => setShowModal(false)}>
                        <div className="completion-modal" onClick={(e) => e.stopPropagation()}>
                            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
                            <span className="modal-kicker">BACKEND DSP COMPLETE</span>
                            <h2>
                                {noteLabel ? (
                                    <>Detected: {pitch.peakFrequency.toFixed(1)} Hz ≈ <span style={{ color: "#f7b801" }}>{noteLabel}</span></>
                                ) : "Input vs. Output"}
                            </h2>
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
                                <span className="file-chip" style={{ borderColor: "rgba(255,93,108,.4)" }}>
                                    <strong style={{ color: "#ff5d6c" }}>SYNTHESIZED TONE y[n]</strong>
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
                                            <span>y[n] · SYNTHESIZED TONE</span>
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
                                    mode="pitch"
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
