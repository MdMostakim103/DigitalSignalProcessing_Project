import { useEffect, useRef, useState } from "react";
import "../styles/time-domain.css";
import { processAudio } from "../services/api";
import ActivityAnimator from "../components/Visualizations/ActivityAnimator";
import EffectAnimator from "../components/Visualizations/EffectAnimator";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";

const THRESHOLD_RANGE = { min: 2, max: 60, step: 1 }; // percent of peak frame energy
const ZCR_HZ_RANGE = { min: 200, max: 4000, step: 50 }; // approximate crossing rate, in Hz

// zcr_ratio (crossings per sample) is not sample-rate-invariant: the same
// physical audio measured at 8kHz reads ~5x higher than at 48kHz, because
// the same real crossings get divided by a smaller per-frame sample count.
// The invariant quantity is an approximate frequency: for a sinusoid at f Hz,
// crossings-per-sample ~= 2f/sr, so f ~= zcr_ratio * sr / 2. Exposing the
// slider in Hz (like the filter module's cutoff) means the same setting
// means the same thing regardless of the uploaded file's sample rate.
function zcrHzToRatio(hz, sampleRate) {
    return (hz * 2) / Math.max(1, sampleRate);
}

const METHOD_CONTENT = {
    energy: {
        label: "SIMPLE (ENERGY)",
        formula: (thresholdPct) =>
            `enter: STE[n] > noise_floor + ${(thresholdPct / 100).toFixed(2)}·headroom   ·   stay active until STE[n] < half that`,
        desc: "Break the signal into short frames and measure short-time energy (STE) per frame. The threshold isn't a flat fraction of the loudest frame — that's fragile, since one loud transient anywhere in the file would rescale it. Instead it's set relative to an estimated background noise floor, and hysteresis (a lower exit threshold than entry threshold) keeps a frame active through brief dips instead of flickering on and off.",
    },
    energy_zcr: {
        label: "SOPHISTICATED (ENERGY + ZCR)",
        formula: (thresholdPct, zcrHz) =>
            `enter: STE[n] > noise_floor + ${(thresholdPct / 100).toFixed(2)}·headroom  AND  ZCR[n] < ${zcrHz} Hz   ·   stay active on energy alone`,
        desc: "Energy alone can't tell loud speech from loud noise, so zero-crossing rate (ZCR) adds a second test: voiced speech crosses zero slowly, broadband/tonal noise crosses zero far more often at the same loudness. But testing BOTH on every single frame would reject real unvoiced speech too (s, f, sh are quiet and high-ZCR) — so ZCR only gates entering the active state; once triggered by a clear voiced onset, staying active only depends on energy, so a trailing unvoiced consonant isn't cut off mid-word.",
    },
};

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

export default function SpeechActivity() {
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

    const [method, setMethod] = useState("energy");
    const [thresholdPct, setThresholdPct] = useState(15);
    const [zcrHz, setZcrHz] = useState(750);
    const [runThresholdPct, setRunThresholdPct] = useState(15);
    const [runZcrHz, setRunZcrHz] = useState(750);
    const [runMethod, setRunMethod] = useState("energy");

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

    const switchMethod = (nextMethod) => {
        if (isProcessing) return;
        setMethod(nextMethod);
        resetVisualizerState();
    };

    // The framing + energy (+ ZCR) computation is simple enough to actually
    // run live in the browser on a small number of samples — same approach
    // ConvolutionAnimator uses for the real convolution sum. So the PROCESS
    // tab computes its own reduced-scale version straight from inputBuffer;
    // the backend separately computes the same thing at full sample-rate
    // resolution for the real gated audio and the time/frequency tabs.
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
        setRunThresholdPct(thresholdPct);
        setRunZcrHz(zcrHz);
        setRunMethod(method);
        // EffectAnimator/FrequencySpectrumAnimator remount naturally because
        // backendResult flips null -> object on every run. ActivityAnimator
        // no longer depends on backendResult at all (it computes live from
        // inputBuffer, which never goes null between runs), so it needs an
        // explicit key bump to restart the animation on a fresh run.
        setProcessRunId((id) => id + 1);

        try {
            const result = await processAudio(
                inputFile,
                "activity",
                2.0,
                5, 5, 5,
                null,
                {
                    threshold_ratio: thresholdPct / 100,
                    activity_method: method,
                    // Sent as Hz, not a pre-computed ratio: the browser
                    // resamples decoded audio to its own AudioContext rate,
                    // which does not reliably match the file's real native
                    // sample rate the backend actually loads and processes
                    // at. The backend converts Hz -> ratio itself, against
                    // the sr it truly used.
                    zcr_threshold_hz: zcrHz,
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

    const activeFrames = backendResult?.activity?.active || [];
    const activePct = activeFrames.length
        ? Math.round((activeFrames.filter(Boolean).length / activeFrames.length) * 100)
        : null;

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 04 / SPEECH &amp; ACTIVITY</span>
            </div>

            <div className="module-intro">
                <h1>
                    FIND THE <span>ACTIVE</span><br />
                    REGIONS OF YOUR SIGNAL.
                </h1>
                <p>{METHOD_CONTENT[method].desc}</p>

                <div className="mode-selector-row">
                    {Object.keys(METHOD_CONTENT).map((key) => (
                        <button
                            key={key}
                            className={method === key ? "selected" : ""}
                            onClick={() => switchMethod(key)}
                        >
                            {METHOD_CONTENT[key].label}
                        </button>
                    ))}
                </div>
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
                            <span>ACTIVITY THRESHOLD</span>
                            <strong>{thresholdPct}% above noise floor</strong>
                        </div>
                        <input
                            type="range"
                            className="range-input"
                            min={THRESHOLD_RANGE.min}
                            max={THRESHOLD_RANGE.max}
                            step={THRESHOLD_RANGE.step}
                            value={thresholdPct}
                            disabled={isProcessing}
                            style={{ "--pos": (thresholdPct - THRESHOLD_RANGE.min) / (THRESHOLD_RANGE.max - THRESHOLD_RANGE.min) }}
                            onChange={(e) => setThresholdPct(Number(e.target.value))}
                        />
                        <small>How far above the estimated background noise level a frame must rise to trigger activity — not a fraction of the loudest frame, which a single transient could distort.</small>
                    </div>

                    {method === "energy_zcr" && (
                        <div className="param-control">
                            <div className="param-control-head">
                                <span>ZCR THRESHOLD</span>
                                <strong>~{zcrHz} Hz</strong>
                            </div>
                            <input
                                type="range"
                                className="range-input"
                                min={ZCR_HZ_RANGE.min}
                                max={ZCR_HZ_RANGE.max}
                                step={ZCR_HZ_RANGE.step}
                                value={zcrHz}
                                disabled={isProcessing}
                                style={{ "--pos": (zcrHz - ZCR_HZ_RANGE.min) / (ZCR_HZ_RANGE.max - ZCR_HZ_RANGE.min) }}
                                onChange={(e) => setZcrHz(Number(e.target.value))}
                            />
                            <small>Frames whose average crossing rate is ABOVE this (noise-like) are rejected even if loud enough. Expressed in Hz, not a raw ratio, so it means the same thing on any sample rate.</small>
                        </div>
                    )}
                </div>

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                <div className="math-visual" style={{ marginBottom: "24px" }}>
                    <div className="math-formula"><span>{METHOD_CONTENT[method].formula(thresholdPct, zcrHz)}</span></div>
                </div>

                <div className="module-action-row" style={{ marginBottom: "8px" }}>
                    <button className="process-main-button" onClick={startProcessing} disabled={isRunDisabled}>
                        {isProcessing ? "FETCHING BACKEND…" : "RUN ACTIVITY DETECTION"}
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
                        title={!backendResult ? "Run the analysis first" : undefined}
                    >
                        TIME DOMAIN
                    </button>
                    <button
                        type="button"
                        className={domain === "frequency" ? "is-selected" : ""}
                        onClick={() => setDomain("frequency")}
                        disabled={!backendResult}
                        title={!backendResult ? "Run the analysis first" : "Show the backend FFT spectrum"}
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
                                        {domain === "process" && (runMethod === "energy_zcr" ? "Framing + Energy + ZCR" : "Framing + Short-Time Energy")}
                                        {domain === "time" && "Gated Signal Playback"}
                                        {domain === "frequency" && "Frequency Spectrum Comparison"}
                                    </h3>
                                    {activePct !== null && (
                                        <small style={{ color: "rgba(255,255,255,.5)" }}>{activePct}% of frames marked active</small>
                                    )}
                                </div>
                                {(backendResult || (domain === "process" && inputBuffer)) && (
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
                                        {backendResult && (
                                            <button className="secondary-button" onClick={() => setShowModal(true)}>
                                                OPEN RESULTS ↗
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>

                            <div style={{ display: domain === "process" ? "block" : "none" }}>
                                {inputBuffer ? (
                                    <ActivityAnimator
                                        key={`process-${processRunId}`}
                                        inputBuffer={inputBuffer}
                                        sampleRate={sampleRate}
                                        method={runMethod}
                                        energyThresholdRatio={runThresholdPct / 100}
                                        zcrThreshold={zcrHzToRatio(runZcrHz, sampleRate)}
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
                                        mode="activity"
                                        loop={false}
                                        formula="y[n] = x[n] · gain[n]   (gain[n] = 1 if active, → 0 if quiet)"
                                        paramLabel={`threshold = ${runThresholdPct}% of peak · ${activePct !== null ? activePct + "% active" : ""}`}
                                        operationLabel="ACTIVITY GATE"
                                        outputCaption="quiet regions gated toward silence"
                                        progressLabel="Applying the activity gate across the signal…"
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
                                        mode="activity"
                                        loop={false}
                                    />
                                ) : (
                                    <div className="spectrum-empty">
                                        Run the analysis once — the frequency spectrum is calculated by the Python backend.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {!showVisualizer && !isProcessing && (
                    <div className="empty-module-state">
                        Upload a signal, then press&nbsp;<strong style={{ color: "#fff" }}>RUN ACTIVITY DETECTION</strong>.
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
                                <span className="file-chip" style={{ borderColor: "rgba(255,93,108,.4)" }}>
                                    <strong style={{ color: "#ff5d6c" }}>OUTPUT y[n]</strong>
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
                                            <span>y[n] · GATED SIGNAL</span>
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
                                    mode="activity"
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
