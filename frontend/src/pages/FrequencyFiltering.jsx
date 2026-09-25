import { useEffect, useRef, useState } from "react";
import "../styles/time-domain.css";
import { processAudio } from "../services/api";
import EffectAnimator from "../components/Visualizations/EffectAnimator";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";
import FilterResponseGraph from "../components/Visualizations/FilterResponseGraph";

// Five IIR design families (same order/cutoff inputs, five different
// trade-offs) plus the textbook "ideal" brick-wall filter.
const FAMILY_CONTENT = {
    butterworth: {
        label: "BUTTERWORTH",
        name: "Butterworth",
        desc: "Maximally flat passband — no ripples — but a gradual roll-off.",
    },
    chebyshev1: {
        label: "CHEBYSHEV I",
        name: "Chebyshev Type I",
        desc: "Sharper roll-off than Butterworth, at the cost of ripples in the passband.",
    },
    chebyshev2: {
        label: "CHEBYSHEV II",
        name: "Chebyshev Type II",
        desc: "Flat passband with a sharp roll-off — the ripples show up in the stopband instead.",
    },
    elliptic: {
        label: "ELLIPTIC",
        name: "Elliptic (Cauer)",
        desc: "The sharpest possible transition for a given order — but ripples in both the passband and the stopband.",
    },
    bessel: {
        label: "BESSEL",
        name: "Bessel",
        desc: "Optimized for a linear phase response (constant group delay) — preserves waveform shape well, but the roll-off is very gradual.",
    },
    ideal: {
        label: "IDEAL",
        name: "Ideal (Brick-Wall)",
        desc: "The textbook ideal filter: a perfectly sharp cutoff in frequency, built by zeroing FFT bins outright — but that sharp edge rings in time (Gibbs phenomenon).",
    },
};
const FAMILY_ORDER = ["butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel", "ideal"];

const BAND_CONTENT = {
    lowpass: { label: "LOWPASS", name: "Low-Pass", desc: "keeps frequencies below the cutoff", formula: "passes f ≤ fc", isBand: false },
    highpass: { label: "HIGHPASS", name: "High-Pass", desc: "keeps frequencies above the cutoff", formula: "passes f ≥ fc", isBand: false },
    bandpass: { label: "BANDPASS", name: "Band-Pass", desc: "keeps only the frequencies between two cutoffs", formula: "passes fL ≤ f ≤ fH", isBand: true },
    bandstop: { label: "BANDSTOP", name: "Band-Stop", desc: "notches out the frequencies between two cutoffs", formula: "rejects fL ≤ f ≤ fH", isBand: true },
};
const BAND_ORDER = ["lowpass", "highpass", "bandpass", "bandstop"];

const BAND_DEFAULTS = {
    lowpass: { cutoff: 1000, cutoff2: 4000 },
    highpass: { cutoff: 3000, cutoff2: 4000 },
    bandpass: { cutoff: 500, cutoff2: 4000 },
    bandstop: { cutoff: 1000, cutoff2: 3000 },
};

const ORDER_RANGE = { min: 1, max: 10, step: 1 };

// A band (bandpass/bandstop) always needs fL strictly below fH — scipy's
// filter design rejects a degenerate or inverted band, so the sliders are
// clamped against each other rather than letting the user pick an invalid
// pair and only finding out after pressing run.
const MIN_BAND_GAP_HZ = 10;

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

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

export default function FrequencyFiltering() {
    const [family, setFamily] = useState("butterworth");
    const [bandType, setBandType] = useState("lowpass");
    const [domain, setDomain] = useState("time");

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

    const [cutoff, setCutoff] = useState(BAND_DEFAULTS.lowpass.cutoff);
    const [cutoff2, setCutoff2] = useState(BAND_DEFAULTS.lowpass.cutoff2);
    const [order, setOrder] = useState(4);
    // Snapshot of the params actually sent to the backend for the run in
    // progress, so labels/animation reflect what was really processed.
    const [runParams, setRunParams] = useState({ family: "butterworth", bandType: "lowpass", cutoff: 1000, cutoff2: 4000, order: 4 });

    // Each animation plays once on its first mount, then freezes fully
    // revealed. Bumping these forces a remount (a fresh play-through) only
    // when the user explicitly presses a replay button.
    const [timeRunId, setTimeRunId] = useState(0);
    const [freqRunId, setFreqRunId] = useState(0);
    const [modalFreqRunId, setModalFreqRunId] = useState(0);

    const inputRef = useRef(null);
    const audioCtxRef = useRef(null);
    const inputUrlRef = useRef("");

    const familyInfo = FAMILY_CONTENT[family];
    const bandInfo = BAND_CONTENT[bandType];
    const isIdeal = family === "ideal";
    const nyquist = Math.max(1000, Math.floor(sampleRate / 2) - 50);
    const cutoffMax = Math.min(20000, nyquist);

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
        setDomain("time");
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

            const maxHz = Math.min(20000, Math.max(1000, Math.floor(decoded.sampleRate / 2) - 50));
            setCutoff((c) => Math.min(c, maxHz));
            setCutoff2((c) => Math.min(c, maxHz));
        } catch (err) {
            console.error(err);
            setWarning("That file could not be decoded as audio in the browser.");
            return;
        }

        setWarning("");
        resetVisualizerState();
    };

    const switchFamily = (nextFamily) => {
        if (isProcessing) return;
        setFamily(nextFamily);
        resetVisualizerState();
    };

    const switchBand = (nextBand) => {
        if (isProcessing) return;
        setBandType(nextBand);
        const d = BAND_DEFAULTS[nextBand];
        setCutoff(Math.min(d.cutoff, cutoffMax));
        setCutoff2(Math.min(d.cutoff2, cutoffMax));
        resetVisualizerState();
    };

    // Filtering (especially the IIR kind) isn't something you can
    // meaningfully hand-roll sample-by-sample in the browser the way
    // convolution is. So — same approach this app already uses for
    // echo/delay — the animation plays back the backend's own reduced
    // (180-point) time series once the real, full-resolution result comes
    // back from FastAPI.
    const startProcessing = async () => {
        if (!inputFile) return;

        setIsProcessing(true);
        setWarning("");
        setBackendResult(null);
        setShowModal(false);
        setAnimationDone(false);
        setDomain("time");
        setModalDomain("time");
        setShowVisualizer(true);

        const paramsSnapshot = { family, bandType, cutoff, cutoff2, order };
        setRunParams(paramsSnapshot);

        try {
            const result = await processAudio(
                inputFile,
                "filter",
                2.0,
                5, 5, 5,
                null,
                {
                    filter_family: family,
                    band_type: bandType,
                    cutoff,
                    cutoff2: bandInfo.isBand ? cutoff2 : undefined,
                    order: isIdeal ? undefined : order,
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

    const runBandInfo = BAND_CONTENT[runParams.bandType];
    const runFamilyInfo = FAMILY_CONTENT[runParams.family];
    const paramLabel = `fc = ${Math.round(runParams.cutoff)} Hz` +
        (runBandInfo?.isBand ? `   fH = ${Math.round(runParams.cutoff2)} Hz` : "") +
        (runParams.family === "ideal" ? "" : `   N = ${runParams.order}`);
    const runTitleWord = `${runFamilyInfo?.name} ${runBandInfo?.name}`;

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 02 / FREQUENCY &amp; FILTERING</span>
            </div>

            <div className="module-intro">
                <h1>
                    SHAPE THE <span>{familyInfo.name.toUpperCase()}</span><br />
                    {bandInfo.name.toUpperCase()} FILTER.
                </h1>
                <p>
                    A {bandInfo.name.toLowerCase()} filter {bandInfo.desc}. {familyInfo.desc}
                </p>

                <div className="mode-selector-row">
                    {FAMILY_ORDER.map((key) => (
                        <button
                            key={key}
                            className={family === key ? "selected" : ""}
                            onClick={() => switchFamily(key)}
                        >
                            {FAMILY_CONTENT[key].label}
                        </button>
                    ))}
                </div>
                <div className="mode-selector-row" style={{ marginTop: "2px" }}>
                    {BAND_ORDER.map((key) => {
                        return (
                            <button
                                key={key}
                                className={bandType === key ? "selected" : ""}
                                onClick={() => switchBand(key)}
                            >
                                {BAND_CONTENT[key].label}
                            </button>
                        );
                    })}
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
                            <span>{bandInfo.isBand ? "LOW CUTOFF (fL)" : "CUTOFF FREQUENCY (fc)"}</span>
                            <strong>{Math.round(cutoff)} Hz</strong>
                        </div>
                        <input
                            type="range"
                            className="range-input"
                            min={20}
                            max={cutoffMax}
                            step={10}
                            value={cutoff}
                            disabled={isProcessing}
                            style={{ "--pos": (cutoff - 20) / (cutoffMax - 20) }}
                            onChange={(e) => {
                                const raw = Number(e.target.value);
                                setCutoff(bandInfo.isBand ? clamp(raw, 20, cutoff2 - MIN_BAND_GAP_HZ) : raw);
                            }}
                        />
                        <small>{bandInfo.isBand ? "The lower edge of the band." : "The frequency where the filter starts taking effect."}</small>
                    </div>

                    {bandInfo.isBand && (
                        <div className="param-control">
                            <div className="param-control-head">
                                <span>HIGH CUTOFF (fH)</span>
                                <strong>{Math.round(cutoff2)} Hz</strong>
                            </div>
                            <input
                                type="range"
                                className="range-input"
                                min={20}
                                max={cutoffMax}
                                step={10}
                                value={cutoff2}
                                disabled={isProcessing}
                                style={{ "--pos": (cutoff2 - 20) / (cutoffMax - 20) }}
                                onChange={(e) => setCutoff2(clamp(Number(e.target.value), cutoff + MIN_BAND_GAP_HZ, cutoffMax))}
                            />
                            <small>The upper edge of the band.</small>
                        </div>
                    )}

                    {!isIdeal && (
                        <div className="param-control">
                            <div className="param-control-head">
                                <span>FILTER ORDER (N)</span>
                                <strong>{order}</strong>
                            </div>
                            <input
                                type="range"
                                className="range-input"
                                min={ORDER_RANGE.min}
                                max={ORDER_RANGE.max}
                                step={ORDER_RANGE.step}
                                value={order}
                                disabled={isProcessing}
                                style={{ "--pos": (order - ORDER_RANGE.min) / (ORDER_RANGE.max - ORDER_RANGE.min) }}
                                onChange={(e) => setOrder(Number(e.target.value))}
                            />
                            <small>Higher order = steeper roll-off, closer to an ideal brick-wall edge.</small>
                        </div>
                    )}
                </div>

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                <div className="processing-panel" style={{ marginBottom: "24px" }}>
                    <div className="panel-heading">
                        <div>
                            <span className="control-kicker">LIVE PREVIEW · NO AUDIO NEEDED</span>
                            <h3>Filter Response |H(f)|</h3>
                        </div>
                    </div>
                    <FilterResponseGraph
                        filterFamily={family}
                        bandType={bandType}
                        cutoff={cutoff}
                        cutoff2={bandInfo.isBand ? cutoff2 : cutoff + 1000}
                        order={order}
                        sampleRate={sampleRate}
                    />
                </div>

                <div className="math-visual" style={{ marginBottom: "24px" }}>
                    <div className="math-formula"><span>H(f) {bandInfo.formula}</span></div>
                </div>

                <div className="module-action-row" style={{ marginBottom: "8px" }}>
                    <button className="process-main-button" onClick={startProcessing} disabled={isRunDisabled}>
                        {isProcessing ? "FETCHING BACKEND…" : `RUN ${bandInfo.label} FILTER`}
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
                        title={!backendResult ? "Run the filter first" : "Show the backend FFT spectrum"}
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
                                    <h3>{domain === "time" ? "Filtered Signal Playback" : "Frequency Spectrum Multiplication"}</h3>
                                </div>
                                {backendResult && (
                                    <div style={{ display: "flex", gap: "10px" }}>
                                        <button
                                            className="secondary-button"
                                            onClick={() => (domain === "time" ? setTimeRunId((id) => id + 1) : setFreqRunId((id) => id + 1))}
                                        >
                                            ⟳ REPLAY ANIMATION
                                        </button>
                                        <button className="secondary-button" onClick={() => setShowModal(true)}>
                                            OPEN RESULTS ↗
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div style={{ display: domain === "time" ? "block" : "none" }}>
                                {backendResult ? (
                                    <EffectAnimator
                                        key={`time-${timeRunId}`}
                                        visualization={backendResult.visualization}
                                        mode="filter"
                                        loop={false}
                                        onComplete={() => setAnimationDone(true)}
                                        formula={`H(f) ${runBandInfo?.formula}`}
                                        paramLabel={paramLabel}
                                        operationLabel={runTitleWord}
                                        outputCaption="filtered signal"
                                        progressLabel="Applying the filter across the signal…"
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
                                        mode="filter"
                                        loop={false}
                                    />
                                ) : (
                                    <div className="spectrum-empty">
                                        Run the filter once — the frequency spectrum is calculated by the Python backend.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {!showVisualizer && !isProcessing && (
                    <div className="empty-module-state">
                        Upload a signal, then press&nbsp;<strong style={{ color: "#fff" }}>RUN {bandInfo.label} FILTER</strong>.
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
                                            <span>y[n] · FILTERED SIGNAL</span>
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
                                    mode="filter"
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
