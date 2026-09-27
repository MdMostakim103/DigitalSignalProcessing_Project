import { useEffect, useMemo, useRef, useState } from "react";
import "../styles/time-domain.css";
import { getSpectrogram, applySpectralMask } from "../services/api";

// A gold/purple/navy ramp so the heatmap matches the rest of the app's
// accent palette instead of a generic scientific colormap.
const COLOR_STOPS = [
    { t: 0.0, rgb: [10, 10, 30] },
    { t: 0.35, rgb: [70, 40, 120] },
    { t: 0.65, rgb: [190, 70, 150] },
    { t: 1.0, rgb: [247, 184, 1] },
];

function colorAt(t) {
    t = Math.min(1, Math.max(0, t));
    for (let i = 1; i < COLOR_STOPS.length; i++) {
        const a = COLOR_STOPS[i - 1];
        const b = COLOR_STOPS[i];
        if (t <= b.t) {
            const span = b.t - a.t || 1;
            const f = (t - a.t) / span;
            return [
                a.rgb[0] + (b.rgb[0] - a.rgb[0]) * f,
                a.rgb[1] + (b.rgb[1] - a.rgb[1]) * f,
                a.rgb[2] + (b.rgb[2] - a.rgb[2]) * f,
            ];
        }
    }
    return COLOR_STOPS[COLOR_STOPS.length - 1].rgb;
}

// Renders {magnitudeDb, freqBins, frameCount, dbMin, dbMax} (row 0 = lowest
// frequency, as the backend sends it) into an offscreen canvas exactly
// frameCount x freqBins in size — one pixel per STFT cell — then that bitmap
// is stretched onto the visible canvas by the browser. Row order is flipped
// (high frequency at the top) to match how a spectrogram is normally read.
function paintHeatmap(offscreen, spectrogram) {
    const { magnitudeDb, freqBins, frameCount, dbMin, dbMax } = spectrogram;
    if (!freqBins || !frameCount) return;

    offscreen.width = frameCount;
    offscreen.height = freqBins;
    const ctx = offscreen.getContext("2d");
    const image = ctx.createImageData(frameCount, freqBins);
    const range = dbMax - dbMin || 1;

    for (let row = 0; row < freqBins; row++) {
        const freqIndex = freqBins - 1 - row;
        const srcRow = magnitudeDb[freqIndex];
        for (let col = 0; col < frameCount; col++) {
            const t = (srcRow[col] - dbMin) / range;
            const [r, g, b] = colorAt(t);
            const p = (row * frameCount + col) * 4;
            image.data[p] = r;
            image.data[p + 1] = g;
            image.data[p + 2] = b;
            image.data[p + 3] = 255;
        }
    }
    ctx.putImageData(image, 0, 0);
}

const REGION_COLORS = {
    keep: "rgba(92,214,150,0.28)",
    erase: "rgba(255,93,108,0.30)",
};
const REGION_STROKES = {
    keep: "rgba(125,237,173,0.9)",
    erase: "rgba(255,139,151,0.9)",
};

function formatHz(hz) {
    return hz >= 1000 ? `${(hz / 1000).toFixed(1)}k` : `${Math.round(hz)}`;
}

// Interactive when onRegionsChange/currentAction are given (the main
// painting canvas); a plain read-only heatmap otherwise (before/after
// previews in the results modal).
function SpectrogramCanvas({ spectrogram, regions = [], onRegionsChange, currentAction, height = 360 }) {
    const canvasRef = useRef(null);
    const offscreenRef = useRef(document.createElement("canvas"));
    const [draft, setDraft] = useState(null);
    const dragRef = useRef(null);
    const interactive = typeof onRegionsChange === "function";

    const duration = spectrogram?.duration || 0;
    const maxFrequency = spectrogram?.maxFrequency || 0;

    useEffect(() => {
        if (spectrogram?.freqBins) paintHeatmap(offscreenRef.current, spectrogram);
        draw();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [spectrogram]);

    useEffect(() => {
        draw();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [regions, draft]);

    function draw() {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (offscreenRef.current.width) {
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "rgba(255,255,255,.03)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        const drawRect = (rect, action) => {
            const x = Math.min(rect.x0, rect.x1);
            const y = Math.min(rect.y0, rect.y1);
            const w = Math.abs(rect.x1 - rect.x0);
            const h = Math.abs(rect.y1 - rect.y0);
            ctx.fillStyle = REGION_COLORS[action];
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = REGION_STROKES[action];
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
        };

        for (const region of regions) {
            drawRect(
                {
                    x0: (region.timeMin / duration) * canvas.width,
                    x1: (region.timeMax / duration) * canvas.width,
                    y0: (1 - region.freqMax / maxFrequency) * canvas.height,
                    y1: (1 - region.freqMin / maxFrequency) * canvas.height,
                },
                region.action
            );
        }
        if (draft) drawRect(draft, currentAction);
    }

    function pixelToRegion(rect) {
        const x0 = Math.min(rect.x0, rect.x1);
        const x1 = Math.max(rect.x0, rect.x1);
        const y0 = Math.min(rect.y0, rect.y1);
        const y1 = Math.max(rect.y0, rect.y1);
        const canvas = canvasRef.current;
        return {
            timeMin: (x0 / canvas.width) * duration,
            timeMax: (x1 / canvas.width) * duration,
            freqMin: (1 - y1 / canvas.height) * maxFrequency,
            freqMax: (1 - y0 / canvas.height) * maxFrequency,
        };
    }

    function pointerPos(e) {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function handlePointerDown(e) {
        if (!interactive || !spectrogram?.freqBins) return;
        const pos = pointerPos(e);
        dragRef.current = { x0: pos.x, y0: pos.y };
        setDraft({ x0: pos.x, y0: pos.y, x1: pos.x, y1: pos.y });
        e.target.setPointerCapture(e.pointerId);
    }

    function handlePointerMove(e) {
        if (!dragRef.current) return;
        const pos = pointerPos(e);
        setDraft({ x0: dragRef.current.x0, y0: dragRef.current.y0, x1: pos.x, y1: pos.y });
    }

    function handlePointerUp(e) {
        if (!dragRef.current) return;
        const pos = pointerPos(e);
        const rect = { x0: dragRef.current.x0, y0: dragRef.current.y0, x1: pos.x, y1: pos.y };
        dragRef.current = null;
        setDraft(null);

        const canvas = canvasRef.current;
        if (Math.abs(rect.x1 - rect.x0) < 4 || Math.abs(rect.y1 - rect.y0) < 4) return; // ignore accidental clicks
        const region = pixelToRegion(rect);
        onRegionsChange([
            ...regions,
            { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, ...region, action: currentAction },
        ]);
    }

    const yTicks = [1, 0.75, 0.5, 0.25, 0].map((f) => Math.round(f * maxFrequency));

    return (
        <div className="portal-canvas-wrap">
            <canvas
                ref={canvasRef}
                width={900}
                height={height}
                className="portal-canvas"
                style={{ cursor: interactive ? "crosshair" : "default", height: `${height}px` }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
            />
            {spectrogram?.freqBins > 0 && (
                <div className="portal-axis-y">
                    {yTicks.map((hz) => (
                        <span key={hz}>{formatHz(hz)} Hz</span>
                    ))}
                </div>
            )}
        </div>
    );
}

function formatBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function SpectralPortal() {
    const [inputFile, setInputFile] = useState(null);
    const [inputName, setInputName] = useState("");
    const [inputUrl, setInputUrl] = useState("");
    const [inputSize, setInputSize] = useState(0);

    const [spectrogram, setSpectrogram] = useState(null);
    const [regions, setRegions] = useState([]);
    const [currentAction, setCurrentAction] = useState("erase");

    const [isLoadingSpectrogram, setIsLoadingSpectrogram] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [warning, setWarning] = useState("");

    const [result, setResult] = useState(null);
    const [showModal, setShowModal] = useState(false);

    const inputUrlRef = useRef("");

    useEffect(() => {
        return () => {
            if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
        };
    }, []);

    const hasKeepRegion = useMemo(() => regions.some((r) => r.action === "keep"), [regions]);

    const handleFileUpload = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = "";
        if (!file) return;

        if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
        const objectUrl = URL.createObjectURL(file);
        inputUrlRef.current = objectUrl;

        setInputFile(file);
        setInputName(file.name);
        setInputUrl(objectUrl);
        setInputSize(file.size);
        setRegions([]);
        setResult(null);
        setShowModal(false);
        setSpectrogram(null);
        setWarning("");

        setIsLoadingSpectrogram(true);
        try {
            const data = await getSpectrogram(file);
            setSpectrogram(data.spectrogram);
        } catch (err) {
            console.error(err);
            setWarning("Failed to connect to the FastAPI backend on port 8000.");
        } finally {
            setIsLoadingSpectrogram(false);
        }
    };

    const removeRegion = (id) => setRegions((prev) => prev.filter((r) => r.id !== id));
    const clearRegions = () => setRegions([]);

    const applyMask = async () => {
        if (!inputFile || regions.length === 0) return;
        setIsProcessing(true);
        setWarning("");
        try {
            const data = await applySpectralMask(inputFile, regions);
            setResult(data);
            setShowModal(true);
        } catch (err) {
            console.error(err);
            setWarning("Failed to connect to the FastAPI backend on port 8000.");
        } finally {
            setIsProcessing(false);
        }
    };

    const isRunDisabled = isProcessing || !inputFile || regions.length === 0;

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={isProcessing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 07 / SPECTRAL PORTAL</span>
            </div>

            <div className="module-intro">
                <h1>
                    PAINT A SOUND <span>INTO EXISTENCE.</span>
                </h1>
                <p>
                    Draw KEEP or ERASE rectangles directly on the spectrogram. Each one becomes part of a
                    mask M(k,m) multiplied against the real STFT — X'(k,m) = M(k,m)·X(k,m) — then inverted
                    back to audio. Edges are feathered before multiplying so the cut doesn't ring.
                </p>
            </div>

            <div className="module-workspace">
                <div className="module-controls" style={{ marginBottom: "24px" }}>
                    <div>
                        <span className="control-kicker">01 / INPUT SIGNAL</span>
                        <h2>Bring in a WAV signal</h2>
                        <p>Any audio works — a clip with a few distinct sounds (voice + background noise, a chirp, a tone) shows this off best.</p>
                    </div>
                    <div className="upload-group">
                        {inputFile && (
                            <div className="file-chip">
                                <strong title={inputName}>{inputName}</strong>
                                <small>{formatBytes(inputSize)}{spectrogram ? ` · ${spectrogram.duration.toFixed(2)}s` : ""}</small>
                            </div>
                        )}
                        <label className={`upload-module-button ${isProcessing ? "is-disabled" : ""}`}>
                            {inputFile ? "CHANGE WAV" : "CHOOSE WAV"}
                            <input type="file" accept="audio/*" disabled={isProcessing} onChange={handleFileUpload} />
                        </label>
                    </div>
                </div>

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                {!inputFile && !isLoadingSpectrogram && (
                    <div className="empty-module-state">
                        Upload a signal to see its spectrogram, then start painting.
                    </div>
                )}

                {isLoadingSpectrogram && (
                    <div className="empty-module-state">Computing the spectrogram on the backend…</div>
                )}

                {inputFile && spectrogram && (
                    <>
                        <div className="portal-toolbar">
                            <div className="portal-action-toggle">
                                <button
                                    className={`is-keep ${currentAction === "keep" ? "is-selected" : ""}`}
                                    onClick={() => setCurrentAction("keep")}
                                >
                                    ✓ KEEP
                                </button>
                                <button
                                    className={`is-erase ${currentAction === "erase" ? "is-selected" : ""}`}
                                    onClick={() => setCurrentAction("erase")}
                                >
                                    ✕ ERASE
                                </button>
                            </div>
                            <button className="secondary-button" onClick={clearRegions} disabled={regions.length === 0}>
                                CLEAR REGIONS
                            </button>
                            <small style={{ color: "rgba(255,255,255,.5)" }}>
                                {hasKeepRegion
                                    ? "At least one KEEP region → everything else starts silent."
                                    : "No KEEP regions yet → everything stays audible except ERASE boxes."}
                            </small>
                        </div>

                        <SpectrogramCanvas
                            spectrogram={spectrogram}
                            regions={regions}
                            onRegionsChange={setRegions}
                            currentAction={currentAction}
                        />
                        <div className="portal-axis-x">
                            <span>0s</span>
                            <span>{(spectrogram.duration / 2).toFixed(2)}s</span>
                            <span>{spectrogram.duration.toFixed(2)}s</span>
                        </div>

                        {regions.length > 0 && (
                            <div className="portal-region-list">
                                {regions.map((r) => (
                                    <span key={r.id} className={`portal-region-chip is-${r.action}`}>
                                        {r.action === "keep" ? "KEEP" : "ERASE"} {formatHz(r.freqMin)}–{formatHz(r.freqMax)} Hz · {r.timeMin.toFixed(2)}–{r.timeMax.toFixed(2)}s
                                        <button onClick={() => removeRegion(r.id)}>×</button>
                                    </span>
                                ))}
                            </div>
                        )}

                        <div className="module-action-row" style={{ marginTop: "24px" }}>
                            <button className="process-main-button" onClick={applyMask} disabled={isRunDisabled}>
                                {isProcessing ? "APPLYING MASK…" : "APPLY MASK"}
                            </button>
                        </div>
                    </>
                )}

                {showModal && result && (
                    <div className="module-modal-backdrop" onClick={() => setShowModal(false)}>
                        <div className="completion-modal" onClick={(e) => e.stopPropagation()}>
                            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
                            <span className="modal-kicker">BACKEND DSP COMPLETE</span>
                            <h2>Before vs. After</h2>
                            <p>
                                The mask was built from your {regions.length} region{regions.length === 1 ? "" : "s"}, feathered, and
                                multiplied against the real STFT — play both clips to hear what the paint actually did.
                            </p>

                            <div className="modal-audio-buttons">
                                <span className="file-chip" style={{ borderColor: "rgba(78,161,255,.4)" }}>
                                    <strong style={{ color: "#4ea1ff" }}>BEFORE x[n]</strong>
                                    <audio controls src={inputUrl} style={{ width: "220px", marginTop: "6px" }} />
                                </span>
                                <span className="file-chip" style={{ borderColor: "rgba(255,93,108,.4)" }}>
                                    <strong style={{ color: "#ff5d6c" }}>AFTER y[n]</strong>
                                    <audio controls src={result.audio_url} style={{ width: "220px", marginTop: "6px" }} />
                                </span>
                            </div>

                            <div className="portal-spectrogram-pair">
                                <div>
                                    <div className="result-graph-head"><span>BEFORE</span><small>{result.input_duration_seconds}s</small></div>
                                    <SpectrogramCanvas spectrogram={result.before} height={220} />
                                </div>
                                <div>
                                    <div className="result-graph-head"><span>AFTER</span><small>{result.processed_duration_seconds}s</small></div>
                                    <SpectrogramCanvas spectrogram={result.after} height={220} />
                                </div>
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
