import { useRef, useState } from "react";
import "../styles/time-domain.css";
import { detectBirdSound } from "../services/api";

const RECORD_SECONDS = 5;

// Turn a decoded AudioBuffer's mono channel into a standard 16-bit PCM WAV
// Blob, entirely in the browser. We do this instead of uploading the raw
// MediaRecorder blob (webm/ogg) because the backend decodes with the same
// librosa/soundfile pipeline every other module uses, and a plain WAV is
// what that pipeline is built and tested against.
function encodeWav(float32Samples, sampleRate) {
    const numSamples = float32Samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, "data");
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, float32Samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
}

// Small dependency-free bar chart for the backend's binned FFT magnitude —
// same shape of data as every other module's frequency view, just drawn
// directly instead of through FrequencySpectrumAnimator (which is built
// for an input-vs-output comparison, not a single detection spectrum).
function SpectrumBars({ magnitude, highlightHz, maxFrequency }) {
    if (!magnitude || magnitude.length === 0) {
        return <div className="spectrum-empty">No spectrum yet.</div>;
    }
    const peak = Math.max(...magnitude, 1e-9);
    const highlightIndex =
        highlightHz && maxFrequency
            ? Math.round((highlightHz / maxFrequency) * (magnitude.length - 1))
            : -1;

    return (
        <div className="spectrum-bars" style={{ alignItems: "flex-end", height: "160px" }}>
            {magnitude.map((v, i) => (
                <span
                    key={i}
                    style={{
                        height: `${Math.max(2, (v / peak) * 100)}%`,
                        background: i === highlightIndex ? "#f7b801" : undefined,
                    }}
                />
            ))}
        </div>
    );
}

function MiniWaveGraph({ data, color = "#4ea1ff" }) {
    const width = 600;
    const height = 120;
    const midY = height / 2;
    const usable = midY - 8;

    if (!data || data.length < 2) {
        return (
            <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                <line x1="0" y1={midY} x2={width} y2={midY} className="result-axis" />
            </svg>
        );
    }

    const max = data.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6);
    const step = width / (data.length - 1);
    const path = data
        .map((v, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(2)} ${(midY - (v / max) * usable).toFixed(2)}`)
        .join(" ");

    return (
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            <line x1="0" y1={midY} x2={width} y2={midY} className="result-axis" />
            <path d={path} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

export default function BirdDetector() {
    const [status, setStatus] = useState("idle"); // idle | requesting | recording | analyzing | done | error
    const [countdown, setCountdown] = useState(RECORD_SECONDS);
    const [warning, setWarning] = useState("");
    const [result, setResult] = useState(null);
    const [recordingUrl, setRecordingUrl] = useState("");

    const streamRef = useRef(null);
    const audioCtxRef = useRef(null);
    const countdownTimerRef = useRef(null);

    const cleanupStream = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }
    };

    const startRecording = async () => {
        setWarning("");
        setResult(null);
        setStatus("requesting");

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error(err);
            setWarning("Microphone access was blocked or is unavailable in this browser.");
            setStatus("error");
            return;
        }
        streamRef.current = stream;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);

        // A ScriptProcessor tap captures raw PCM directly as it arrives, so
        // the exact samples that get analyzed are the exact samples that
        // were captured — no intermediate lossy codec (webm/opus) in between.
        // Deprecated but universally supported; fine for a fixed 3s capture.
        const bufferSize = 4096;
        const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
        const chunks = [];

        processor.onaudioprocess = (e) => {
            chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);

        setStatus("recording");
        setCountdown(RECORD_SECONDS);
        countdownTimerRef.current = setInterval(() => {
            setCountdown((c) => (c > 1 ? c - 1 : 0));
        }, 1000);

        setTimeout(async () => {
            clearInterval(countdownTimerRef.current);
            processor.disconnect();
            source.disconnect();
            cleanupStream();

            const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
            const merged = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                merged.set(chunk, offset);
                offset += chunk.length;
            }

            const wavBlob = encodeWav(merged, audioCtx.sampleRate);
            setRecordingUrl((prev) => {
                if (prev) URL.revokeObjectURL(prev);
                return URL.createObjectURL(wavBlob);
            });

            setStatus("analyzing");
            try {
                const response = await detectBirdSound(wavBlob);
                setResult(response);
                setStatus("done");
            } catch (err) {
                console.error(err);
                setWarning("Failed to connect to the FastAPI backend on port 8000.");
                setStatus("error");
            }
        }, RECORD_SECONDS * 1000);
    };

    const detection = result?.detection;
    const spectrum = result?.visualization?.spectrum;
    const waveform = result?.visualization?.waveform;

    return (
        <main className="time-domain-page">
            <div className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = ""; }} disabled={status === "recording" || status === "analyzing"}>
                    ← DSP MODULES
                </button>
                <span>BONUS MODULE / BIRD SOUND DETECTOR</span>
            </div>

            <div className="module-intro">
                <h1>
                    IDENTIFY A BIRD CALL <span>WITH DSP, NOT AI.</span>
                </h1>
                <p>
                    Record {RECORD_SECONDS} seconds of audio. The backend reduces it to six classic signal descriptors — energy,
                    zero-crossing rate, dominant frequency, spectral centroid, bandwidth, and rolloff — and reports
                    which reference species' averaged descriptors it is closest to, using a weighted distance, not a
                    trained model. Below a similarity threshold it reports "Unknown" instead of guessing.
                </p>
            </div>

            <div className="module-workspace">
                <div className="module-controls" style={{ marginBottom: "24px" }}>
                    <div>
                        <span className="control-kicker">01 / CAPTURE</span>
                        <h2>Record a {RECORD_SECONDS}-second clip</h2>
                        <p>Nothing you record is saved to disk — it's analyzed in memory and discarded.</p>
                    </div>
                    <div className="module-action-row">
                        <button
                            className="process-main-button"
                            onClick={startRecording}
                            disabled={status === "requesting" || status === "recording" || status === "analyzing"}
                        >
                            {status === "recording"
                                ? `RECORDING… ${countdown}s`
                                : status === "analyzing"
                                ? "ANALYZING…"
                                : status === "requesting"
                                ? "REQUESTING MIC…"
                                : `START ${RECORD_SECONDS}-SECOND RECORDING`}
                        </button>
                    </div>
                </div>

                {warning && <div className="module-error" style={{ marginBottom: "20px" }}><strong>Wait!</strong> {warning}</div>}

                {!result && status !== "recording" && status !== "analyzing" && (
                    <div className="empty-module-state">
                        Press&nbsp;<strong style={{ color: "#fff" }}>START {RECORD_SECONDS}-SECOND RECORDING</strong>&nbsp;and make (or play) a bird call near your mic.
                    </div>
                )}

                {result && (
                    <div className="signal-sections">
                        <div className="processing-panel is-processing" style={{ gridColumn: "1 / -1", width: "100%" }}>
                            <div className="panel-heading">
                                <div>
                                    <span className="control-kicker">RESULT</span>
                                    <h3>
                                        {detection.isConfident ? (
                                            <>Most similar to: <span style={{ color: "#f7b801" }}>{detection.species}</span></>
                                        ) : (
                                            <>Unknown — no confident match</>
                                        )}
                                    </h3>
                                    <small style={{ color: "rgba(255,255,255,.5)" }}>
                                        similarity confidence: {detection.confidence}% · distance {detection.bestDistance} (threshold {detection.threshold})
                                    </small>
                                </div>
                            </div>

                            {recordingUrl && (
                                <audio controls src={recordingUrl} style={{ width: "260px", margin: "12px 0" }} />
                            )}

                            <div className="result-graphs">
                                <div className="result-graph-card">
                                    <div className="result-graph-head">
                                        <span>RECORDED WAVEFORM</span>
                                        <small>amplitude vs time</small>
                                    </div>
                                    <div className="result-graph-body">
                                        <MiniWaveGraph data={waveform} />
                                    </div>
                                </div>
                                <div className="result-graph-card">
                                    <div className="result-graph-head">
                                        <span>FFT SPECTRUM</span>
                                        <small>0 – {Math.round(spectrum?.maxFrequency || 0)} Hz</small>
                                    </div>
                                    <div className="result-graph-body">
                                        <SpectrumBars
                                            magnitude={spectrum?.magnitude}
                                            maxFrequency={spectrum?.maxFrequency}
                                            highlightHz={detection.featureVector?.dominantFreq}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="math-visual" style={{ marginTop: "20px" }}>
                                <div className="math-formula">
                                    <span>
                                        distance(clip, species) = ‖ w ⊙ (features_clip − features_species) / σ ‖₂
                                    </span>
                                </div>
                            </div>

                            <div className="result-graph-hint" style={{ marginTop: "12px" }}>
                                All distances this clip got, closest first:{" "}
                                {Object.entries(detection.allDistances)
                                    .sort((a, b) => a[1] - b[1])
                                    .map(([species, d]) => `${species} (${d})`)
                                    .join(" · ")}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}
