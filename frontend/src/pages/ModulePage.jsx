import { useEffect, useMemo, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import { modules } from "../data/modules";

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

function WaveCanvas({ mode, progress, parameter, samples }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(320, Math.floor(rect.width));
        const height = 300;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);

        const mid = height / 2;
        ctx.strokeStyle = "rgba(255,255,255,.10)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (height * i) / 4;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
        }
        for (let i = 0; i <= 8; i++) {
            const x = (width * i) / 8;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        }

        const drawWave = (offset, amp, freq, dashed = false) => {
            ctx.beginPath();
            ctx.setLineDash(dashed ? [6, 7] : []);
            for (let x = 0; x <= width; x++) {
                const t = x / width;
                const sampleIndex = samples?.length
                    ? Math.min(samples.length - 1, Math.floor(t * samples.length))
                    : 0;
                let y = samples?.length
                    ? samples[sampleIndex] * amp
                    : Math.sin(t * Math.PI * 2 * freq + offset) * amp;
                if (!samples?.length) {
                    y += Math.sin(t * Math.PI * 2 * freq * 2.1 + offset * .7) * amp * .22;
                }
                if (mode === "filter" && progress > .42) {
                    const attenuation = t > clamp(parameter / 100, .08, .92) ? .22 : 1;
                    y *= attenuation;
                }
                if (mode === "activity" && t > .35 && t < .68 && progress > .45) y *= 1.65;
                if (mode === "pitch" && progress > .5) y = Math.sin(t * Math.PI * 2 * (freq + parameter / 8) + offset) * amp;
                if (mode === "morph" && progress > .55) y = Math.sin(t * Math.PI * 2 * freq + offset) * amp * (0.7 + parameter / 160);
                const py = mid - y;
                if (x === 0) ctx.moveTo(x, py); else ctx.lineTo(x, py);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        };

        if (mode === "convolution") {
            drawWave(0, 54, 3);
            const impulseX = width * (0.16 + progress * .64);
            ctx.strokeStyle = "rgba(247,184,1,.95)";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(impulseX, 62);
            ctx.lineTo(impulseX, 238);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(impulseX - 28, 238);
            ctx.lineTo(impulseX, 62);
            ctx.lineTo(impulseX + 28, 238);
            ctx.stroke();
        } else {
            ctx.strokeStyle = "rgba(255,255,255,.9)";
            drawWave(progress * Math.PI * 2, 62, mode === "pitch" ? 4 : 3.2);
            if (progress > .52) {
                ctx.strokeStyle = "rgba(247,184,1,.8)";
                drawWave(progress * Math.PI * 2 + .35, 38, mode === "pitch" ? 5 : 2.2, true);
            }
        }
    }, [mode, progress, parameter, samples]);

    return <canvas ref={canvasRef} className="module-visual-canvas" aria-label="Interactive signal visualization" />;
}

function Spectrum({ progress, parameter, samples }) {
    const bars = Array.from({ length: 48 }, (_, i) => {
        if (samples?.length) {
            const frame = samples.subarray(
                Math.max(0, Math.floor(samples.length / 2) - 512),
                Math.min(samples.length, Math.floor(samples.length / 2) + 512)
            );
            const bin = Math.max(1, Math.floor((i + 1) * frame.length / 96));
            let real = 0;
            let imag = 0;
            for (let n = 0; n < frame.length; n += 4) {
                const angle = (2 * Math.PI * bin * n) / frame.length;
                real += frame[n] * Math.cos(angle);
                imag -= frame[n] * Math.sin(angle);
            }
            const magnitude = Math.sqrt(real * real + imag * imag) / Math.max(1, frame.length / 4);
            return clamp(magnitude * 7, .03, 1);
        }
        const peak = Math.exp(-Math.pow((i - (8 + parameter / 9)) / 5, 2));
        const noise = 0.12 + ((i * 17) % 9) / 80;
        const base = progress < .35 ? noise + peak * .55 : noise + peak;
        return clamp(base * (progress > .55 && i > parameter / 2.1 ? .18 : 1), .03, 1);
    });

    return (
        <div className="spectrum-bars" aria-label="Frequency spectrum visualization">
            {bars.map((height, i) => (
                <span key={i} style={{ height: `${height * 100}%` }} />
            ))}
        </div>
    );
}

function StageRow({ stages, active }) {
    return (
        <div className="module-stages">
            {stages.map((stage, i) => (
                <div className={`module-stage ${i === active ? "active" : ""} ${i < active ? "done" : ""}`} key={stage}>
                    <span>{String(i + 1).padStart(2, "0")}</span>
                    <strong>{stage}</strong>
                </div>
            ))}
        </div>
    );
}

function ModulePage({ moduleNumber }) {
    const module = useMemo(
        () => modules.find((item) => item.number === moduleNumber) || modules[0],
        [moduleNumber]
    );

    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [parameter, setParameter] = useState(55);
    const [fileName, setFileName] = useState("");
    const [signalSamples, setSignalSamples] = useState(null);

    useEffect(() => {
        setRunning(false);
        setProgress(0);
        setFileName("");
        setSignalSamples(null);
    }, [moduleNumber]);

    useEffect(() => {
        if (!running) return undefined;
        const started = performance.now();
        const duration = 5200;
        let frame;

        const tick = (now) => {
            const p = clamp((now - started) / duration, 0, 1);
            setProgress(p);
            if (p < 1) frame = requestAnimationFrame(tick);
            else setRunning(false);
        };

        frame = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(frame);
    }, [running]);

    const activeStage = Math.min(
        module.stages.length - 1,
        Math.floor(progress * module.stages.length)
    );

    const visualMode =
        moduleNumber === "02" ? "filter" :
        moduleNumber === "03" ? "convolution" :
        moduleNumber === "04" ? "activity" :
        moduleNumber === "05" ? "pitch" :
        moduleNumber === "06" ? "morph" : "amplitude";

    const chooseFile = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        setFileName(file.name);
        try {
            const bytes = await file.arrayBuffer();
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const context = new AudioCtx();
            const buffer = await context.decodeAudioData(bytes.slice(0));
            const source = buffer.getChannelData(0);
            const target = Math.min(4096, source.length);
            const samples = new Float32Array(target);
            for (let i = 0; i < target; i += 1) {
                samples[i] = source[Math.floor((i / target) * source.length)] || 0;
            }
            setSignalSamples(samples);
            await context.close();
        } catch {
            setSignalSamples(null);
        }
    };

    const play = () => {
        setProgress(0);
        setRunning(true);
    };

    return (
        <>
            <Navbar />
            <main className="module-page">
                <section className="module-hero">
                    <button className="module-back" type="button" onClick={() => { window.location.hash = "home"; }}>
                        ← Back to modules
                    </button>

                    <div className="module-page-kicker">
                        MODULE {module.number} / INTERACTIVE DSP
                    </div>

                    <h1>{module.title}</h1>
                    <p>{module.intro}</p>

                    <div className="module-input-row">
                        <label className="module-file">
                            <span>{fileName || "Optional: choose a WAV file"}</span>
                            <input type="file" accept="audio/*" onChange={chooseFile} />
                        </label>
                        <button className="module-run" type="button" onClick={play}>
                            {running ? "PROCESSING…" : "PLAY THE PROCESS"}
                        </button>
                    </div>
                </section>

                <section className="module-workspace">
                    <div className="module-workspace-head">
                        <div>
                            <span className="eyebrow">LIVE SIGNAL VIEW</span>
                            <h2>{module.shortTitle}</h2>
                        </div>
                        <span className="progress-readout">{Math.round(progress * 100)}%</span>
                    </div>

                    <div className="module-visual">
                        <WaveCanvas mode={visualMode} progress={progress} parameter={parameter} samples={signalSamples} />
                        {(moduleNumber === "02" || moduleNumber === "05") && (
                            <div className="spectrum-overlay">
                                <span>FREQUENCY DOMAIN</span>
                                <Spectrum progress={progress} parameter={parameter} samples={signalSamples} />
                            </div>
                        )}
                        {moduleNumber === "03" && (
                            <div className="convolution-note">
                                {progress < .35 ? "Input signal" :
                                    progress < .58 ? "Slide the impulse response" :
                                        progress < .82 ? "Multiply + accumulate" :
                                            "Output samples are built"}
                            </div>
                        )}
                    </div>

                    <StageRow stages={module.stages} active={activeStage} />

                    <div className="module-controls">
                        <label>
                            <span>PROCESS PARAMETER</span>
                            <input
                                type="range"
                                min="10"
                                max="90"
                                value={parameter}
                                onChange={(e) => setParameter(Number(e.target.value))}
                            />
                        </label>
                        <div className="module-explanation" aria-live="polite">
                            <span>WHAT IS HAPPENING</span>
                            <strong>{module.stages[activeStage]}</strong>
                            <p>
                                {moduleNumber === "02" && activeStage === 1
                                    ? "The waveform is represented as frequency components so we can inspect and modify individual regions of the spectrum."
                                    : moduleNumber === "03" && activeStage >= 2
                                        ? "The impulse response moves across the signal. Samples that overlap are multiplied and added to form each output sample."
                                        : moduleNumber === "04" && activeStage >= 2
                                            ? "Each short frame gets an energy value. Frames above the activity threshold are marked as active."
                                            : moduleNumber === "05" && activeStage >= 2
                                                ? "The strongest spectral peak provides an estimate of the dominant frequency and therefore the perceived pitch."
                                                : moduleNumber === "06" && activeStage >= 2
                                                    ? "Spectral components are shifted and reconstructed, illustrating how pitch can be manipulated without simply changing playback speed."
                                                    : "The signal is being measured and transformed at this stage of the DSP pipeline."}
                            </p>
                        </div>
                    </div>
                </section>

                <section className="module-footer-note">
                    <span>LEARNING MODE</span>
                    <p>
                        This view is designed to expose the DSP operation itself.
                        The visualization progresses through the same conceptual stages
                        you can explain during your project demonstration.
                    </p>
                </section>
            </main>
        </>
    );
}

export default ModulePage;
