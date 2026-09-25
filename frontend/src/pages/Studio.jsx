import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./../styles/studio.css";
import { processChain, getFilterResponse } from "../services/api";

const FILTER_FAMILIES = [
    { value: "butterworth", label: "Butterworth" },
    { value: "chebyshev1", label: "Chebyshev I" },
    { value: "chebyshev2", label: "Chebyshev II" },
    { value: "elliptic", label: "Elliptic" },
    { value: "bessel", label: "Bessel" },
    { value: "ideal", label: "Ideal (brick-wall)" },
];

const BAND_TYPES = [
    { value: "lowpass", label: "Low-pass" },
    { value: "highpass", label: "High-pass" },
    { value: "bandpass", label: "Band-pass" },
    { value: "bandstop", label: "Band-stop" },
];

const MORPH_MODES = [
    { value: "pitch", label: "Pitch Shift" },
    { value: "stretch", label: "Time Stretch" },
    { value: "robot", label: "Robot (zero phase)" },
    { value: "whisper", label: "Whisper (random phase)" },
];

// One entry per chainable backend operation. `defaultParams` mirrors exactly
// what /process-chain expects for that step type, so building the request
// payload is just `{ type, params }` with no translation layer.
const OPERATIONS = [
    { type: "amplify", label: "Amplify", short: "GAIN", defaultParams: { value: 2 } },
    {
        type: "filter", label: "Filter", short: "FILTER",
        defaultParams: { filter_family: "butterworth", band_type: "lowpass", cutoff: 1000, cutoff2: 4000, order: 4 },
    },
    { type: "convolution", label: "Convolution", short: "CONV", defaultParams: {}, requiresIr: true },
    { type: "echo", label: "Echo", short: "ECHO", defaultParams: { delay_ms: 280, decay: 0.55, repeats: 5 } },
    { type: "delay", label: "Delay", short: "DELAY", defaultParams: { delay_ms: 350, wet: 0.85 } },
    { type: "reverb", label: "Reverb", short: "REVERB", defaultParams: {} },
    { type: "equalizer", label: "Equalizer", short: "EQ", defaultParams: { low: 5, mid: 5, high: 5 } },
    { type: "noise", label: "Noise Reduction", short: "DENOISE", defaultParams: {} },
    { type: "activity", label: "Activity Gate", short: "GATE", defaultParams: { threshold_ratio: 0.15 } },
    {
        type: "morph", label: "Voice Morph", short: "MORPH",
        defaultParams: { morph_mode: "pitch", n_steps: 4, rate: 1.5 },
    },
];

const OPERATION_BY_TYPE = Object.fromEntries(OPERATIONS.map((op) => [op.type, op]));

const HISTORY_KEY = "dsp-studio-chain-history";
const MAX_HISTORY = 5;
const MAX_CHAIN_STEPS = 10;

// A band (bandpass/bandstop) always needs its low edge strictly below its
// high edge — scipy's filter design rejects a degenerate/inverted band, so
// the two cutoff sliders are clamped against each other.
const MIN_BAND_GAP_HZ = 10;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

function formatDb(factor) {
    const db = 20 * Math.log10(Math.max(factor, 0.0001));
    return `${db >= 0 ? "+" : "−"}${Math.abs(db).toFixed(1)} dB`;
}

function loadHistory() {
    try {
        const raw = window.localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function saveHistory(history) {
    try {
        window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
        // Storage full or unavailable — history just won't persist across reloads.
    }
}

function describeStep(type, params) {
    switch (type) {
        case "amplify":
            return `Amplify · x${Number(params.value).toFixed(2)} (${formatDb(params.value)})`;
        case "filter": {
            const family = FILTER_FAMILIES.find((f) => f.value === params.filter_family)?.label || params.filter_family;
            const band = BAND_TYPES.find((b) => b.value === params.band_type)?.label || params.band_type;
            const isBand = params.band_type === "bandpass" || params.band_type === "bandstop";
            const range = isBand ? `${Math.round(params.cutoff)}–${Math.round(params.cutoff2)} Hz` : `${Math.round(params.cutoff)} Hz`;
            const order = params.filter_family === "ideal" ? "" : ` · order ${params.order}`;
            return `Filter · ${family} ${band} @ ${range}${order}`;
        }
        case "convolution":
            return `Convolution · IR: ${params.irFileName || "not set"}`;
        case "echo":
            return `Echo · ${Math.round(params.delay_ms)}ms delay · decay ${params.decay.toFixed(2)} · ×${params.repeats}`;
        case "delay":
            return `Delay · ${Math.round(params.delay_ms)}ms · wet ${Math.round(params.wet * 100)}%`;
        case "reverb":
            return "Reverb · fixed room impulse response";
        case "equalizer":
            return `Equalizer · low ${params.low} · mid ${params.mid} · high ${params.high}`;
        case "noise":
            return "Noise Reduction · spectral gate";
        case "activity":
            return `Activity Gate · threshold ${Math.round(params.threshold_ratio * 100)}%`;
        case "morph": {
            const mode = MORPH_MODES.find((m) => m.value === params.morph_mode)?.label || params.morph_mode;
            const extra = params.morph_mode === "pitch" ? ` · ${params.n_steps > 0 ? "+" : ""}${params.n_steps} st`
                : params.morph_mode === "stretch" ? ` · ×${params.rate}` : "";
            return `Voice Morph · ${mode}${extra}`;
        }
        default:
            return type;
    }
}

function NumberField({ label, value, min, max, step, unit = "", format, onChange, disabled }) {
    return (
        <label className="chain-field">
            <span>{label}</span>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                disabled={disabled}
                onChange={(event) => onChange(Number(event.target.value))}
            />
            <output>{format ? format(value) : `${value}${unit}`}</output>
        </label>
    );
}

function SelectField({ label, value, options, onChange, disabled }) {
    return (
        <label className="chain-field chain-field--select">
            <span>{label}</span>
            <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
            </select>
        </label>
    );
}

// A small always-visible sparkline of |H(f)| for the filter step currently
// configured — lets the user see the shape of the filter they're about to
// apply while still arranging the chain, before anything has run.
function FilterMiniPreview({ params, sampleRate }) {
    const [curve, setCurve] = useState(null);

    useEffect(() => {
        let cancelled = false;
        const timer = window.setTimeout(async () => {
            try {
                const data = await getFilterResponse({
                    filterFamily: params.filter_family,
                    bandType: params.band_type,
                    cutoff: params.cutoff,
                    cutoff2: params.cutoff2,
                    order: params.order,
                    sampleRate: sampleRate || 44100,
                });
                if (!cancelled) setCurve(data);
            } catch {
                if (!cancelled) setCurve(null);
            }
        }, 200);

        return () => {
            cancelled = true;
            window.clearTimeout(timer);
        };
    }, [params.filter_family, params.band_type, params.cutoff, params.cutoff2, params.order, sampleRate]);

    const path = useMemo(() => {
        if (!curve?.magnitude?.length) return "";
        const max = Math.max(...curve.magnitude, 1e-9);
        const w = 160;
        const h = 34;
        return curve.magnitude
            .map((m, i) => {
                const x = (i / (curve.magnitude.length - 1)) * w;
                const y = h - clamp(m / max, 0, 1) * (h - 4) - 2;
                return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
            })
            .join(" ");
    }, [curve]);

    return (
        <div className="filter-mini-preview">
            <span>|H(f)| preview</span>
            <svg viewBox="0 0 160 34" preserveAspectRatio="none">
                {path && <path d={path} fill="none" stroke="var(--input-accent)" strokeWidth="1.6" />}
            </svg>
        </div>
    );
}

// Static time or frequency graph — no scanning cursor, no zoom, no
// animation. Draws once from whatever data it's given and redraws on
// resize only.
function StaticGraph({ title, type, data, color = "var(--input-accent)", height = 150 }) {
    const canvasRef = useRef(null);

    // Resolve CSS variable colours once so we can use them in gradients.
    const resolveColor = (cssColor) => {
        if (!cssColor.startsWith("var(")) return cssColor;
        const varName = cssColor.slice(4, -1).trim();
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || "#72e6ff";
    };

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        canvas.width = width * dpr;
        canvas.height = h * dpr;

        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, h);

        // Dark purple background matching the project palette
        ctx.fillStyle = "rgba(25, 27, 95, 0.80)";
        ctx.fillRect(0, 0, width, h);

        const left = 12;
        const right = width - 12;
        const top = 12;
        const bottom = h - 12;
        const gw = right - left;
        const gh = bottom - top;

        // Subtle horizontal grid lines
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        for (let g = 0; g <= 4; g++) {
            const y = top + (g / 4) * gh;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(right, y);
            ctx.stroke();
        }

        // Centre line — more visible
        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(left, top + gh / 2);
        ctx.lineTo(right, top + gh / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        const resolvedColor = resolveColor(color);

        if (type === "time") {
            const values = data || [];
            if (!values.length) return;

            let peak = 0;
            for (let i = 0; i < values.length; i += 1) peak = Math.max(peak, Math.abs(values[i]));
            const scale = peak > 0 ? peak : 1;

            const midY = top + gh / 2;

            // Build path
            const points = values.map((v, i) => ({
                x: left + (i / Math.max(1, values.length - 1)) * gw,
                y: midY - (v / scale) * (gh / 2) * 0.88,
            }));

            // Gradient fill under the waveform
            const fillGrad = ctx.createLinearGradient(0, top, 0, bottom);
            fillGrad.addColorStop(0, `${resolvedColor}55`);
            fillGrad.addColorStop(0.5, `${resolvedColor}22`);
            fillGrad.addColorStop(1, `${resolvedColor}05`);

            ctx.beginPath();
            ctx.moveTo(points[0].x, midY);
            points.forEach((p) => ctx.lineTo(p.x, p.y));
            ctx.lineTo(points[points.length - 1].x, midY);
            ctx.closePath();
            ctx.fillStyle = fillGrad;
            ctx.fill();

            // Stroke line on top
            const lineGrad = ctx.createLinearGradient(left, 0, right, 0);
            lineGrad.addColorStop(0, `${resolvedColor}bb`);
            lineGrad.addColorStop(0.5, resolvedColor);
            lineGrad.addColorStop(1, `${resolvedColor}bb`);

            ctx.beginPath();
            points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
            ctx.strokeStyle = lineGrad;
            ctx.lineWidth = 2.5;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.stroke();
        } else {
            const magnitude = data?.magnitude || [];
            if (!magnitude.length) return;

            const displayMax = data.displayMax || Math.max(...magnitude, 1e-9);
            const barCount = magnitude.length;
            const barWidth = Math.max(2, gw / barCount);
            const gap = barWidth > 4 ? 1 : 0;

            magnitude.forEach((m, i) => {
                const ratio = clamp(m / (displayMax || 1), 0, 1);
                if (ratio < 0.002) return;
                const barHeight = Math.max(2, ratio * gh);
                const x = left + i * (gw / barCount);
                const y = bottom - barHeight;

                const barGrad = ctx.createLinearGradient(0, y, 0, bottom);
                barGrad.addColorStop(0, resolvedColor);
                barGrad.addColorStop(1, `${resolvedColor}55`);

                ctx.globalAlpha = 0.55 + ratio * 0.45;
                ctx.fillStyle = barGrad;
                ctx.fillRect(x, y, Math.max(1, barWidth - gap), barHeight);

                // Bright cap on top of each bar
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = resolvedColor;
                ctx.fillRect(x, y, Math.max(1, barWidth - gap), Math.min(2, barHeight));
            });
            ctx.globalAlpha = 1;
        }
    }, [data, type, color]);

    useEffect(() => {
        draw();
        const onResize = () => draw();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [draw]);

    return (
        <div className="stage-graph">
            <div className="stage-graph__title">{title}</div>
            <canvas ref={canvasRef} className="stage-graph__canvas" style={{ height }} />
        </div>
    );
}

function StatusDot({ status }) {
    return <span className={`step-status-dot step-status-dot--${status}`} title={status} />;
}

function ChainStepCard({
    step,
    index,
    total,
    status,
    stageResult,
    sampleRate,
    expanded,
    onToggleExpanded,
    onMoveUp,
    onMoveDown,
    onRemove,
    onParamsChange,
    onOpenIrModal,
    disabled,
}) {
    const meta = OPERATION_BY_TYPE[step.type];
    const params = step.params;
    const irMissing = step.type === "convolution" && !step.irFile;
    const isBandType = params.band_type === "bandpass" || params.band_type === "bandstop";

    const setParam = (key, value) => onParamsChange({ ...params, [key]: value });

    return (
        <div className={`chain-step ${irMissing ? "chain-step--warning" : ""}`}>
            <div className="chain-step__header">
                <div className="chain-step__title">
                    <StatusDot status={status} />
                    <span className="chain-step__index">{index + 1}</span>
                    <strong>{meta.label}</strong>
                    {stageResult && <span className="chain-step__timing">{stageResult.elapsed_ms.toFixed(1)} ms</span>}
                </div>

                <div className="chain-step__actions">
                    <button type="button" disabled={disabled || index === 0} onClick={onMoveUp} aria-label="Move step up">↑</button>
                    <button type="button" disabled={disabled || index === total - 1} onClick={onMoveDown} aria-label="Move step down">↓</button>
                    <button type="button" disabled={disabled} onClick={onRemove} aria-label="Remove step" className="chain-step__remove">✕</button>
                </div>
            </div>

            <p className="chain-step__summary">{describeStep(step.type, step.type === "convolution" ? { irFileName: step.irFileName } : params)}</p>

            <div className="chain-step__params">
                {step.type === "amplify" && (
                    <NumberField label="Gain factor" min={0.01} max={4} step={0.01} value={params.value}
                        format={(v) => `x${v.toFixed(2)} (${formatDb(v)})`} disabled={disabled}
                        onChange={(v) => setParam("value", v)} />
                )}

                {step.type === "filter" && (
                    <>
                        <SelectField label="Family" value={params.filter_family} options={FILTER_FAMILIES} disabled={disabled}
                            onChange={(v) => setParam("filter_family", v)} />
                        <SelectField label="Band type" value={params.band_type} options={BAND_TYPES} disabled={disabled}
                            onChange={(v) => setParam("band_type", v)} />
                        <NumberField label={isBandType ? "Low cutoff" : "Cutoff"}
                            min={20} max={Math.min(20000, sampleRate ? sampleRate / 2 - 100 : 20000)} step={10}
                            value={params.cutoff} unit=" Hz" disabled={disabled}
                            onChange={(v) => setParam("cutoff", isBandType ? clamp(v, 20, params.cutoff2 - MIN_BAND_GAP_HZ) : v)} />
                        {isBandType && (
                            <NumberField label="High cutoff" min={20} max={Math.min(20000, sampleRate ? sampleRate / 2 - 50 : 20000)} step={10}
                                value={params.cutoff2} unit=" Hz" disabled={disabled}
                                onChange={(v) => setParam("cutoff2", clamp(v, params.cutoff + MIN_BAND_GAP_HZ, Math.min(20000, sampleRate ? sampleRate / 2 - 50 : 20000)))} />
                        )}
                        {params.filter_family !== "ideal" && (
                            <NumberField label="Order" min={1} max={10} step={1} value={params.order} disabled={disabled}
                                onChange={(v) => setParam("order", v)} />
                        )}
                        <FilterMiniPreview params={params} sampleRate={sampleRate} />
                    </>
                )}

                {step.type === "convolution" && (
                    <div className="chain-field chain-field--ir">
                        <span>Impulse response</span>
                        <div className="ir-row">
                            <span className={irMissing ? "ir-missing" : "ir-name"}>
                                {irMissing ? "No impulse response selected" : step.irFileName}
                            </span>
                            <button type="button" disabled={disabled} onClick={onOpenIrModal}>
                                {irMissing ? "Upload IR" : "Change IR"}
                            </button>
                        </div>
                    </div>
                )}

                {step.type === "echo" && (
                    <>
                        <NumberField label="Delay" min={20} max={1000} step={10} value={params.delay_ms} unit=" ms" disabled={disabled}
                            onChange={(v) => setParam("delay_ms", v)} />
                        <NumberField label="Decay" min={0.05} max={0.95} step={0.01} value={params.decay} disabled={disabled}
                            onChange={(v) => setParam("decay", v)} />
                        <NumberField label="Repeats" min={1} max={10} step={1} value={params.repeats} disabled={disabled}
                            onChange={(v) => setParam("repeats", v)} />
                    </>
                )}

                {step.type === "delay" && (
                    <>
                        <NumberField label="Delay" min={20} max={1000} step={10} value={params.delay_ms} unit=" ms" disabled={disabled}
                            onChange={(v) => setParam("delay_ms", v)} />
                        <NumberField label="Wet mix" min={0} max={1} step={0.01} value={params.wet}
                            format={(v) => `${Math.round(v * 100)}%`} disabled={disabled}
                            onChange={(v) => setParam("wet", v)} />
                    </>
                )}

                {step.type === "reverb" && <p className="chain-field-note">Fixed room impulse response — no parameters to tune.</p>}
                {step.type === "noise" && <p className="chain-field-note">Spectral-subtraction gate — no parameters to tune.</p>}

                {step.type === "equalizer" && (
                    <>
                        <NumberField label="Low" min={0} max={10} step={1} value={params.low} disabled={disabled} onChange={(v) => setParam("low", v)} />
                        <NumberField label="Mid" min={0} max={10} step={1} value={params.mid} disabled={disabled} onChange={(v) => setParam("mid", v)} />
                        <NumberField label="High" min={0} max={10} step={1} value={params.high} disabled={disabled} onChange={(v) => setParam("high", v)} />
                    </>
                )}

                {step.type === "activity" && (
                    <NumberField label="Threshold" min={0.02} max={0.6} step={0.01} value={params.threshold_ratio}
                        format={(v) => `${Math.round(v * 100)}%`} disabled={disabled}
                        onChange={(v) => setParam("threshold_ratio", v)} />
                )}

                {step.type === "morph" && (
                    <>
                        <SelectField label="Mode" value={params.morph_mode} options={MORPH_MODES} disabled={disabled}
                            onChange={(v) => setParam("morph_mode", v)} />
                        {params.morph_mode === "pitch" && (
                            <NumberField label="Semitones" min={-12} max={12} step={1} value={params.n_steps} unit=" st" disabled={disabled}
                                onChange={(v) => setParam("n_steps", v)} />
                        )}
                        {params.morph_mode === "stretch" && (
                            <NumberField label="Rate" min={0.5} max={2} step={0.05} value={params.rate} disabled={disabled}
                                onChange={(v) => setParam("rate", v)} />
                        )}
                    </>
                )}
            </div>

            {stageResult && (
                <div className="chain-step__result">
                    <button type="button" className="chain-step__toggle" onClick={onToggleExpanded}>
                        {expanded ? "▾ Hide result" : "▸ View result"}
                    </button>

                    {expanded && (
                        <div className="chain-step__graphs">
                            {stageResult.impulseResponse && (
                                <>
                                    <StaticGraph title="Impulse Response · Time" type="time" data={stageResult.impulseResponse.waveform} color="#ff8fd6" />
                                    <StaticGraph title="Impulse Response · Frequency" type="spectrum" data={stageResult.impulseResponse.spectrum} color="#ff8fd6" />
                                </>
                            )}
                            <StaticGraph title={`${meta.label} Output · Time`} type="time" data={stageResult.waveform} color="var(--output-accent)" />
                            <StaticGraph title={`${meta.label} Output · Frequency`} type="spectrum" data={stageResult.spectrum} color="var(--output-accent)" />
                            {stageResult.filterResponse && (
                                <StaticGraph title="Filter Response |H(f)|" type="spectrum" data={stageResult.filterResponse} color="#72e6ff" />
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function CustomAudioPlayer({ src, compact = false }) {
    const audioRef = useRef(null);
    const [playing, setPlaying] = useState(false);
    const [muted, setMuted] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;
        setPlaying(false);
        setProgress(0);
        audio.pause();
        audio.currentTime = 0;
    }, [src]);

    const togglePlay = () => {
        const audio = audioRef.current;
        if (!audio) return;
        if (playing) { audio.pause(); setPlaying(false); }
        else { audio.play().then(() => setPlaying(true)).catch(() => {}); }
    };

    const toggleMute = () => {
        const audio = audioRef.current;
        if (!audio) return;
        audio.muted = !muted;
        setMuted(!muted);
    };

    const handleTimeUpdate = () => {
        const a = audioRef.current;
        if (a && a.duration) setProgress(a.currentTime / a.duration);
    };

    const handleLoadedMetadata = () => setDuration(audioRef.current?.duration || 0);
    const handleEnded = () => setPlaying(false);

    const handleSeek = (e) => {
        const audio = audioRef.current;
        if (!audio || !audio.duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        audio.currentTime = ratio * audio.duration;
        setProgress(ratio);
    };

    const fmt = (t) => {
        if (!Number.isFinite(t)) return "0:00";
        const m = Math.floor(t / 60);
        const s = Math.floor(t % 60).toString().padStart(2, "0");
        return `${m}:${s}`;
    };

    return (
        <div className={`custom-player${compact ? " custom-player--compact" : ""}`}>
            <audio
                ref={audioRef}
                src={src}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
            />
            <button type="button" className="cp-btn cp-play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
                {playing
                    ? <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><rect x="2" y="2" width="4" height="12" rx="1"/><rect x="10" y="2" width="4" height="12" rx="1"/></svg>
                    : <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M3 2l11 6-11 6z"/></svg>
                }
            </button>
            <div className="cp-track" onClick={handleSeek} role="slider" aria-label="Seek">
                <div className="cp-fill" style={{ width: `${progress * 100}%` }} />
                <div className="cp-thumb" style={{ left: `${progress * 100}%` }} />
            </div>
            {!compact && (
                <span className="cp-time">{fmt(duration * progress)} / {fmt(duration)}</span>
            )}
            <button type="button" className="cp-btn cp-mute" onClick={toggleMute} aria-label={muted ? "Unmute" : "Mute"}>
                {muted
                    ? <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M9 2L5 6H2v4h3l4 4V2zm4.5 3.5l-4 4m4-4l-4 4" strokeWidth="1.5" stroke="currentColor" strokeLinecap="round"/><line x1="11" y1="5" x2="15" y2="9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/><line x1="15" y1="5" x2="11" y2="9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                    : <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M9 2L5 6H2v4h3l4 4V2z"/><path d="M11.5 5.5a4 4 0 0 1 0 5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/><path d="M13.5 3.5a7 7 0 0 1 0 9" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                }
            </button>
        </div>
    );
}

function ConvolutionModal({ onConfirm, onCancel }) {
    const [file, setFile] = useState(null);
    const inputRef = useRef(null);

    return (
        <div className="ir-modal-backdrop" onClick={onCancel}>
            <div className="ir-modal" onClick={(e) => e.stopPropagation()}>
                <h3>Choose an impulse response</h3>
                <p>Convolution needs a second audio file — the impulse response it will be convolved with.</p>

                <input
                    ref={inputRef}
                    type="file"
                    accept="audio/*,.wav,.mp3,.ogg"
                    onChange={(event) => setFile(event.target.files?.[0] || null)}
                />

                <div className="ir-modal__actions">
                    <button type="button" className="secondary-button" onClick={onCancel}>Cancel</button>
                    <button type="button" className="process-button" disabled={!file} onClick={() => file && onConfirm(file)}>
                        Use this impulse response
                    </button>
                </div>
            </div>
        </div>
    );
}

function HistoryPanel({ history, compareIds, onToggleCompare, onRemove }) {
    if (!history.length) {
        return (
            <div className="history-empty">
                Runs you complete will appear here (up to {MAX_HISTORY}) so you can compare processed results side by side.
            </div>
        );
    }

    return (
        <div className="history-list">
            {history.map((entry) => (
                <div key={entry.id} className={`history-item ${compareIds.includes(entry.id) ? "is-selected" : ""}`}>
                    <label className="history-item__select">
                        <input
                            type="checkbox"
                            checked={compareIds.includes(entry.id)}
                            onChange={() => onToggleCompare(entry.id)}
                        />
                        <div>
                            <strong>{entry.chainLabel}</strong>
                            <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                        </div>
                    </label>
                    <div className="history-item__actions">
                        <CustomAudioPlayer src={entry.audioUrl} compact />
                        <button type="button" onClick={() => onRemove(entry.id)} aria-label="Remove from history">✕</button>
                    </div>
                </div>
            ))}
        </div>
    );
}

export default function Studio() {
    const fileInputRef = useRef(null);
    const timersRef = useRef([]);

    const [audioFile, setAudioFile] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");
    const [fileMeta, setFileMeta] = useState(null); // { duration, sampleRate, channels }

    const [chain, setChain] = useState([]);
    const [stepStatus, setStepStatus] = useState({});
    const [expandedSteps, setExpandedSteps] = useState({});
    const [convModalStepId, setConvModalStepId] = useState(null);

    const [running, setRunning] = useState(false);
    const [runError, setRunError] = useState("");
    const [result, setResult] = useState(null);

    const [history, setHistory] = useState(loadHistory);
    const [compareIds, setCompareIds] = useState([]);

    useEffect(() => () => {
        timersRef.current.forEach((t) => window.clearTimeout(t));
        if (audioUrl) URL.revokeObjectURL(audioUrl);
    }, [audioUrl]);

    const handleFile = async (file) => {
        if (!file) return;

        if (audioUrl) URL.revokeObjectURL(audioUrl);
        const url = URL.createObjectURL(file);

        setAudioFile(file);
        setAudioUrl(url);
        setResult(null);
        setRunError("");

        try {
            const context = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await file.arrayBuffer();
            const decoded = await context.decodeAudioData(arrayBuffer);
            setFileMeta({
                duration: decoded.duration,
                sampleRate: decoded.sampleRate,
                channels: decoded.numberOfChannels,
            });
            context.close();
        } catch {
            setFileMeta(null);
        }
    };

    const handleFileChange = (event) => {
        handleFile(event.target.files?.[0]);
        event.target.value = "";
    };

    const addOperation = (type) => {
        if (chain.length >= MAX_CHAIN_STEPS || running) return;

        const meta = OPERATION_BY_TYPE[type];
        const id = makeId();
        const newStep = { id, type, params: { ...meta.defaultParams }, irFile: null, irFileName: "" };

        setChain((current) => [...current, newStep]);
        setResult(null);

        if (meta.requiresIr) {
            setConvModalStepId(id);
        }
    };

    const updateStepParams = (id, params) => {
        setChain((current) => current.map((s) => (s.id === id ? { ...s, params } : s)));
        setResult(null);
    };

    const removeStep = (id) => {
        setChain((current) => current.filter((s) => s.id !== id));
        setResult(null);
    };

    const moveStep = (id, direction) => {
        setChain((current) => {
            const index = current.findIndex((s) => s.id === id);
            const target = index + direction;
            if (index < 0 || target < 0 || target >= current.length) return current;
            const next = [...current];
            [next[index], next[target]] = [next[target], next[index]];
            return next;
        });
        setResult(null);
    };

    const setStepIr = (id, file) => {
        setChain((current) => current.map((s) => (s.id === id ? { ...s, irFile: file, irFileName: file.name } : s)));
        setResult(null);
    };

    const toggleExpanded = (id) => {
        setExpandedSteps((current) => ({ ...current, [id]: !current[id] }));
    };

    const toggleCompare = (id) => {
        setCompareIds((current) => (current.includes(id) ? current.filter((c) => c !== id) : [...current, id]));
    };

    const removeHistoryEntry = (id) => {
        setHistory((current) => {
            const next = current.filter((e) => e.id !== id);
            saveHistory(next);
            return next;
        });
        setCompareIds((current) => current.filter((c) => c !== id));
    };

    const missingIrSteps = chain.filter((s) => s.type === "convolution" && !s.irFile);
    const canRun = audioFile && chain.length > 0 && missingIrSteps.length === 0 && !running;

    const runChain = async () => {
        if (!canRun) {
            if (missingIrSteps.length > 0) {
                setRunError("Every Convolution step needs an impulse response file before you can run the chain.");
            } else if (chain.length === 0) {
                setRunError("Add at least one operation to the chain first.");
            }
            return;
        }

        setRunError("");
        setRunning(true);
        setResult(null);
        setExpandedSteps({});

        const orderSnapshot = chain.map((s) => s.id);
        const initialStatus = Object.fromEntries(orderSnapshot.map((id) => [id, "pending"]));
        setStepStatus(initialStatus);

        try {
            const payload = chain.map((s) => ({
                type: s.type,
                params: s.type === "convolution" ? {} : s.params,
            }));
            const irFiles = chain.filter((s) => s.type === "convolution").map((s) => s.irFile);

            const data = await processChain(audioFile, payload, irFiles);
            setResult(data);

            // Reveal each step's green status in order, timed to that step's
            // real measured duration (clamped so short steps still read as a
            // visible, distinct tick rather than an instant flash).
            let cumulative = 0;
            data.stages.forEach((stage, i) => {
                const delay = clamp(stage.elapsed_ms, 150, 900);
                cumulative += delay;
                const timer = window.setTimeout(() => {
                    setStepStatus((current) => ({ ...current, [orderSnapshot[i]]: "done" }));
                    if (i === data.stages.length - 1) {
                        setRunning(false);

                        const entry = {
                            id: makeId(),
                            timestamp: Date.now(),
                            chainLabel: chain.map((s) => OPERATION_BY_TYPE[s.type].short).join(" → "),
                            audioUrl: data.output.audio_url,
                            waveform: data.output.waveform,
                            spectrum: data.output.spectrum,
                            stats: data.output.stats,
                        };
                        setHistory((current) => {
                            const next = [entry, ...current].slice(0, MAX_HISTORY);
                            saveHistory(next);
                            return next;
                        });
                    }
                }, cumulative);
                timersRef.current.push(timer);
            });
        } catch (error) {
            console.error(error);
            setRunError(error.message || "Chain processing failed.");
            setStepStatus(Object.fromEntries(orderSnapshot.map((id) => [id, "error"])));
            setRunning(false);
        }
    };

    const compareEntries = history.filter((e) => compareIds.includes(e.id));

    return (
        <main className="studio-page">
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.m4a"
                className="studio-file-input"
                onChange={handleFileChange}
            />

            <div className="studio-shell">
                <header className="studio-header">
                    <div>
                        <div className="studio-eyebrow">DSP / STUDIO</div>
                        <h1>Multi-Operation Signal Studio</h1>
                        <p>Chain any of the course's DSP effects in the order you choose, then compare input and output.</p>
                    </div>
                    <button type="button" className="back-button" onClick={() => { window.location.hash = ""; }}>← Back</button>
                </header>

                <section className="audio-panel has-audio">
                    <div className="panel-label">AUDIO INPUT</div>
                    {!audioFile ? (
                        <button type="button" className="upload-zone" onClick={() => fileInputRef.current?.click()}>
                            <div className="upload-icon">＋</div>
                            <strong>Drop an audio file here</strong>
                            <span>or click anywhere to upload WAV, MP3, OGG or another browser-supported format</span>
                        </button>
                    ) : (
                        <>
                            <div className="audio-file-row">
                                <div>
                                    <strong>{audioFile.name}</strong>
                                    <span>
                                        {fileMeta
                                            ? `${fileMeta.channels} ch · ${Math.round(fileMeta.sampleRate)} Hz · ${fileMeta.duration.toFixed(1)}s`
                                            : "Loading metadata…"}
                                    </span>
                                </div>
                                <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()}>Replace</button>
                            </div>
                            <CustomAudioPlayer src={audioUrl} />
                        </>
                    )}
                </section>

                <section className="chain-builder">
                    <div className="panel-label">OPERATION CHAIN</div>
                    <p className="chain-builder__hint">
                        Add operations, arrange their order, then run the chain. Each step's output feeds the next.
                        {chain.length >= MAX_CHAIN_STEPS && ` (limit of ${MAX_CHAIN_STEPS} steps reached)`}
                    </p>

                    <div className="operation-palette">
                        {OPERATIONS.map((op) => (
                            <button
                                key={op.type}
                                type="button"
                                disabled={running || chain.length >= MAX_CHAIN_STEPS}
                                onClick={() => addOperation(op.type)}
                            >
                                + {op.label}
                            </button>
                        ))}
                    </div>

                    {chain.length === 0 ? (
                        <div className="chain-empty">No operations yet — add one above to get started.</div>
                    ) : (
                        <div className="chain-list">
                            {chain.map((step, index) => (
                                <ChainStepCard
                                    key={step.id}
                                    step={step}
                                    index={index}
                                    total={chain.length}
                                    status={stepStatus[step.id] || "idle"}
                                    stageResult={result?.stages?.[index]}
                                    sampleRate={fileMeta?.sampleRate}
                                    expanded={!!expandedSteps[step.id]}
                                    onToggleExpanded={() => toggleExpanded(step.id)}
                                    onMoveUp={() => moveStep(step.id, -1)}
                                    onMoveDown={() => moveStep(step.id, 1)}
                                    onRemove={() => removeStep(step.id)}
                                    onParamsChange={(params) => updateStepParams(step.id, params)}
                                    onOpenIrModal={() => setConvModalStepId(step.id)}
                                    disabled={running}
                                />
                            ))}
                        </div>
                    )}

                    {runError && <div className="chain-error">{runError}</div>}

                    <button type="button" className="process-button chain-run-button" disabled={!canRun} onClick={runChain}>
                        {running ? "Processing…" : "Run Chain"}
                    </button>
                </section>

                {result && (
                    <section className="analysis-section always-visible">
                        <div className="analysis-heading">
                            <div>
                                <div className="studio-eyebrow">FINAL RESULT</div>
                                <h2>Input vs. Output</h2>
                            </div>
                            <div className="analysis-status">
                                <span className="status-dot active" />
                                {chain.length} step{chain.length !== 1 ? "s" : ""} applied
                            </div>
                        </div>

                        <div className="analysis-grid">
                            <StaticGraph title="INPUT · Time Domain" type="time" data={result.input.waveform} color="var(--input-accent)" />
                            <StaticGraph title="OUTPUT · Time Domain" type="time" data={result.output.waveform} color="var(--output-accent)" />
                            <StaticGraph title="INPUT · Frequency Domain" type="spectrum" data={result.input.spectrum} color="var(--input-accent)" />
                            <StaticGraph title="OUTPUT · Frequency Domain" type="spectrum" data={result.output.spectrum} color="var(--output-accent)" />
                        </div>

                        <div className="analysis-footer">
                            <span>Peak output <strong>{Math.round(result.output.stats.peak * 100)}%</strong></span>
                            <span>RMS output <strong>{Number.isFinite(result.output.stats.db) ? `${result.output.stats.db.toFixed(1)} dB` : "—"}</strong></span>
                            <CustomAudioPlayer src={result.output.audio_url} />
                        </div>
                    </section>
                )}

                <section className="history-section">
                    <div className="panel-label">RESULT HISTORY ({history.length}/{MAX_HISTORY})</div>
                    <HistoryPanel history={history} compareIds={compareIds} onToggleCompare={toggleCompare} onRemove={removeHistoryEntry} />

                    {compareEntries.length > 0 && (
                        <div className="compare-grid">
                            {compareEntries.map((entry) => (
                                <div key={entry.id} className="compare-item">
                                    <strong>{entry.chainLabel}</strong>
                                    <StaticGraph title="Time Domain" type="time" data={entry.waveform} color="var(--output-accent)" height={110} />
                                    <StaticGraph title="Frequency Domain" type="spectrum" data={entry.spectrum} color="var(--output-accent)" height={110} />
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>

            {convModalStepId && (
                <ConvolutionModal
                    onCancel={() => setConvModalStepId(null)}
                    onConfirm={(file) => {
                        setStepIr(convModalStepId, file);
                        setConvModalStepId(null);
                    }}
                />
            )}
        </main>
    );
}
