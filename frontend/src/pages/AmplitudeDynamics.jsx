import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { processAudio } from "../services/api";
import "../styles/amplitude-dynamics.css";
import FrequencySpectrumAnimator from "../components/Visualizations/FrequencySpectrumAnimator";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const INPUT_COLOR = "#4ea1ff";
const PROCESS_COLOR = "#b66cff";
const OUTPUT_COLOR = "#ff5d6c";

const GAIN_MIN = 0.01;
const GAIN_MAX = 4;
const GAIN_PRESETS = [0.5, 1, 2, 3, 4];

// The loudest point of the input sits at 80% of the +/-1 axis, so at 1x gain
// the line has headroom, and gains above 1.25x visibly push past +/-1 (the
// axis then widens to +/-1.5, +/-2 ... so nothing is ever cut off).
const DISPLAY_PEAK = 0.8;
const AXIS_STEPS = [1, 1.5, 2, 3, 4];
const DISPLAY_POINTS = 240;

// Padding around the plot area (room for the axis labels).
const PAD = { left: 44, right: 20, top: 18, bottom: 32 };

// Animation timings
const INPUT_REVEAL_MS = 4200;   // input wave slowly drawing itself after upload
const OUTPUT_REVEAL_MS = 3000;  // output wave drawing itself after processing
const HANDOFF_MS = 450;         // draw-head dot fades into the playhead dot
const PROCESS_TOTAL_MS = 9000;  // whole load -> multiply -> write animation

// Share of PROCESS_TOTAL_MS that each phase takes.
const PHASE_SPANS = {
    load: [0, 0.24],
    multiply: [0.24, 0.82],
    write: [0.82, 1],
};

const PHASE_STEPS = [
    { key: "load", label: "LOAD", hint: "Read x[n]" },
    { key: "multiply", label: "MULTIPLY", hint: "Scale by gain" },
    { key: "write", label: "WRITE", hint: "Store y[n]" },
];

/* ------------------------------------------------------------------ */
/*  Small helpers                                                      */
/* ------------------------------------------------------------------ */

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function easeInOut(t) {
    return 0.5 - 0.5 * Math.cos(Math.PI * clamp(t, 0, 1));
}

function easeOut(t) {
    const k = 1 - clamp(t, 0, 1);
    return 1 - k * k * k;
}

function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const minutes = Math.floor(Math.max(0, seconds) / 60);
    const secs = Math.floor(Math.max(0, seconds) % 60).toString().padStart(2, "0");
    return `${minutes}:${secs}`;
}

function formatAxisTime(seconds) {
    return `${seconds.toFixed(1)}s`;
}

function formatAxisValue(value) {
    if (value === 0) return "0";
    const abs = Math.abs(value);
    const text = Number.isInteger(abs) ? String(abs) : abs.toFixed(1);
    return `${value > 0 ? "+" : "−"}${text}`;
}

function formatSigned(value) {
    if (value === null || value === undefined) return "—";
    return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}

function formatDb(factor) {
    const db = 20 * Math.log10(Math.max(factor, 0.0001));
    return `${db >= 0 ? "+" : "−"}${Math.abs(db).toFixed(1)} dB`;
}

function formatBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatMeta(meta) {
    if (!meta) return "—";
    return `${meta.channels} ch · ${meta.sampleRate} Hz`;
}

function peakOf(data) {
    if (!data?.length) return 0;
    let peak = 0;
    for (let i = 0; i < data.length; i += 1) {
        const value = Math.abs(data[i]);
        if (value > peak) peak = value;
    }
    return peak;
}

// Smallest "nice" axis half-range (+/-R) that fits the given peak.
function pickRange(peak) {
    const safe = Number.isFinite(peak) ? peak : 1;
    for (const step of AXIS_STEPS) {
        if (safe <= step + 1e-6) return step;
    }
    return Math.ceil(safe);
}

function phaseFromProgress(progress) {
    if (progress < PHASE_SPANS.load[1]) return "load";
    if (progress < PHASE_SPANS.multiply[1]) return "multiply";
    return "write";
}

// 0..1 progress *inside* the current phase.
function phaseLocal(phase, progress) {
    if (phase === "complete") return 1;
    const span = PHASE_SPANS[phase];
    if (!span) return 0;
    return clamp((progress - span[0]) / (span[1] - span[0]), 0, 1);
}

/* ------------------------------------------------------------------ */
/*  Audio -> display waveform                                          */
/* ------------------------------------------------------------------ */

// Reduces the raw samples to a small, smooth set of points for drawing.
// Values are NOT normalised here: input and output are later scaled by the
// same reference peak so the gain difference stays visible.
function makeDisplayWaveform(samples, targetPoints = DISPLAY_POINTS) {
    if (!samples?.length) return new Float32Array(0);

    const count = Math.min(targetPoints, samples.length);
    const stride = samples.length / count;
    const reduced = new Float32Array(count);

    for (let i = 0; i < count; i += 1) {
        const start = Math.floor(i * stride);
        const end = Math.max(start + 1, Math.min(samples.length, Math.floor((i + 1) * stride)));
        let peakIndex = start;
        let peakAbs = 0;

        for (let j = start; j < end; j += 1) {
            const abs = Math.abs(samples[j]);
            if (abs > peakAbs) {
                peakAbs = abs;
                peakIndex = j;
            }
        }

        const peak = samples[peakIndex] || 0;
        let localMean = 0;
        for (let j = start; j < end; j += 1) localMean += samples[j];
        localMean /= Math.max(1, end - start);

        // Preserve the strongest local excursion but use the local mean to
        // choose its sign. This avoids long recordings collapsing to zero.
        reduced[i] = localMean < 0 ? -Math.abs(peak) : Math.abs(peak);
    }

    const smoothed = new Float32Array(count);
    const radius = Math.max(1, Math.min(3, Math.floor(count / 80)));

    for (let i = 0; i < count; i += 1) {
        let sum = 0;
        let weight = 0;
        for (let j = Math.max(0, i - radius); j <= Math.min(count - 1, i + radius); j += 1) {
            const w = radius + 1 - Math.abs(i - j);
            sum += reduced[j] * w;
            weight += w;
        }
        smoothed[i] = weight ? sum / weight : 0;
    }

    return smoothed;
}
// referencePeak: pass the INPUT's raw peak when decoding the OUTPUT so both
// signals share one scale (that is what makes the amplification visible).
async function decodeAudio(source, referencePeak = null) {
    const response = await fetch(source);
    if (!response.ok) throw new Error("Could not read audio");
    const buffer = await response.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();

    try {
        const decoded = await context.decodeAudioData(buffer.slice(0));
        const raw = makeDisplayWaveform(decoded.getChannelData(0));
        const peak = Math.max(peakOf(raw), 1e-9);
        const scale = DISPLAY_PEAK / (referencePeak || peak);

        return {
            data: Array.from(raw, (value) => value * scale),
            peak,
            duration: decoded.duration,
            channels: decoded.numberOfChannels,
            sampleRate: decoded.sampleRate,
            length: decoded.length,
        };
    } finally {
        await context.close();
    }
}

/* ------------------------------------------------------------------ */
/*  Canvas drawing toolkit                                             */
/* ------------------------------------------------------------------ */

function getGeometry(width, height) {
    const left = PAD.left;
    const right = width - PAD.right;
    const top = PAD.top;
    const bottom = height - PAD.bottom;

    return {
        width,
        height,
        left,
        right,
        top,
        bottom,
        plotW: Math.max(1, right - left),
        plotH: Math.max(1, bottom - top),
        centerY: (top + bottom) / 2,
    };
}

function toPoints(data, geo, range, gain = 1) {
    if (!data?.length) return [];

    const half = geo.plotH / 2;
    const last = Math.max(1, data.length - 1);
    const points = new Array(data.length);

    for (let i = 0; i < data.length; i += 1) {
        const level = clamp((data[i] * gain) / range, -1, 1);
        points[i] = {
            x: geo.left + geo.plotW * (i / last),
            y: geo.centerY - level * half,
        };
    }

    return points;
}

// Point at fractional position t (0..1) along the curve.
function pointAt(points, t) {
    if (!points.length) return null;
    const f = clamp(t, 0, 1) * (points.length - 1);
    const i = Math.floor(f);
    const j = Math.min(points.length - 1, i + 1);
    const frac = f - i;

    return {
        x: points[i].x + (points[j].x - points[i].x) * frac,
        y: points[i].y + (points[j].y - points[i].y) * frac,
    };
}

function line(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function roundedRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
}

// Smooth line. clipX reveals only the part of the curve left of that x, which
// is what makes the "drawing itself" animation exact and flicker-free.
function drawCurve(ctx, points, { color, width = 2.2, glow = 8, alpha = 1, clipX = null }) {
    if (points.length < 2 || alpha <= 0) return;

    let end = points.length - 1;
    if (clipX !== null) {
        while (end > 1 && points[end - 1].x > clipX + 4) end -= 1;
    }

    ctx.save();

    if (clipX !== null) {
        ctx.beginPath();
        ctx.rect(0, -10, Math.max(0, clipX), 100000);
        ctx.clip();
    }

    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    if (glow > 0) {
        ctx.shadowColor = color;
        ctx.shadowBlur = glow;
    }

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 0; i < end; i += 1) {
        const current = points[i];
        const next = points[i + 1];
        const previous = points[Math.max(0, i - 1)];
        const following = points[Math.min(points.length - 1, i + 2)];
        const tension = 0.16;

        ctx.bezierCurveTo(
            current.x + (next.x - previous.x) * tension,
            current.y + (next.y - previous.y) * tension,
            next.x - (following.x - current.x) * tension,
            next.y - (following.y - current.y) * tension,
            next.x,
            next.y
        );
    }

    ctx.stroke();
    ctx.restore();
}

function drawPointer(ctx, point, color, radius = 6, alpha = 1) {
    if (!point || alpha <= 0) return;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowColor = color;
    ctx.shadowBlur = 16;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius + 1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}

function drawScanLine(ctx, geo, x, color, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 5]);
    line(ctx, x, geo.top, x, geo.bottom);
    ctx.restore();
}

function drawScanBand(ctx, geo, x) {
    const width = Math.min(90, x - geo.left);
    if (width <= 0) return;

    const gradient = ctx.createLinearGradient(x - width, 0, x, 0);
    gradient.addColorStop(0, "rgba(182,108,255,0)");
    gradient.addColorStop(1, "rgba(182,108,255,.13)");
    ctx.save();
    ctx.fillStyle = gradient;
    ctx.fillRect(x - width, geo.top, width, geo.plotH);
    ctx.restore();
}

function drawPill(ctx, text, x, y, font, color) {
    ctx.save();
    ctx.font = `700 11px ${font}`;
    const width = ctx.measureText(text).width + 18;
    const height = 20;

    roundedRect(ctx, x - width / 2, y - height / 2, width, height, 10);
    ctx.fillStyle = "rgba(30,25,80,.92)";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y + 0.5);
    ctx.restore();
}

// A short trail of individual samples right behind the scan head: each blue
// dot (x[n]) is joined to the purple dot it becomes (gain * x[n]).
function drawSampleStems(ctx, inputPoints, outputPoints, t) {
    const count = inputPoints.length;
    if (count < 4) return;

    const stride = Math.max(2, Math.round(count / 34));
    const last = Math.floor((t * (count - 1)) / stride) * stride;
    const trail = 9;

    ctx.save();
    for (let k = 0; k < trail; k += 1) {
        const index = last - k * stride;
        if (index < 0) break;

        const a = inputPoints[index];
        const b = outputPoints[index];
        ctx.globalAlpha = 0.9 * (1 - k / trail);

        ctx.strokeStyle = PROCESS_COLOR;
        ctx.lineWidth = 1;
        line(ctx, a.x, a.y, b.x, b.y);

        ctx.fillStyle = INPUT_COLOR;
        ctx.beginPath();
        ctx.arc(a.x, a.y, 2.6, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = PROCESS_COLOR;
        ctx.beginPath();
        ctx.arc(b.x, b.y, 3.2, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.restore();
}

// Axes, grid, +/- amplitude labels and time labels (the reference look).
function drawFrame(ctx, geo, { range, duration, font }) {
    const { left, right, top, bottom, centerY, plotW, plotH } = geo;

    ctx.save();

    ctx.fillStyle = "rgba(255,255,255,.022)";
    ctx.fillRect(left, top, plotW, plotH);

    ctx.lineWidth = 1;
    [-1, -0.5, 0, 0.5, 1].forEach((level) => {
        const y = Math.round(centerY - level * (plotH / 2)) + 0.5;
        ctx.strokeStyle = level === 0 ? "rgba(255,255,255,.34)" : "rgba(255,255,255,.085)";
        line(ctx, left, y, right, y);
    });

    const ticks = plotW < 260 ? 3 : 5;
    ctx.font = `500 10px ${font}`;
    ctx.fillStyle = "rgba(255,255,255,.55)";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (let i = 0; i <= ticks; i += 1) {
        const x = Math.round(left + (plotW * i) / ticks) + 0.5;
        if (i > 0) {
            ctx.strokeStyle = "rgba(255,255,255,.085)";
            line(ctx, x, top, x, bottom);
        }
        if (duration > 0) {
            ctx.fillText(formatAxisTime((duration * i) / ticks), x, bottom + 11);
        }
    }

    ctx.strokeStyle = "rgba(255,255,255,.42)";
    line(ctx, left + 0.5, top, left + 0.5, bottom);

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    [[top, range], [centerY, 0], [bottom, -range]].forEach(([y, value]) => {
        ctx.fillText(formatAxisValue(value), left - 9, y);
    });

    ctx.restore();
}

function drawEmptyMessage(ctx, geo, text, font) {
    if (!text) return;

    ctx.save();
    ctx.font = `600 12px ${font}`;
    const width = ctx.measureText(text).width + 28;
    const cx = geo.left + geo.plotW / 2;

    roundedRect(ctx, cx - width / 2, geo.centerY - 15, width, 30, 15);
    ctx.fillStyle = "rgba(24,20,66,.92)";
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,.62)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cx, geo.centerY + 0.5);
    ctx.restore();
}

/* ------------------------------------------------------------------ */
/*  useCanvas: sizes a canvas to its container (HiDPI aware)           */
/* ------------------------------------------------------------------ */

function useCanvas(drawRef) {
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const fontRef = useRef("sans-serif");

    const redraw = useCallback(() => {
        const box = containerRef.current;
        const canvas = canvasRef.current;
        if (!box || !canvas) return;

        const width = box.clientWidth;
        const height = box.clientHeight;
        if (!width || !height) return;

        const dpr = window.devicePixelRatio || 1;
        const pixelW = Math.round(width * dpr);
        const pixelH = Math.round(height * dpr);

        if (canvas.width !== pixelW || canvas.height !== pixelH) {
            canvas.width = pixelW;
            canvas.height = pixelH;
            fontRef.current = getComputedStyle(box).fontFamily || "sans-serif";
        }

        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);
        drawRef.current?.(ctx, width, height, fontRef.current);
    }, [drawRef]);

    useLayoutEffect(() => {
        const box = containerRef.current;
        if (!box || typeof ResizeObserver === "undefined") return undefined;

        const observer = new ResizeObserver(() => redraw());
        observer.observe(box);
        return () => observer.disconnect();
    }, [redraw]);

    return { containerRef, canvasRef, redraw };
}

/* ------------------------------------------------------------------ */
/*  SignalPlot: axes + one or more waveforms + playhead + reveal       */
/* ------------------------------------------------------------------ */

function SignalPlot({
    layers,
    range = 1,
    duration = 0,
    reveal = 0,
    revealKey = null,
    onSeek,
    disabled = false,
    label,
    emptyText,
    title,
    meta,
}) {
    const drawRef = useRef(null);
    const revealRef = useRef({ draw: 1, hand: 1 });
    const hoverRef = useRef(null);
    const { containerRef, canvasRef, redraw } = useCanvas(drawRef);

    const hasData = layers.some((layer) => layer.data?.length);
    const interactive = Boolean(onSeek) && !disabled && hasData;

    drawRef.current = (ctx, width, height, font) => {
        const geo = getGeometry(width, height);
        drawFrame(ctx, geo, { range, duration, font });

        if (!hasData) {
            drawEmptyMessage(ctx, geo, emptyText, font);
            return;
        }

        const { draw, hand } = revealRef.current;
        const revealing = draw < 1;

        layers.forEach((layer) => {
            if (!layer.data?.length) return;

            const points = toPoints(layer.data, geo, range);
            drawCurve(ctx, points, {
                color: layer.color,
                width: layer.width ?? 2.2,
                glow: 9,
                clipX: revealing ? geo.left + geo.plotW * draw : null,
            });

            // Head of the line while it is being drawn; fades into the playhead.
            const headAlpha = revealing ? 1 : 1 - hand;
            if (headAlpha > 0.01) {
                const head = pointAt(points, revealing ? draw : 1);
                drawScanLine(ctx, geo, head.x, layer.color, 0.4 * headAlpha);
                drawPointer(ctx, head, layer.color, 6, headAlpha);
            }

            if (Number.isFinite(layer.marker)) {
                const alpha = revealing ? 0 : hand;
                if (alpha > 0.01) {
                    const playhead = pointAt(points, layer.marker);
                    drawScanLine(ctx, geo, playhead.x, "#ffffff", 0.3 * alpha);
                    drawPointer(ctx, playhead, layer.color, 6, alpha);
                }
            }
        });

        const hover = hoverRef.current;
        if (hover !== null && interactive && !revealing) {
            const x = geo.left + geo.plotW * hover;
            drawScanLine(ctx, geo, x, "#ffffff", 0.24);

            if (duration > 0) {
                ctx.save();
                ctx.font = `600 10px ${font}`;
                ctx.fillStyle = "rgba(255,255,255,.8)";
                ctx.textBaseline = "top";
                ctx.textAlign = hover > 0.5 ? "right" : "left";
                ctx.fillText(formatAxisTime(hover * duration), x + (hover > 0.5 ? -6 : 6), geo.top + 5);
                ctx.restore();
            }
        }
    };

    // Slow "draw-on" reveal whenever a new signal arrives.
    useLayoutEffect(() => {
        if (!reveal || !revealKey?.length) {
            revealRef.current = { draw: 1, hand: 1 };
            redraw();
            return undefined;
        }

        revealRef.current = { draw: 0, hand: 0 };
        redraw();

        const start = performance.now();
        let frame = 0;

        const tick = (now) => {
            const elapsed = now - start;
            revealRef.current = {
                draw: easeInOut(elapsed / reveal),
                hand: clamp((elapsed - reveal) / HANDOFF_MS, 0, 1),
            };
            redraw();
            if (elapsed < reveal + HANDOFF_MS) frame = requestAnimationFrame(tick);
        };

        frame = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(frame);
    }, [revealKey, reveal, redraw]);

    useLayoutEffect(() => {
        redraw();
    }, [layers, range, duration, disabled, emptyText, redraw]);

    const readPosition = (event) => {
        const rect = containerRef.current.getBoundingClientRect();
        const plotWidth = Math.max(1, rect.width - PAD.left - PAD.right);
        return {
            ratio: clamp((event.clientX - rect.left - PAD.left) / plotWidth, 0, 1),
            yRatio: (event.clientY - rect.top) / Math.max(1, rect.height),
        };
    };

    const handleClick = (event) => {
        if (!interactive) return;
        const { ratio, yRatio } = readPosition(event);
        onSeek(ratio, yRatio);
    };

    const handleMove = (event) => {
        if (!interactive) return;
        hoverRef.current = readPosition(event).ratio;
        redraw();
    };

    const handleLeave = () => {
        if (hoverRef.current === null) return;
        hoverRef.current = null;
        redraw();
    };

    return (
        <div className="plot-card">
            {(title || meta) && (
                <div className="plot-strip">
                    <span className="plot-strip-title">{title}</span>
                    <span className="plot-strip-meta">{meta}</span>
                </div>
            )}
            <div
                ref={containerRef}
                className={`plot-body ${interactive ? "is-seekable" : ""}`}
                onClick={handleClick}
                onMouseMove={handleMove}
                onMouseLeave={handleLeave}
                role="button"
                tabIndex={interactive ? 0 : -1}
                aria-label={`${label} waveform. Click to seek.`}
                aria-disabled={!interactive}
            >
                <canvas ref={canvasRef} />
            </div>
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  Transport: play / pause + seek bar under a plot                    */
/* ------------------------------------------------------------------ */

function Transport({ tone, playing, current, total, disabled, onToggle, onScrub }) {
    const ratio = total ? clamp(current / total, 0, 1) : 0;

    return (
        <div className={`transport is-${tone}`}>
            <button
                type="button"
                className="transport-play"
                onClick={onToggle}
                disabled={disabled}
                aria-label={playing ? "Pause" : "Play"}
            >
                {playing ? (
                    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                        <rect x="2" y="1.5" width="3" height="9" rx="1" fill="currentColor" />
                        <rect x="7" y="1.5" width="3" height="9" rx="1" fill="currentColor" />
                    </svg>
                ) : (
                    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                        <path d="M3 1.6v8.8a.6.6 0 0 0 .9.5l7-4.4a.6.6 0 0 0 0-1L3.9 1.1a.6.6 0 0 0-.9.5Z" fill="currentColor" />
                    </svg>
                )}
            </button>
            <input
                className="range-input transport-slider"
                type="range"
                min="0"
                max="1000"
                step="1"
                value={Math.round(ratio * 1000)}
                style={{ "--pos": ratio }}
                onChange={(event) => onScrub(Number(event.target.value) / 1000)}
                disabled={disabled}
                aria-label="Seek"
            />
            <span className="transport-time">{formatTime(current)} / {formatTime(total)}</span>
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  ProcessingPanel: load -> multiply -> write animation               */
/* ------------------------------------------------------------------ */

function ProcessingPanel({
    data,
    duration,
    sampleCount,
    factor,
    runFactor,
    progress,
    phase,
    processing,
    phaseLabel,
}) {
    const drawRef = useRef(null);
    const { containerRef, canvasRef, redraw } = useCanvas(drawRef);
    const [loopProgress, setLoopProgress] = useState(0);

    // Once the backend run has completed, keep the center visualization alive.
    // The modal can open over it, and after the modal closes the transformation
    // continues cycling instead of becoming a static "COMPLETE" card.
    useEffect(() => {
        if (phase !== "complete") {
            setLoopProgress(0);
            return undefined;
        }

        let raf;
        const start = performance.now();
        const duration = 7600;

        const tick = (now) => {
            const p = ((now - start) % duration) / duration;
            setLoopProgress(p);
            raf = requestAnimationFrame(tick);
        };

        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [phase]);

    // Before the first run we preview the live slider value; after completion
    // the center panel uses its own repeating animation clock.
    const activeFactor = phase === "idle" ? factor : runFactor;
    const displayProgress = phase === "complete" ? loopProgress : progress;
    const displayPhase = phase === "complete" ? phaseFromProgress(loopProgress) : phase;
    const range = pickRange(DISPLAY_PEAK * activeFactor);
    const local = phaseLocal(displayPhase, displayProgress);
    const count = data?.length || 0;

    drawRef.current = (ctx, width, height, font) => {
        const geo = getGeometry(width, height);
        drawFrame(ctx, geo, { range, duration, font });

        if (!count) {
            drawEmptyMessage(ctx, geo, "Upload a WAV to begin", font);
            return;
        }

        if (displayPhase === "idle") {
            drawEmptyMessage(ctx, geo, "Press run to load the samples", font);
            return;
        }

        const inputPoints = toPoints(data, geo, range, 1);
        const outputPoints = toPoints(data, geo, range, activeFactor);
        const headX = geo.left + geo.plotW * local;

        // 1) LOAD: the input wave draws itself, exactly like in the input box.
        if (displayPhase === "load") {
            drawCurve(ctx, inputPoints, { color: INPUT_COLOR, width: 2.3, glow: 9, clipX: headX });
            const head = pointAt(inputPoints, local);
            drawScanLine(ctx, geo, head.x, INPUT_COLOR, 0.4);
            drawPointer(ctx, head, INPUT_COLOR, 6);
            return;
        }

        // The input stays behind, dimmed, so the change is easy to see.
        const dim = displayPhase === "multiply" ? easeOut(local / 0.08) : 1;
        drawCurve(ctx, inputPoints, {
            color: INPUT_COLOR,
            width: lerp(2.3, 1.5, dim),
            glow: 9 * (1 - dim),
            alpha: lerp(1, 0.36, dim),
        });

        // 2) MULTIPLY: scan again from the first sample; each point is scaled.
        if (displayPhase === "multiply") {
            drawScanBand(ctx, geo, headX);
            drawCurve(ctx, outputPoints, { color: PROCESS_COLOR, width: 2.6, glow: 10, clipX: headX });
            drawSampleStems(ctx, inputPoints, outputPoints, local);

            const from = pointAt(inputPoints, local);
            const to = pointAt(outputPoints, local);

            drawScanLine(ctx, geo, from.x, PROCESS_COLOR, 0.35);
            ctx.save();
            ctx.strokeStyle = PROCESS_COLOR;
            ctx.lineWidth = 1.5;
            line(ctx, from.x, from.y, to.x, to.y);
            ctx.restore();

            drawPointer(ctx, from, INPUT_COLOR, 4);
            drawPointer(ctx, to, PROCESS_COLOR, 7);
            drawPill(
                ctx,
                `× ${activeFactor.toFixed(2)}`,
                clamp(from.x, geo.left + 34, geo.right - 34),
                geo.top + 14,
                font,
                PROCESS_COLOR
            );
            return;
        }

        // 3) WRITE / COMPLETE: the scaled wave is committed to the output (red).
        const written = displayPhase === "write" ? local : 1;

        if (written < 1) {
            drawCurve(ctx, outputPoints, {
                color: PROCESS_COLOR,
                width: 2.2,
                glow: 0,
                alpha: lerp(1, 0.5, easeOut(written / 0.1)),
            });
        }

        drawCurve(ctx, outputPoints, {
            color: OUTPUT_COLOR,
            width: 2.8,
            glow: 11,
            clipX: written < 1 ? headX : null,
        });

        if (written < 1) {
            const head = pointAt(outputPoints, written);
            drawScanLine(ctx, geo, head.x, OUTPUT_COLOR, 0.45);
            drawPointer(ctx, head, OUTPUT_COLOR, 7);
        }
    };

    useLayoutEffect(() => {
        redraw();
    }, [data, activeFactor, displayProgress, displayPhase, duration, redraw]);

    // Live sample readout under the plot.
    const index = count ? Math.round(local * (count - 1)) : 0;
    const inputSample = count ? data[index] : null;
    const showInput = displayPhase === "load" || displayPhase === "multiply";
    const showOutput = displayPhase === "multiply" || displayPhase === "write";
    const xValue = showInput ? inputSample : null;
    const yValue = showOutput && inputSample !== null ? inputSample * activeFactor : null;
    const sampleNumber =
        count > 1 && (showInput || showOutput)
            ? Math.round((index / (count - 1)) * Math.max(0, (sampleCount || count) - 1))
            : null;

    const currentStep = PHASE_STEPS.findIndex((step) => step.key === displayPhase);

    const statusText = (() => {
        if (!count) return "Upload a WAV to begin.";
        if (phase === "complete") return "Backend result complete — replaying the transformation continuously.";
        if (displayPhase === "load") return "Reading the input samples from left to right…";
        if (displayPhase === "multiply") return `Multiplying every sample by ${activeFactor.toFixed(2)}×…`;
        if (displayPhase === "write") return "Writing the amplified samples to the output…";
        return "Press run to watch the samples move.";
    })();

    return (
        <section className={`processing-panel ${processing ? "is-processing" : ""}`}>
            <div className="panel-heading">
                <div><span>PROCESSING</span><h3>Live sample transformation</h3></div>
                <small>{phase === "complete" ? "LIVE LOOP" : phaseLabel}</small>
            </div>

            <div className="plot-card">
                <div className="plot-strip">
                    <span className="plot-legend">
                        <span><i className="swatch swatch-input" />x[n]</span>
                        <span><i className="swatch swatch-process" />gain · x[n]</span>
                        <span><i className="swatch swatch-output" />y[n]</span>
                    </span>
                    <span className="plot-strip-meta">AXIS ±{range}</span>
                </div>
                <div ref={containerRef} className="plot-body">
                    <canvas ref={canvasRef} />
                </div>
            </div>

            <div className="processing-phase-strip">
                <span className={displayPhase === "load" ? "is-active" : ""}>READ</span>
                <i>→</i>
                <span className={displayPhase === "multiply" ? "is-active" : ""}>MULTIPLY × {activeFactor.toFixed(2)}</span>
                <i>→</i>
                <span className={displayPhase === "write" ? "is-active" : ""}>WRITE</span>
            </div>

            <div className="processing-status">
                <div className="processing-status-top">
                    <span>{statusText}</span>
                    <strong>{Math.round(displayProgress * 100)}%</strong>
                </div>
            </div>

            <div className="math-visual">
                <div className="math-formula">
                    <span>y[n]</span><strong>=</strong><span>{activeFactor.toFixed(2)}</span><strong>×</strong><span>x[n]</span>
                </div>
                <div className="math-readouts">
                    <div className="math-readout">
                        <span>SAMPLE n</span>
                        <b>{sampleNumber === null ? "—" : sampleNumber.toLocaleString()}</b>
                    </div>
                    <div className="math-readout is-input">
                        <span>x[n]</span>
                        <b>{formatSigned(xValue)}</b>
                    </div>
                    <div className="math-readout is-output">
                        <span>y[n]</span>
                        <b>{formatSigned(yValue)}</b>
                    </div>
                </div>
            </div>
        </section>
    );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

function AmplitudeDynamics() {
    const [audioFile, setAudioFile] = useState(null);
    const [inputUrl, setInputUrl] = useState("");
    const [inputData, setInputData] = useState(null);
    const [outputData, setOutputData] = useState(null);
    const [inputMeta, setInputMeta] = useState(null);
    const [outputMeta, setOutputMeta] = useState(null);
    const [duration, setDuration] = useState(0);
    const [outputDuration, setOutputDuration] = useState(0);
    const [factor, setFactor] = useState(2);
    const [runFactor, setRunFactor] = useState(2);
    const [processing, setProcessing] = useState(false);
    const [processProgress, setProcessProgress] = useState(0);
    const [processPhase, setProcessPhase] = useState("idle");
    const [processedUrl, setProcessedUrl] = useState("");
    const [error, setError] = useState("");
    const [showCompletion, setShowCompletion] = useState(false);
    const [compareOpen, setCompareOpen] = useState(false);
    const [inputTime, setInputTime] = useState(0);
    const [outputTime, setOutputTime] = useState(0);
    const [inputPlaying, setInputPlaying] = useState(false);
    const [outputPlaying, setOutputPlaying] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [domain, setDomain] = useState("time");
    const [backendResult, setBackendResult] = useState(null);

    const inputAudioRef = useRef(null);
    const outputAudioRef = useRef(null);
    const inputObjectUrlRef = useRef("");
    const inputPeakRef = useRef(null);
    const loadIdRef = useRef(0);
    const animationFrameRef = useRef(null);
    const animationDoneRef = useRef(null);
    const completionTimerRef = useRef(null);

    const canProcess = Boolean(audioFile) && Boolean(inputData) && !processing;

    const phaseLabel = useMemo(() => {
        if (processPhase === "load") return "LOADING INPUT";
        if (processPhase === "multiply") return "APPLYING GAIN";
        if (processPhase === "write") return "WRITING OUTPUT";
        if (processPhase === "complete") return "COMPLETE";
        return outputData ? "COMPLETE" : "READY";
    }, [processPhase, outputData]);

    useEffect(() => {
        return () => {
            if (inputObjectUrlRef.current) URL.revokeObjectURL(inputObjectUrlRef.current);
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
            if (completionTimerRef.current) clearTimeout(completionTimerRef.current);
        };
    }, []);

    // Keeps the playheads smooth while audio is playing (timeupdate is too coarse).
    useEffect(() => {
        if (!inputPlaying && !outputPlaying) return undefined;

        let frame = 0;
        const loop = () => {
            if (inputPlaying && inputAudioRef.current) setInputTime(inputAudioRef.current.currentTime);
            if (outputPlaying && outputAudioRef.current) setOutputTime(outputAudioRef.current.currentTime);
            frame = requestAnimationFrame(loop);
        };

        frame = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frame);
    }, [inputPlaying, outputPlaying]);

    useEffect(() => {
        if (!showCompletion && !compareOpen) return undefined;

        const onKey = (event) => {
            if (event.key !== "Escape") return;
            setShowCompletion(false);
            setCompareOpen(false);
        };

        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [showCompletion, compareOpen]);

    const selectFile = async (file) => {
        if (!file || processing) return;
        if (!file.name.toLowerCase().endsWith(".wav")) {
            setError("Please choose a WAV file.");
            return;
        }

        const loadId = loadIdRef.current + 1;
        loadIdRef.current = loadId;

        if (completionTimerRef.current) clearTimeout(completionTimerRef.current);
        inputAudioRef.current?.pause();
        outputAudioRef.current?.pause();

        setError("");
        setShowCompletion(false);
        setCompareOpen(false);
        setInputData(null);
        setInputMeta(null);
        setDuration(0);
        setInputTime(0);
        setOutputData(null);
        setOutputMeta(null);
        setProcessedUrl("");
        setOutputDuration(0);
        setOutputTime(0);
        setProcessProgress(0);
        setProcessPhase("idle");
        setBackendResult(null);
        setDomain("time");
        setAudioFile(file);

        if (inputObjectUrlRef.current) URL.revokeObjectURL(inputObjectUrlRef.current);
        const url = URL.createObjectURL(file);
        inputObjectUrlRef.current = url;
        setInputUrl(url);

        try {
            const decoded = await decodeAudio(url);
            if (loadId !== loadIdRef.current) return;

            inputPeakRef.current = decoded.peak;
            setInputData(decoded.data); // the input plot slowly draws itself from here
            setInputMeta({ channels: decoded.channels, sampleRate: decoded.sampleRate, length: decoded.length });
            setDuration(decoded.duration);
            setInputTime(0);
        } catch (err) {
            if (loadId !== loadIdRef.current) return;
            console.error(err);
            setInputData(null);
            setDuration(0);
            setError("The WAV file could not be decoded in the browser.");
        }
    };

    const seekAudio = (which, ratio) => {
        if (processing) return;

        const ref = which === "input" ? inputAudioRef.current : outputAudioRef.current;
        if (!ref) return;

        const targetDuration = which === "input" ? duration : outputDuration;
        const time = clamp(ratio, 0, 1) * targetDuration;
        ref.currentTime = time;

        if (which === "input") setInputTime(time);
        else setOutputTime(time);

        ref.play().catch(() => {});
    };

    // Moves the playhead without starting playback (used by the seek bars).
    const scrubAudio = (which, ratio) => {
        if (processing) return;

        const ref = which === "input" ? inputAudioRef.current : outputAudioRef.current;
        const targetDuration = which === "input" ? duration : outputDuration;
        if (!ref || !targetDuration) return;

        const time = clamp(ratio, 0, 1) * targetDuration;
        ref.currentTime = time;

        if (which === "input") setInputTime(time);
        else setOutputTime(time);
    };

    const togglePlay = (which) => {
        if (processing) return;

        const ref = which === "input" ? inputAudioRef.current : outputAudioRef.current;
        if (!ref) return;

        if (!ref.paused) {
            ref.pause();
            return;
        }

        if (ref.ended || (ref.duration && ref.currentTime >= ref.duration - 0.05)) ref.currentTime = 0;
        ref.play().catch(() => {});
    };

    const animateProcessing = () => new Promise((resolve) => {
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        animationDoneRef.current = resolve;

        const start = performance.now();

        const tick = (now) => {
            const overall = clamp((now - start) / PROCESS_TOTAL_MS, 0, 1);
            setProcessProgress(overall);
            setProcessPhase(phaseFromProgress(overall));

            if (overall >= 1) {
                // Stay on the last frame of "write"; "complete" is set once the
                // processed audio has actually arrived.
                animationFrameRef.current = null;
                if (animationDoneRef.current) {
                    animationDoneRef.current();
                    animationDoneRef.current = null;
                }
                return;
            }

            animationFrameRef.current = requestAnimationFrame(tick);
        };

        animationFrameRef.current = requestAnimationFrame(tick);
    });

    const runProcess = async () => {
        if (!audioFile || !inputData || processing) return;

        const safeFactor = Number(factor);
        if (!Number.isFinite(safeFactor) || safeFactor <= 0 || safeFactor > GAIN_MAX) {
            setError("Gain must be between 0.01× and 4×.");
            return;
        }

        if (completionTimerRef.current) clearTimeout(completionTimerRef.current);
        inputAudioRef.current?.pause();
        outputAudioRef.current?.pause();

        setError("");
        setProcessing(true);
        setShowCompletion(false);
        setCompareOpen(false);
        setRunFactor(safeFactor);
        setProcessProgress(0);
        setProcessPhase("load");
        setOutputData(null);
        setOutputMeta(null);
        setProcessedUrl("");
        setOutputTime(0);
        setBackendResult(null);
        setDomain("time");

        const visualAnimation = animateProcessing();

        try {
            const result = await processAudio(audioFile, "amplify", safeFactor, 5, 5, 5);
            setBackendResult(result);
            // Decode the output on the same scale as the input so the gain is visible.
            const decodedPromise = decodeAudio(result.audio_url, inputPeakRef.current);
            const [, decoded] = await Promise.all([visualAnimation, decodedPromise]);

            setProcessedUrl(result.audio_url);
            setOutputMeta({ channels: decoded.channels, sampleRate: decoded.sampleRate, length: decoded.length });
            setOutputDuration(decoded.duration || result.processed_duration_seconds || duration);
            setOutputTime(0);
            setProcessProgress(1);
            setProcessPhase("complete");
            setOutputData(decoded.data); // the output plot draws itself from here

            // Let the output finish drawing before the dialog appears.
            completionTimerRef.current = setTimeout(
                () => setShowCompletion(true),
                OUTPUT_REVEAL_MS + HANDOFF_MS + 300
            );
        } catch (err) {
            console.error(err);
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
            setProcessPhase("idle");
            setProcessProgress(0);
            setError("The audio could not be processed. Make sure the backend is running.");
        } finally {
            setProcessing(false);
        }
    };

    const openCompare = () => {
        if (!outputData || processing) return;
        setCompareOpen(true);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setDragging(false);
        selectFile(event.dataTransfer.files?.[0]);
    };

    /* ---- derived plot data ---- */

    const inputMarker = duration ? inputTime / duration : null;
    const outputMarker = outputDuration ? outputTime / outputDuration : null;

    const inputLayers = useMemo(
        () => [{ data: inputData, color: INPUT_COLOR, marker: inputMarker }],
        [inputData, inputMarker]
    );
    const outputLayers = useMemo(
        () => [{ data: outputData, color: OUTPUT_COLOR, marker: outputMarker }],
        [outputData, outputMarker]
    );
    const compareLayers = useMemo(
        () => [
            { data: inputData, color: INPUT_COLOR, marker: inputMarker },
            { data: outputData, color: OUTPUT_COLOR, marker: outputMarker },
        ],
        [inputData, outputData, inputMarker, outputMarker]
    );

    const inputRange = useMemo(() => pickRange(peakOf(inputData)), [inputData]);
    const outputRange = useMemo(() => pickRange(peakOf(outputData)), [outputData]);
    const compareRange = useMemo(
        () => pickRange(Math.max(peakOf(inputData), peakOf(outputData))),
        [inputData, outputData]
    );

    const gainPosition = (factor - GAIN_MIN) / (GAIN_MAX - GAIN_MIN);

    return (
        <main className="amplitude-page">
            <header className="module-topbar">
                <button className="back-button" onClick={() => { window.location.hash = "modules"; }} disabled={processing}>
                    ← DSP MODULES
                </button>
                <span>MODULE 01 / AMPLITUDE &amp; GAIN</span>
            </header>

            <section className="module-intro">
                <p className="section-label">AMPLITUDE &amp; GAIN</p>
                <h1>CHANGE THE <span>ENERGY</span><br />OF YOUR SIGNAL.</h1>
                <p>
                    Watch the signal move through the operation: samples are read from left to right,
                    multiplied by the gain factor, and written into the amplified output.
                </p>
            </section>

            <section className="module-workspace">
                <div
                    className={`module-controls ${dragging ? "is-dragging" : ""}`}
                    onDragOver={(event) => { event.preventDefault(); if (!processing) setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={handleDrop}
                >
                    <div>
                        <span className="control-kicker">01 / INPUT</span>
                        <h2>Bring in a WAV signal</h2>
                        <p>Choose an audio file or drop it here. The signal becomes the input to the operation.</p>
                    </div>
                    <div className="upload-group">
                        {audioFile && (
                            <div className="file-chip">
                                <strong title={audioFile.name}>{audioFile.name}</strong>
                                <small>{formatBytes(audioFile.size)}</small>
                            </div>
                        )}
                        <label className={`upload-module-button ${processing ? "is-disabled" : ""}`}>
                            <input
                                type="file"
                                accept="audio/*"
                                disabled={processing}
                                onChange={(event) => {
                                    const file = event.target.files?.[0];
                                    event.target.value = "";
                                    selectFile(file);
                                }}
                            />
                            {audioFile ? "CHANGE WAV" : "CHOOSE WAV"}
                        </label>
                    </div>
                </div>

                <div className="gain-control-row">
                    <div className="gain-readout">
                        <span className="control-kicker">GAIN FACTOR</span>
                        <strong>{Number(factor).toFixed(2)}×</strong>
                        <small>{formatDb(factor)}</small>
                    </div>
                    <div className="gain-slider-wrap">
                        <input
                            className="range-input"
                            type="range"
                            min={GAIN_MIN}
                            max={GAIN_MAX}
                            step="0.01"
                            value={factor}
                            style={{ "--pos": gainPosition }}
                            onChange={(event) => setFactor(Number(event.target.value))}
                            disabled={processing}
                            aria-label="Gain factor"
                        />
                        <div className="gain-ticks">
                            {GAIN_PRESETS.map((value) => (
                                <button
                                    key={value}
                                    type="button"
                                    style={{ "--pos": (value - GAIN_MIN) / (GAIN_MAX - GAIN_MIN) }}
                                    className={Math.abs(factor - value) < 0.005 ? "is-active" : ""}
                                    disabled={processing}
                                    onClick={() => setFactor(value)}
                                >
                                    {value}×
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="gain-formula">y[n] = {Number(factor).toFixed(2)} · x[n]</div>
                </div>

                {error && <div className="module-error">{error}</div>}

                <div className="domain-switch-row">
                    <button
                        type="button"
                        className={domain === "time" ? "is-selected" : ""}
                        onClick={() => setDomain("time")}
                        disabled={processing}
                    >
                        TIME DOMAIN
                    </button>
                    <button
                        type="button"
                        className={domain === "frequency" ? "is-selected" : ""}
                        onClick={() => setDomain("frequency")}
                        disabled={processing || !backendResult}
                        title={!backendResult ? "Run amplification first" : "Show the backend FFT spectrum"}
                    >
                        FREQUENCY SPECTRUM
                    </button>
                </div>

                {domain === "time" ? (
                <div className="signal-sections">
                    <section className={`signal-panel clickable-panel ${processing ? "panel-disabled" : ""}`}>
                        <div className="panel-heading">
                            <div><span>INPUT SIGNAL</span><h3>Original</h3></div>
                            <small className="panel-tag" title={audioFile?.name}>{audioFile ? audioFile.name : "No file yet"}</small>
                        </div>
                        <SignalPlot
                            layers={inputLayers}
                            range={inputRange}
                            duration={duration}
                            reveal={INPUT_REVEAL_MS}
                            revealKey={inputData}
                            onSeek={(ratio) => seekAudio("input", ratio)}
                            disabled={processing}
                            label="Input"
                            emptyText="Upload a WAV to begin"
                            title="x[n] · ORIGINAL SIGNAL"
                            meta={formatMeta(inputMeta)}
                        />
                        <Transport
                            tone="input"
                            playing={inputPlaying}
                            current={inputTime}
                            total={duration}
                            disabled={processing || !inputData}
                            onToggle={() => togglePlay("input")}
                            onScrub={(ratio) => scrubAudio("input", ratio)}
                        />
                        <p>Click the waveform to seek and listen.</p>
                    </section>

                    <ProcessingPanel
                        data={inputData}
                        duration={duration}
                        sampleCount={inputMeta?.length}
                        factor={factor}
                        runFactor={runFactor}
                        progress={processProgress}
                        phase={processPhase}
                        processing={processing}
                        phaseLabel={phaseLabel}
                    />

                    <section className={`signal-panel clickable-panel output-panel ${processing ? "panel-disabled" : ""}`}>
                        <div className="panel-heading">
                            <div><span>OUTPUT SIGNAL</span><h3>Amplified</h3></div>
                            <small className="panel-tag">{outputData ? `${runFactor.toFixed(2)}× gain` : "Waiting for the run"}</small>
                        </div>
                        <SignalPlot
                            layers={outputLayers}
                            range={outputRange}
                            duration={outputDuration}
                            reveal={OUTPUT_REVEAL_MS}
                            revealKey={outputData}
                            onSeek={(ratio) => seekAudio("output", ratio)}
                            disabled={processing || !outputData}
                            label="Output"
                            emptyText="Appears after the run"
                            title="y[n] · AMPLIFIED SIGNAL"
                            meta={formatMeta(outputMeta)}
                        />
                        <Transport
                            tone="output"
                            playing={outputPlaying}
                            current={outputTime}
                            total={outputDuration}
                            disabled={processing || !outputData}
                            onToggle={() => togglePlay("output")}
                            onScrub={(ratio) => scrubAudio("output", ratio)}
                        />
                        <p>{outputData ? "Click the waveform to seek and listen." : "Run the amplification to fill this in."}</p>
                    </section>
                </div>
                ) : (
                    <section className="processing-panel frequency-main-panel is-processing">
                        <div className="panel-heading">
                            <div>
                                <span>FREQUENCY DOMAIN</span>
                                <h3>What changed in the spectrum?</h3>
                            </div>
                            <small>{backendResult ? "BACKEND FFT" : "RUN FIRST"}</small>
                        </div>
                        {backendResult ? (
                            <FrequencySpectrumAnimator
                                visualization={backendResult.visualization}
                                mode="amplify"
                                gain={runFactor}
                                loop
                            />
                        ) : (
                            <div className="spectrum-empty">
                                Run the amplification once. The frequency spectrum is calculated by the Python backend.
                            </div>
                        )}
                    </section>
                )}

                <div className="module-action-row">
                    <button className="process-main-button" disabled={!canProcess} onClick={runProcess}>
                        {processing ? "PROCESSING…" : outputData ? "RUN AGAIN" : "RUN AMPLIFICATION"}
                    </button>
                    <button className="compare-button" disabled={!outputData || processing} onClick={openCompare}>COMPARE THE SIGNALS</button>
                </div>
            </section>

            <audio
                ref={inputAudioRef}
                src={inputUrl || undefined}
                preload="metadata"
                onTimeUpdate={(event) => setInputTime(event.currentTarget.currentTime)}
                onPlay={() => { setInputPlaying(true); outputAudioRef.current?.pause(); }}
                onPause={() => setInputPlaying(false)}
                onEnded={() => setInputPlaying(false)}
            />
            <audio
                ref={outputAudioRef}
                src={processedUrl || undefined}
                preload="metadata"
                onTimeUpdate={(event) => setOutputTime(event.currentTarget.currentTime)}
                onPlay={() => { setOutputPlaying(true); inputAudioRef.current?.pause(); }}
                onPause={() => setOutputPlaying(false)}
                onEnded={() => setOutputPlaying(false)}
            />

            {showCompletion && (
                <div className="module-modal-backdrop" onClick={(event) => { if (event.target === event.currentTarget) setShowCompletion(false); }}>
                    <section className="completion-modal" role="dialog" aria-modal="true" aria-label="Processing complete">
                        <button className="modal-close" onClick={() => setShowCompletion(false)} aria-label="Close">×</button>
                        <span className="modal-kicker">PROCESSING COMPLETE</span>
                        <h2>Your signal has been amplified.</h2>
                        <p>The animated transformation is complete. Choose a signal to listen to, or close this window to explore the waveforms.</p>
                        {domain === "time" && (
                            <div className="modal-audio-buttons">
                                <button onClick={() => togglePlay("input")} className={inputPlaying ? "selected" : ""}>
                                    {inputPlaying ? "❚❚ PAUSE INPUT" : "▶ INPUT AUDIO"}
                                </button>
                                <button onClick={() => togglePlay("output")} className={outputPlaying ? "selected" : ""}>
                                    {outputPlaying ? "❚❚ PAUSE OUTPUT" : "▶ OUTPUT AUDIO"}
                                </button>
                            </div>
                        )}
                        <div className="modal-domain-toggle">
                            <button
                                className={domain === "time" ? "is-selected" : ""}
                                onClick={() => setDomain("time")}
                            >
                                TIME DOMAIN
                            </button>
                            <button
                                className={domain === "frequency" ? "is-selected" : ""}
                                onClick={() => setDomain("frequency")}
                                disabled={!backendResult}
                            >
                                FREQUENCY SPECTRUM
                            </button>
                        </div>

                        {domain === "time" ? (
                            <div className="modal-mini-graph">
                                <SignalPlot
                                    layers={inputLayers}
                                    range={compareRange}
                                    duration={duration}
                                    onSeek={(ratio) => seekAudio("input", ratio)}
                                    label="Input preview"
                                    title="x[n] · INPUT"
                                    meta={formatMeta(inputMeta)}
                                />
                                <SignalPlot
                                    layers={outputLayers}
                                    range={compareRange}
                                    duration={outputDuration}
                                    onSeek={(ratio) => seekAudio("output", ratio)}
                                    label="Output preview"
                                    title="y[n] · OUTPUT"
                                    meta={formatMeta(outputMeta)}
                                />
                            </div>
                        ) : (
                            <FrequencySpectrumAnimator
                                visualization={backendResult?.visualization}
                                mode="amplify"
                                gain={runFactor}
                                loop
                            />
                        )}
                    </section>
                </div>
            )}

            {compareOpen && (
                <div className="module-modal-backdrop" onClick={(event) => { if (event.target === event.currentTarget) setCompareOpen(false); }}>
                    <section className="compare-modal" role="dialog" aria-modal="true" aria-label="Compare signals">
                        <button className="modal-close" onClick={() => setCompareOpen(false)} aria-label="Close">×</button>
                        <span className="modal-kicker">SIGNAL COMPARISON</span>
                        <h2>Input + output, on the same graph.</h2>
                        <p>Blue is the original signal. Red is the amplified signal, drawn on the same amplitude scale. Click the upper half for input or the lower half for output.</p>
                        <div className="combined-graph">
                            <SignalPlot
                                layers={compareLayers}
                                range={compareRange}
                                duration={duration}
                                onSeek={(ratio, yRatio) => seekAudio(yRatio < 0.5 ? "input" : "output", ratio)}
                                label="Combined input and output"
                                title="x[n] vs y[n]"
                                meta={`AXIS ±${compareRange}`}
                            />
                        </div>
                        <div className="comparison-legend"><span><i className="blue-dot" /> INPUT</span><span><i className="red-dot" /> OUTPUT</span></div>
                        <div className="modal-audio-buttons">
                            <button onClick={() => togglePlay("input")} className={inputPlaying ? "selected" : ""}>
                                {inputPlaying ? "❚❚ PAUSE INPUT" : "▶ INPUT AUDIO"}
                            </button>
                            <button onClick={() => togglePlay("output")} className={outputPlaying ? "selected" : ""}>
                                {outputPlaying ? "❚❚ PAUSE OUTPUT" : "▶ OUTPUT AUDIO"}
                            </button>
                        </div>
                    </section>
                </div>
            )}
        </main>
    );
}

export default AmplitudeDynamics;