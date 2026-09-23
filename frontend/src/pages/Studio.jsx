import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./../styles/studio.css";

const MODULE_DEFAULTS = {
    amplification: { enabled: true, value: 6 },
    filtering: { enabled: false, value: 55 },
    echo: { enabled: true, value: 35 },
    gate: { enabled: false, value: 25 },
    pitch: { enabled: false, value: 0 },
    morph: { enabled: false, value: 15 }
};

const MODULE_META = {
    amplification: {
        name: "Amplification",
        short: "GAIN",
        min: 0,
        max: 20,
        step: 1,
        unit: "dB"
    },
    filtering: {
        name: "Filtering",
        short: "LOW-PASS",
        min: 0,
        max: 100,
        step: 1,
        unit: "%"
    },
    echo: {
        name: "Echo",
        short: "MIX",
        min: 0,
        max: 100,
        step: 1,
        unit: "%"
    },
    gate: {
        name: "Speech Gate",
        short: "THRESHOLD",
        min: 0,
        max: 100,
        step: 1,
        unit: "%"
    },
    pitch: {
        name: "Pitch Shift",
        short: "SEMITONES",
        min: -12,
        max: 12,
        step: 1,
        unit: "st"
    },
    morph: {
        name: "Voice Morph",
        short: "CHARACTER",
        min: 0,
        max: 100,
        step: 1,
        unit: "%"
    }
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const formatTime = (seconds) => {
    if (!Number.isFinite(seconds)) return "00:00";
    const safe = Math.max(0, seconds);
    const minutes = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60);
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

const copyModules = () =>
    Object.fromEntries(
        Object.entries(MODULE_DEFAULTS).map(([key, value]) => [
            key,
            { ...value }
        ])
    );

function makeDisplayWaveform(samples, targetPoints = 180) {
    if (!samples?.length) return [];

    // VISUALIZATION ONLY.
    // The real y[n] array is never replaced. We deliberately reduce the
    // display to a small number of representative points and then smooth
    // those points so the canvas reads like a continuous signal/function
    // instead of a dense collection of spikes.
    const count = Math.min(targetPoints, samples.length);
    const stride = samples.length / count;
    const points = new Float32Array(count);

    for (let i = 0; i < count; i += 1) {
        const start = Math.floor(i * stride);
        const end = Math.max(
            start + 1,
            Math.min(samples.length, Math.floor((i + 1) * stride))
        );

        let sum = 0;
        let weight = 0;

        // Use a small weighted low-pass average instead of taking the
        // strongest sample. Taking the peak is what created the vertical
        // spike / zig-zag appearance in the previous visualization.
        for (let j = start; j < end; j += 1) {
            const distance = Math.abs(
                j - (start + end - 1) / 2
            );
            const span = Math.max(1, (end - start) / 2);
            const w = 1 - Math.min(0.9, distance / span) * 0.45;
            sum += samples[j] * w;
            weight += w;
        }

        points[i] = weight > 0 ? sum / weight : 0;
    }

    // Smooth the reduced points with a Gaussian-like moving window.
    // This is strictly a display operation and does not alter audio.
    const smoothed = new Float32Array(count);
    const radius = Math.max(2, Math.floor(count / 28));

    for (let i = 0; i < count; i += 1) {
        let sum = 0;
        let weight = 0;

        for (
            let j = Math.max(0, i - radius);
            j <= Math.min(count - 1, i + radius);
            j += 1
        ) {
            const distance = Math.abs(i - j);
            const sigma = Math.max(1, radius * 0.55);
            const w = Math.exp(
                -(distance * distance) /
                    (2 * sigma * sigma)
            );

            sum += points[j] * w;
            weight += w;
        }

        smoothed[i] = weight > 0 ? sum / weight : 0;
    }

    let peak = 0;

    for (let i = 0; i < smoothed.length; i += 1) {
        peak = Math.max(peak, Math.abs(smoothed[i]));
    }

    if (peak < 0.000001) return Array(count).fill(0);

    // Keep the visible signal comfortably inside the graph. This only
    // changes visualization coordinates; the actual audio remains intact.
    const targetPeak = 0.78;

    return Array.from(smoothed, (value) =>
        clamp((value / peak) * targetPeak, -0.92, 0.92)
    );
}

function calculateSpectrum(samples, bins = 96) {
    if (!samples?.length) return [];

    const frameSize = Math.min(2048, samples.length);
    const start = Math.max(0, Math.floor((samples.length - frameSize) / 2));
    const step = Math.max(1, Math.floor(frameSize / 512));

    const frame = new Float32Array(frameSize);

    for (let i = 0; i < frameSize; i += 1) {
        const index = start + i;
        const window =
            0.5 -
            0.5 *
                Math.cos((2 * Math.PI * i) / Math.max(1, frameSize - 1));

        frame[i] = (samples[index] || 0) * window;
    }

    const result = [];

    for (let k = 0; k < bins; k += 1) {
        const bin = Math.floor((k / bins) * (frameSize / 2));

        let real = 0;
        let imag = 0;

        for (let n = 0; n < frameSize; n += step) {
            const angle = (2 * Math.PI * bin * n) / frameSize;
            real += frame[n] * Math.cos(angle);
            imag -= frame[n] * Math.sin(angle);
        }

        const magnitude =
            Math.sqrt(real * real + imag * imag) /
            Math.max(1, frameSize / step);

        result.push({
            frequency: bin,
            magnitude
        });
    }

    const max = Math.max(...result.map((point) => point.magnitude), 0.000001);

    return result.map((point) => ({
        ...point,
        normalized: clamp(point.magnitude / max, 0, 1)
    }));
}

function applyPitchPreview(samples, semitones) {
    if (!samples?.length || semitones === 0) return new Float32Array(samples);

    const ratio = Math.pow(2, semitones / 12);
    const output = new Float32Array(samples.length);

    for (let i = 0; i < output.length; i += 1) {
        const sourcePosition = i * ratio;
        const wrapped = sourcePosition % samples.length;
        const left = Math.floor(wrapped);
        const right = (left + 1) % samples.length;
        const fraction = wrapped - left;

        output[i] =
            samples[left] * (1 - fraction) + samples[right] * fraction;
    }

    return output;
}

function processChannel(samples, sampleRate, modules) {
    let working = new Float32Array(samples);

    if (modules.pitch.enabled && modules.pitch.value !== 0) {
        working = applyPitchPreview(working, modules.pitch.value);
    }

    const gain = modules.amplification.enabled
        ? Math.pow(10, modules.amplification.value / 20)
        : 1;

    const filterAmount = modules.filtering.enabled
        ? modules.filtering.value / 100
        : 0;

    const echoAmount = modules.echo.enabled ? modules.echo.value / 100 : 0;

    const gateAmount = modules.gate.enabled ? modules.gate.value / 100 : 0;

    const morphAmount = modules.morph.enabled ? modules.morph.value / 100 : 0;

    const output = new Float32Array(working.length);

    const cutoff = 18000 - filterAmount * 15500;
    const alpha = clamp(
        (2 * Math.PI * cutoff) /
            (2 * Math.PI * cutoff + Math.max(1, sampleRate)),
        0.015,
        0.95
    );

    let lowPassState = 0;

    const echoDelay = Math.max(1, Math.floor(sampleRate * 0.18));
    const echoBuffer = new Float32Array(echoDelay);

    let echoIndex = 0;

    for (let i = 0; i < working.length; i += 1) {
        let value = working[i] * gain;

        if (filterAmount > 0) {
            lowPassState += alpha * (value - lowPassState);
            value = lowPassState;
        }

        if (gateAmount > 0) {
            const threshold = 0.015 + gateAmount * 0.25;
            if (Math.abs(value) < threshold) {
                value *= 1 - gateAmount;
            }
        }

        if (echoAmount > 0) {
            const delayed = echoBuffer[echoIndex];
            echoBuffer[echoIndex] = value;
            value = value * (1 - echoAmount * 0.45) + delayed * echoAmount * 0.45;
            echoIndex = (echoIndex + 1) % echoDelay;
        }

        if (morphAmount > 0) {
            const harmonic = Math.tanh(value * (1 + morphAmount * 4));
            value = value * (1 - morphAmount * 0.35) + harmonic * morphAmount * 0.35;
        }

        output[i] = value;
    }

    return output;
}

function processAudioBuffer(buffer, modules) {
    const channels = [];

    let clippedSamples = 0;
    let peak = 0;

    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const input = buffer.getChannelData(channel);
        const processed = processChannel(input, buffer.sampleRate, modules);

        for (let i = 0; i < processed.length; i += 1) {
            const absolute = Math.abs(processed[i]);
            peak = Math.max(peak, absolute);

            if (absolute > 1) {
                clippedSamples += 1;
                processed[i] = clamp(processed[i], -1, 1);
            }
        }

        channels.push(processed);
    }

    const context = new AudioContext();
    const output = context.createBuffer(
        buffer.numberOfChannels,
        buffer.length,
        buffer.sampleRate
    );

    channels.forEach((channel, index) => {
        output.copyToChannel(channel, index);
    });

    context.close();

    return {
        buffer: output,
        clippedSamples,
        peak
    };
}

function levelStats(samples) {
    if (!samples?.length) {
        return { peak: 0, rms: 0, db: -Infinity };
    }

    let peak = 0;
    let sum = 0;

    for (let i = 0; i < samples.length; i += 1) {
        const value = samples[i];
        peak = Math.max(peak, Math.abs(value));
        sum += value * value;
    }

    const rms = Math.sqrt(sum / samples.length);

    return {
        peak,
        rms,
        db: 20 * Math.log10(Math.max(rms, 0.000001))
    };
}

function GraphCanvas({
    title,
    subtitle,
    badge,
    data,
    type,
    color,
    cursorProgress,
    zoom,
    viewCenter,
    onZoomChange,
    onCenterChange,
    onOpen,
    playMode,
    playActive,
    onPlay,
    duration = 1,
    compact = false
}) {
    const canvasRef = useRef(null);
    const [hover, setHover] = useState(null);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !data?.length) return;

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));

        canvas.width = width * dpr;
        canvas.height = height * dpr;

        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const background =
            getComputedStyle(canvas).getPropertyValue("--graph-canvas").trim() ||
            "#5557ae";

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = background;
        ctx.fillRect(0, 0, width, height);

        const left = compact ? 34 : 42;
        const right = width - 18;
        const top = 18;
        const bottom = height - 30;
        const graphWidth = right - left;
        const graphHeight = bottom - top;

        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i += 1) {
            const y = top + (graphHeight * i) / 4;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(right, y);
            ctx.stroke();
        }

        for (let i = 0; i <= 5; i += 1) {
            const x = left + (graphWidth * i) / 5;
            ctx.beginPath();
            ctx.moveTo(x, top);
            ctx.lineTo(x, bottom);
            ctx.stroke();
        }

        ctx.strokeStyle = "rgba(255,255,255,0.26)";
        ctx.beginPath();
        ctx.moveTo(left, top + graphHeight / 2);
        ctx.lineTo(right, top + graphHeight / 2);
        ctx.stroke();

        ctx.fillStyle = "rgba(255,255,255,0.54)";
        ctx.font = `${compact ? 9 : 10}px Inter, system-ui, sans-serif`;

        if (type === "time") {
            ctx.fillText("+1", 8, top + 4);
            ctx.fillText("0", 12, top + graphHeight / 2 + 3);
            ctx.fillText("-1", 8, bottom);

            const visibleCount = Math.max(
                2,
                Math.floor(data.length / Math.max(1, zoom))
            );

            const centerIndex = Math.floor(viewCenter * data.length);
            const start = clamp(
                centerIndex - Math.floor(visibleCount / 2),
                0,
                Math.max(0, data.length - visibleCount)
            );
            const end = Math.min(data.length, start + visibleCount);

            let visiblePeak = 0;

            for (let i = start; i < end; i += 1) {
                visiblePeak = Math.max(visiblePeak, Math.abs(data[i]));
            }

            // Keep at least ~60% of the graph useful vertically, while
            // preserving the waveform shape.
            const yScale = Math.max(0.60, Math.min(0.94, visiblePeak));

            // Draw a smooth cubic curve through the sparse display points.
            // The dense audio samples are never used as the visible trace.
            const visiblePoints = [];

            for (let i = start; i < end; i += 1) {
                const ratio =
                    (i - start) / Math.max(1, end - start - 1);
                const x = left + graphWidth * ratio;
                const y =
                    top +
                    graphHeight / 2 -
                    (data[i] / yScale) * (graphHeight * 0.42);

                visiblePoints.push({ x, y });
            }

            ctx.beginPath();

            if (visiblePoints.length > 0) {
                ctx.moveTo(
                    visiblePoints[0].x,
                    visiblePoints[0].y
                );

                for (let i = 0; i < visiblePoints.length - 1; i += 1) {
                    const current = visiblePoints[i];
                    const next = visiblePoints[i + 1];
                    const previous =
                        visiblePoints[Math.max(0, i - 1)];
                    const following =
                        visiblePoints[
                            Math.min(
                                visiblePoints.length - 1,
                                i + 2
                            )
                        ];

                    const tension = 0.16;
                    const cp1x =
                        current.x +
                        (next.x - previous.x) * tension;
                    const cp1y =
                        current.y +
                        (next.y - previous.y) * tension;
                    const cp2x =
                        next.x -
                        (following.x - current.x) * tension;
                    const cp2y =
                        next.y -
                        (following.y - current.y) * tension;

                    ctx.bezierCurveTo(
                        cp1x,
                        cp1y,
                        cp2x,
                        cp2y,
                        next.x,
                        next.y
                    );
                }
            }

            ctx.strokeStyle = color;
            ctx.lineWidth = compact ? 3 : 2.4;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.stroke();

            for (let i = 0; i <= 5; i += 1) {
                const ratio = i / 5;
                const x = left + graphWidth * ratio;
                const timeRatio = clamp(
                    viewCenter - 0.5 / zoom + ratio / zoom,
                    0,
                    1
                );
                const seconds = timeRatio * duration;

                ctx.fillText(
                    `${seconds.toFixed(1)}s`,
                    x - 12,
                    height - 10
                );
            }

            const cursor = clamp(cursorProgress, 0, 1);

            if (
                cursor >=
                    viewCenter - 0.5 / zoom &&
                cursor <=
                    viewCenter + 0.5 / zoom
            ) {
                const ratio =
                    (cursor - (viewCenter - 0.5 / zoom)) * zoom;
                const x = left + graphWidth * clamp(ratio, 0, 1);

                const cursorIndex = clamp(
                    Math.floor(cursor * (data.length - 1)),
                    0,
                    data.length - 1
                );

                const cursorY =
                    top +
                    graphHeight / 2 -
                    (data[cursorIndex] / yScale) *
                        (graphHeight * 0.42);

                ctx.strokeStyle = "rgba(255,255,255,0.30)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();

                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, cursorY, compact ? 7 : 6, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, cursorY, compact ? 7 : 6, 0, Math.PI * 2);
                ctx.stroke();
            }
        } else {
            ctx.fillText("MAG", 8, top + 4);
            ctx.fillText("0", 18, bottom);

            const visibleBins = Math.max(
                4,
                Math.floor(data.length / Math.max(1, zoom))
            );

            const centerIndex = Math.floor(viewCenter * data.length);
            const start = clamp(
                centerIndex - Math.floor(visibleBins / 2),
                0,
                Math.max(0, data.length - visibleBins)
            );
            const end = Math.min(data.length, start + visibleBins);

            let visibleMax = 0;

            for (let i = start; i < end; i += 1) {
                visibleMax = Math.max(
                    visibleMax,
                    data[i]?.normalized || 0
                );
            }

            const yScale = Math.max(0.60, visibleMax);
            const spacing =
                graphWidth / Math.max(1, end - start);

            for (let i = start; i < end; i += 1) {
                const ratio =
                    (i - start) /
                    Math.max(1, end - start - 1);
                const x = left + graphWidth * ratio;
                const magnitude = clamp(
                    (data[i].normalized || 0) / yScale,
                    0,
                    1
                );
                const y =
                    bottom -
                    magnitude * graphHeight * 0.82;

                ctx.strokeStyle = color;
                ctx.globalAlpha = 0.30 + magnitude * 0.70;
                ctx.lineWidth = Math.max(
                    1,
                    Math.min(3, spacing * 0.55)
                );
                ctx.beginPath();
                ctx.moveTo(x, bottom);
                ctx.lineTo(x, y);
                ctx.stroke();

                ctx.globalAlpha = 1;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(
                    x,
                    y,
                    magnitude > 0.07 ? 3.2 : 2.1,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            }

            ctx.fillStyle = "rgba(255,255,255,0.50)";
            ctx.fillText("0 Hz", left, height - 10);
            ctx.fillText("Nyquist", right - 44, height - 10);

            // Frequency is not a time axis, so the moving playback cursor
            // is intentionally not drawn here. Instead, highlight the
            // strongest visible spectral component.
            let peakIndex = start;
            let peakValue = 0;

            for (let i = start; i < end; i += 1) {
                const value = data[i]?.normalized || 0;
                if (value > peakValue) {
                    peakValue = value;
                    peakIndex = i;
                }
            }

            if (peakIndex >= start && peakIndex < end) {
                const ratio =
                    (peakIndex - start) /
                    Math.max(1, end - start - 1);
                const x = left + graphWidth * ratio;
                const y =
                    bottom -
                    clamp(
                        (data[peakIndex].normalized || 0) / yScale,
                        0,
                        1
                    ) *
                        graphHeight *
                        0.82;

                ctx.strokeStyle = "rgba(255,255,255,0.24)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();

                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, compact ? 7 : 6, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y, compact ? 7 : 6, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    }, [
        data,
        type,
        color,
        cursorProgress,
        zoom,
        viewCenter,
        duration,
        compact
    ]);

    useEffect(() => {
        draw();

        const handleResize = () => draw();
        window.addEventListener("resize", handleResize);

        return () => window.removeEventListener("resize", handleResize);
    }, [draw]);

    const handleWheel = (event) => {
        // Zoom is intentionally available only in the enlarged analysis view.
        // On the 2×2 overview grid, normal page scrolling should remain normal.
        if (!compact) return;

        event.preventDefault();
        event.stopPropagation();

        const next = clamp(
            zoom + (event.deltaY < 0 ? 0.5 : -0.5),
            1,
            16
        );

        onZoomChange(next);
    };

    const handleMouseMove = (event) => {
        const canvas = canvasRef.current;
        if (!canvas || !data?.length) return;

        const rect = canvas.getBoundingClientRect();
        const ratio = clamp(
            (event.clientX - rect.left) / rect.width,
            0,
            1
        );

        if (type === "spectrum") {
            const index = clamp(
                Math.floor(ratio * data.length),
                0,
                data.length - 1
            );

            const point = data[index];

            if (point) {
                setHover({
                    x: event.clientX - rect.left,
                    y: event.clientY - rect.top,
                    text: `${point.frequency.toFixed(0)} Hz · ${(point.normalized * 100).toFixed(1)}%`
                });
            }
        }
    };

    const handleMouseLeave = () => setHover(null);

    const handleDoubleClick = () => {
        onZoomChange(1);
        onCenterChange(0.5);
    };

    return (
        <article
            className={`graph-card ${compact ? "graph-card--expanded" : ""}`}
            onClick={(event) => {
                if (!onOpen) return;
                if (
                    event.target.closest("button") ||
                    event.target.closest("input")
                ) {
                    return;
                }
                onOpen();
            }}
            style={onOpen ? { cursor: "zoom-in" } : undefined}
        >
            <div className="graph-card__header">
                <div>
                    <div className="graph-card__title-row">
                        <span
                            className="signal-dot"
                            style={{ background: color }}
                        />
                        <h3>{title}</h3>
                    </div>
                    <p>{subtitle}</p>
                </div>

                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 7
                    }}
                >
                    {onPlay && (
                        <PlaybackButton
                            mode={playMode}
                            active={playActive}
                            onClick={(event) => {
                                event.stopPropagation();
                                onPlay();
                            }}
                        />
                    )}

                    <span
                        className="graph-badge"
                        style={{ borderColor: color, color }}
                    >
                        {badge}
                    </span>
                </div>
            </div>

            <div className="graph-toolbar">
                <span>
                    {compact
                        ? type === "time"
                            ? "Scroll to zoom time · click to reposition · double-click to reset"
                            : "Scroll to zoom frequency · click to reposition · double-click to reset"
                        : "Click the graph to open the enlarged analysis view"}
                </span>

                {compact && (
                    <div className="graph-zoom">
                        <button
                            type="button"
                            onClick={(event) => {
                                event.stopPropagation();
                                onZoomChange(clamp(zoom - 0.5, 1, 16));
                            }}
                            aria-label={`Zoom out ${title}`}
                        >
                            −
                        </button>

                        <span>{Math.round(zoom * 100)}%</span>

                        <button
                            type="button"
                            onClick={(event) => {
                                event.stopPropagation();
                                onZoomChange(clamp(zoom + 0.5, 1, 16));
                            }}
                            aria-label={`Zoom in ${title}`}
                        >
                            +
                        </button>

                        <button
                            type="button"
                            className="graph-reset"
                            onClick={(event) => {
                                event.stopPropagation();
                                onZoomChange(1);
                                onCenterChange(0.5);
                            }}
                        >
                            Reset
                        </button>
                    </div>
                )}
            </div>

            <div
                className="graph-canvas-wrap"
                style={compact ? { height: "520px" } : undefined}
            >
                <canvas
                    ref={canvasRef}
                    className="graph-canvas"
                    onWheel={handleWheel}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    onDoubleClick={handleDoubleClick}
                    onClick={(event) => {
                        const rect =
                            event.currentTarget.getBoundingClientRect();

                        const ratio = clamp(
                            (event.clientX - rect.left) /
                                rect.width,
                            0,
                            1
                        );

                        onCenterChange(
                            clamp(
                                viewCenter -
                                    0.5 / zoom +
                                    ratio / zoom,
                                0,
                                1
                            )
                        );
                    }}
                />

                {hover && (
                    <div
                        className="spectrum-tooltip"
                        style={{
                            left: hover.x,
                            top: hover.y
                        }}
                    >
                        {hover.text}
                    </div>
                )}
            </div>
        </article>
    );
}

function GraphModal({
    graph,
    onClose,
    zoom,
    viewCenter,
    onZoomChange,
    onCenterChange,
    cursorProgress,
    playActive,
    onPlay,
    duration = 1
}) {
    if (!graph) return null;

    return (
        <div
            role="dialog"
            aria-modal="true"
            aria-label={`${graph.title} enlarged view`}
            onClick={onClose}
            onWheel={(event) => {
                event.preventDefault();
            }}
            style={{
                position: "fixed",
                inset: 0,
                zIndex: 1000,
                background: "rgba(3, 7, 16, 0.78)",
                backdropFilter: "blur(12px)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "28px"
            }}
        >
            <div
                onClick={(event) => event.stopPropagation()}
                style={{
                    width: "min(1180px, 94vw)",
                    maxHeight: "92vh",
                    overflow: "hidden",
                    border: "1px solid rgba(255,255,255,0.18)",
                    borderRadius: "20px",
                    background: "var(--studio-surface, rgba(18,25,45,0.96))",
                    boxShadow: "0 28px 100px rgba(0,0,0,0.45)",
                    padding: "18px"
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: "16px",
                        marginBottom: "12px"
                    }}
                >
                    <div>
                        <div className="studio-eyebrow">
                            ENLARGED ANALYSIS
                        </div>
                        <h2 style={{ margin: 0 }}>
                            {graph.title}
                        </h2>
                        <p
                            style={{
                                margin: "4px 0 0",
                                opacity: 0.7,
                                fontSize: "12px"
                            }}
                        >
                            {graph.subtitle}
                        </p>
                    </div>

                    <div
                        style={{
                            display: "flex",
                            gap: "8px",
                            alignItems: "center"
                        }}
                    >
                        {onPlay && (
                            <PlaybackButton
                                mode={graph.playMode}
                                active={playActive}
                                onClick={onPlay}
                            />
                        )}

                        <button
                            type="button"
                            className="secondary-button"
                            onClick={onClose}
                        >
                            Close
                        </button>
                    </div>
                </div>

                <GraphCanvas
                    title={graph.title}
                    subtitle={graph.subtitle}
                    badge={graph.badge}
                    data={graph.data}
                    type={graph.type}
                    color={graph.color}
                    cursorProgress={cursorProgress}
                    zoom={zoom}
                    viewCenter={viewCenter}
                    onZoomChange={onZoomChange}
                    onCenterChange={onCenterChange}
                    duration={duration}
                    playMode={graph.playMode}
                    playActive={playActive}
                    onPlay={onPlay}
                    compact
                />

                <div
                    style={{
                        marginTop: "10px",
                        fontSize: "11px",
                        opacity: 0.65
                    }}
                >
                    Scroll to zoom horizontally · click to reposition ·
                    double-click to reset · the vertical scale auto-fits the
                    visible signal
                </div>
            </div>
        </div>
    );
}

function ModuleControl({
    moduleKey,
    config,
    onToggle,
    onChange
}) {
    const meta = MODULE_META[moduleKey];

    const displayValue =
        moduleKey === "amplification"
            ? `${config.value} dB`
            : moduleKey === "pitch"
              ? `${config.value > 0 ? "+" : ""}${config.value} st`
              : `${config.value}%`;

    return (
        <div className={`module-control ${config.enabled ? "is-on" : ""}`}>
            <div className="module-control__top">
                <div>
                    <strong>{meta.name}</strong>
                    <span>{meta.short}</span>
                </div>

                <button
                    type="button"
                    className={`module-toggle ${
                        config.enabled ? "is-active" : ""
                    }`}
                    onClick={() => onToggle(moduleKey)}
                >
                    {config.enabled ? "ON" : "OFF"}
                </button>
            </div>

            <div className="module-control__slider-row">
                <input
                    type="range"
                    min={meta.min}
                    max={meta.max}
                    step={meta.step}
                    value={config.value}
                    onChange={(event) =>
                        onChange(moduleKey, Number(event.target.value))
                    }
                    disabled={!config.enabled}
                    aria-label={`${meta.name} ${meta.short}`}
                />

                <output>{displayValue}</output>
            </div>
        </div>
    );
}

function LevelMeter({ label, stats, clipped }) {
    const level = clamp(
        stats?.peak ?? 0,
        0,
        1
    );

    const rms = clamp(
        stats?.rms ?? 0,
        0,
        1
    );

    return (
        <div className="level-meter">
            <div className="level-meter__heading">
                <span>{label}</span>
                {clipped > 0 && (
                    <strong className="clip-warning">CLIP</strong>
                )}
            </div>

            <div className="meter-track">
                <div
                    className="meter-rms"
                    style={{ width: `${rms * 100}%` }}
                />
                <div
                    className="meter-peak"
                    style={{ width: `${level * 100}%` }}
                />
            </div>

            <div className="level-meter__footer">
                <span>
                    RMS{" "}
                    {Number.isFinite(stats?.db)
                        ? `${stats.db.toFixed(1)} dB`
                        : "—"}
                </span>
                <span>PEAK {Math.round(level * 100)}%</span>
            </div>
        </div>
    );
}

function PlaybackButton({ mode, active, onClick }) {
    return (
        <button
            type="button"
            className={`playback-button ${active ? "is-active" : ""}`}
            onClick={onClick}
        >
            <span>{active ? "Ⅱ" : "▶"}</span>
            {mode === "input" ? "Input" : "Output"}
        </button>
    );
}

export default function Studio() {
    const fileInputRef = useRef(null);
    const audioContextRef = useRef(null);
    const sourceRef = useRef(null);
    const animationRef = useRef(null);
    const scanStartRef = useRef(0);
    const playbackStartRef = useRef(0);
    const playbackOffsetRef = useRef(0);
    const autoProcessTimerRef = useRef(null);

    const [audioBuffer, setAudioBuffer] = useState(null);
    const [processedBuffer, setProcessedBuffer] = useState(null);
    const [fileName, setFileName] = useState("");
    const [inputWaveform, setInputWaveform] = useState([]);
    const [outputWaveform, setOutputWaveform] = useState([]);
    const [inputSpectrum, setInputSpectrum] = useState([]);
    const [outputSpectrum, setOutputSpectrum] = useState([]);
    const [modules, setModules] = useState(copyModules);
    const [analyzed, setAnalyzed] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playMode, setPlayMode] = useState(null);
    const [progress, setProgress] = useState(0);
    const [scanProgress, setScanProgress] = useState(0);
    const [graphViews, setGraphViews] = useState({});

    const getGraphView = useCallback(
        (key) =>
            graphViews[key] || {
                zoom: 1,
                center: 0.5
            },
        [graphViews]
    );

    const updateGraphView = useCallback((key, patch) => {
        setGraphViews((current) => ({
            ...current,
            [key]: {
                ...(current[key] || {
                    zoom: 1,
                    center: 0.5
                }),
                ...patch
            }
        }));
    }, []);
    const [loopPlayback, setLoopPlayback] = useState(true);
    const [abMode, setAbMode] = useState(false);
    const [expandedGraph, setExpandedGraph] = useState(null);
    const [hoverMode, setHoverMode] = useState(false);
    const [stats, setStats] = useState({
        input: { peak: 0, rms: 0, db: -Infinity },
        output: { peak: 0, rms: 0, db: -Infinity },
        clipped: 0
    });

    const duration = audioBuffer?.duration || 0;

    const currentProgress = isPlaying ? progress : scanProgress;

    const stopPlayback = useCallback(() => {
        if (sourceRef.current) {
            try {
                sourceRef.current.onended = null;
                sourceRef.current.stop();
            } catch {
                // Source may already be stopped.
            }

            sourceRef.current.disconnect();
            sourceRef.current = null;
        }

        setIsPlaying(false);
        setPlayMode(null);
    }, []);

    const createAudioContext = useCallback(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext();
        }

        return audioContextRef.current;
    }, []);

    const playBuffer = useCallback(
        async (mode, offset = 0) => {
            const buffer =
                mode === "input" ? audioBuffer : processedBuffer;

            if (!buffer) return;

            stopPlayback();

            const context = createAudioContext();

            if (context.state === "suspended") {
                await context.resume();
            }

            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);

            const safeOffset = clamp(
                offset,
                0,
                Math.max(0, buffer.duration - 0.01)
            );

            playbackStartRef.current =
                context.currentTime - safeOffset;

            playbackOffsetRef.current = safeOffset;

            source.onended = () => {
                if (sourceRef.current !== source) return;

                if (loopPlayback) {
                    playBuffer(mode, 0);
                    return;
                }

                setIsPlaying(false);
                setPlayMode(null);
                sourceRef.current = null;
            };

            source.start(0, safeOffset);
            sourceRef.current = source;

            setPlayMode(mode);
            setIsPlaying(true);
        },
        [
            audioBuffer,
            processedBuffer,
            createAudioContext,
            loopPlayback,
            stopPlayback
        ]
    );

    const processAndAnalyze = useCallback(
        async (buffer = audioBuffer) => {
            if (!buffer) return;

            setProcessing(true);
            stopPlayback();

            await new Promise((resolve) => requestAnimationFrame(resolve));

            const result = processAudioBuffer(buffer, modules);
            const inputChannel = buffer.getChannelData(0);
            const outputChannel = result.buffer.getChannelData(0);

            setProcessedBuffer(result.buffer);
            setInputWaveform(makeDisplayWaveform(inputChannel));
            setOutputWaveform(makeDisplayWaveform(outputChannel));
            setInputSpectrum(calculateSpectrum(inputChannel));
            setOutputSpectrum(calculateSpectrum(outputChannel));
            setStats({
                input: levelStats(inputChannel),
                output: levelStats(outputChannel),
                clipped: result.clippedSamples
            });
            setAnalyzed(true);
            setProcessing(false);
            setScanProgress(0);
            scanStartRef.current = performance.now();
        },
        [audioBuffer, modules, stopPlayback]
    );

    const handleFile = async (file) => {
        if (!file) return;

        stopPlayback();
        setProcessing(true);

        try {
            const context = createAudioContext();
            const arrayBuffer = await file.arrayBuffer();
            const decoded = await context.decodeAudioData(arrayBuffer);

            setAudioBuffer(decoded);
            setProcessedBuffer(null);
            setFileName(file.name);
            setAnalyzed(false);
            setProgress(0);
            setScanProgress(0);
            setInputWaveform(
                makeDisplayWaveform(decoded.getChannelData(0))
            );
            setOutputWaveform([]);
            setInputSpectrum([]);
            setOutputSpectrum([]);
            setStats({
                input: levelStats(decoded.getChannelData(0)),
                output: { peak: 0, rms: 0, db: -Infinity },
                clipped: 0
            });
        } catch (error) {
            console.error("Unable to decode audio:", error);
            alert("This audio file could not be decoded by the browser.");
        } finally {
            setProcessing(false);
        }
    };

    const handleFileChange = (event) => {
        handleFile(event.target.files?.[0]);
        event.target.value = "";
    };

    const handleUploadBoxClick = () => {
        fileInputRef.current?.click();
    };

    const updateModule = (key, value) => {
        setModules((current) => ({
            ...current,
            [key]: {
                ...current[key],
                value
            }
        }));
    };

    const toggleModule = (key) => {
        setModules((current) => ({
            ...current,
            [key]: {
                ...current[key],
                enabled: !current[key].enabled
            }
        }));
    };

    const toggleBypass = () => {
        setModules((current) => {
            const anyEnabled = Object.values(current).some(
                (module) => module.enabled
            );

            return Object.fromEntries(
                Object.entries(current).map(([key, module]) => [
                    key,
                    {
                        ...module,
                        enabled: !anyEnabled
                    }
                ])
            );
        });
    };

    useEffect(() => {
        if (!analyzed || !audioBuffer) return;

        window.clearTimeout(autoProcessTimerRef.current);

        autoProcessTimerRef.current = window.setTimeout(() => {
            processAndAnalyze(audioBuffer);
        }, 320);

        return () => window.clearTimeout(autoProcessTimerRef.current);
    }, [modules, analyzed, audioBuffer, processAndAnalyze]);

    useEffect(() => {
        let frame;

        const animate = (now) => {
            if (isPlaying && audioBuffer) {
                const context = audioContextRef.current;
                const elapsed =
                    context?.currentTime - playbackStartRef.current;

                const raw = Number.isFinite(elapsed)
                    ? elapsed
                    : 0;

                const next = duration
                    ? clamp(raw / duration, 0, 1)
                    : 0;

                setProgress(next);
            } else if (analyzed) {
                // When audio is not playing, the analysis cursor performs a
                // slow overview pass. One pass is approximately the real
                // audio duration, with a small minimum so short clips remain
                // readable. This prevents the cursor from racing across the
                // graph in a few seconds on long recordings.
                const scanDuration = Math.max(
                    12_000,
                    (duration || 1) * 1000
                );
                const elapsed =
                    (now - scanStartRef.current) / scanDuration;

                setScanProgress(elapsed % 1);
            }

            frame = requestAnimationFrame(animate);
        };

        frame = requestAnimationFrame(animate);

        return () => cancelAnimationFrame(frame);
    }, [analyzed, audioBuffer, duration, isPlaying]);

    useEffect(() => {
        if (!abMode || !isPlaying || !duration || !playMode) return;

        const switchDelay = Math.max(1800, Math.min(5000, duration * 1000));

        const timer = window.setTimeout(() => {
            const nextMode = playMode === "input" ? "output" : "input";
            const currentOffset = progress * duration;
            playBuffer(nextMode, currentOffset);
        }, switchDelay);

        return () => window.clearTimeout(timer);
    }, [
        abMode,
        isPlaying,
        playMode,
        duration,
        playBuffer
    ]);

    useEffect(() => {
        if (!expandedGraph) return;

        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";

        const onKeyDown = (event) => {
            if (event.key === "Escape") {
                setExpandedGraph(null);
            }
        };

        window.addEventListener("keydown", onKeyDown);

        return () => {
            document.body.style.overflow = previousOverflow;
            window.removeEventListener("keydown", onKeyDown);
        };
    }, [expandedGraph]);

    useEffect(() => {
        return () => {
            stopPlayback();
            window.clearTimeout(autoProcessTimerRef.current);
            audioContextRef.current?.close();
        };
    }, [stopPlayback]);

    const seek = (event) => {
        if (!duration) return;

        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = clamp(
            (event.clientX - rect.left) / rect.width,
            0,
            1
        );

        setProgress(ratio);

        if (isPlaying && playMode) {
            playBuffer(playMode, ratio * duration);
        }
    };

    const handlePlayInput = () => {
        if (!audioBuffer) return;

        if (isPlaying && playMode === "input") {
            stopPlayback();
            return;
        }

        playBuffer("input", progress * duration);
    };

    const handlePlayOutput = () => {
        if (!processedBuffer) return;

        if (isPlaying && playMode === "output") {
            stopPlayback();
            return;
        }

        playBuffer("output", progress * duration);
    };

    const activeStats = useMemo(
        () => ({
            input: stats.input,
            output: stats.output
        }),
        [stats]
    );

    const graphDefinitions = useMemo(
        () => [
            {
                key: "input-time",
                title: "INPUT · ORIGINAL",
                subtitle: "Time Domain / Amplitude",
                badge: "INPUT",
                data: inputWaveform,
                type: "time",
                color: "var(--input-accent)",
                playMode: "input"
            },
            {
                key: "output-time",
                title: "OUTPUT · PROCESSED",
                subtitle: "Time Domain / Amplitude",
                badge: "OUTPUT",
                data: outputWaveform,
                type: "time",
                color: "var(--output-accent)",
                playMode: "output"
            },
            {
                key: "input-spectrum",
                title: "INPUT · ORIGINAL",
                subtitle: "Frequency Domain / Discrete FFT",
                badge: "INPUT",
                data: inputSpectrum,
                type: "spectrum",
                color: "var(--input-accent)",
                playMode: "input"
            },
            {
                key: "output-spectrum",
                title: "OUTPUT · PROCESSED",
                subtitle: "Frequency Domain / Discrete FFT",
                badge: "OUTPUT",
                data: outputSpectrum,
                type: "spectrum",
                color: "var(--output-accent)",
                playMode: "output"
            }
        ],
        [inputWaveform, outputWaveform, inputSpectrum, outputSpectrum]
    );

    const expandedGraphData = expandedGraph
        ? graphDefinitions.find((graph) => graph.key === expandedGraph)
        : null;

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
                        <h1>Interactive Signal Studio</h1>
                        <p>
                            Analyze, shape and visualize your audio signal in
                            real time.
                        </p>
                    </div>

                    <button
                        type="button"
                        className="back-button"
                        onClick={() => {
                            window.location.hash = "";
                        }}
                    >
                        ← Back
                    </button>
                </header>

                <section
                    className="studio-workspace"
                    style={
                        analyzed
                            ? { gridTemplateColumns: "minmax(0, 1fr)" }
                            : undefined
                    }
                >
                    <div
                        className={`audio-panel ${
                            audioBuffer ? "has-audio" : ""
                        }`}
                    >
                        <div className="panel-label">AUDIO INPUT</div>

                        {!audioBuffer ? (
                            <button
                                type="button"
                                className="upload-zone"
                                onClick={handleUploadBoxClick}
                            >
                                <div className="upload-icon">＋</div>
                                <strong>Drop an audio file here</strong>
                                <span>
                                    or click anywhere to upload WAV, MP3,
                                    OGG or another browser-supported format
                                </span>
                            </button>
                        ) : (
                            <>
                                <div className="audio-file-row">
                                    <div>
                                        <strong>{fileName}</strong>
                                        <span>
                                            {analyzed
                                                ? "Analysis complete"
                                                : "Ready for analysis"}
                                        </span>
                                    </div>

                                    <button
                                        type="button"
                                        className="secondary-button"
                                        onClick={handleUploadBoxClick}
                                    >
                                        Replace
                                    </button>
                                </div>

                                <div className="main-waveform-wrap">
                                    <div className="main-waveform-meta">
                                        <span>ORIGINAL SIGNAL</span>
                                        <span>
                                            {audioBuffer
                                                ? `${audioBuffer.numberOfChannels} ch · ${Math.round(
                                                      audioBuffer.sampleRate
                                                  )} Hz`
                                                : ""}
                                        </span>
                                    </div>

                                    <GraphCanvas
                                        title=""
                                        subtitle=""
                                        badge=""
                                        data={inputWaveform}
                                        type="time"
                                        color="var(--input-accent)"
                                        cursorProgress={currentProgress}
                                        zoom={getGraphView("main").zoom}
                                        viewCenter={getGraphView("main").center}
                                        onZoomChange={(value) =>
                                            updateGraphView("main", {
                                                zoom: value
                                            })
                                        }
                                        onCenterChange={(value) =>
                                            updateGraphView("main", {
                                                center: value
                                            })
                                        }
                                        duration={duration}
                                    />
                                </div>

                                <div className="player-row">
                                    <div
                                        className="transport-track"
                                        onClick={seek}
                                        role="slider"
                                        tabIndex={0}
                                        aria-label="Audio position"
                                        aria-valuemin={0}
                                        aria-valuemax={100}
                                        aria-valuenow={Math.round(
                                            progress * 100
                                        )}
                                    >
                                        <div
                                            className="transport-fill"
                                            style={{
                                                width: `${progress * 100}%`
                                            }}
                                        />
                                        <div
                                            className="transport-cursor"
                                            style={{
                                                left: `${progress * 100}%`
                                            }}
                                        />
                                    </div>

                                    <span className="time-readout">
                                        {formatTime(progress * duration)}
                                        <span>/</span>
                                        {formatTime(duration)}
                                    </span>
                                </div>

                                <div className="player-options">
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={loopPlayback}
                                            onChange={(event) =>
                                                setLoopPlayback(
                                                    event.target.checked
                                                )
                                            }
                                        />
                                        <span>Loop</span>
                                    </label>

                                    <button
                                        type="button"
                                        className={`ab-button ${
                                            abMode ? "is-active" : ""
                                        }`}
                                        onClick={() => {
                                            setAbMode((current) => !current);
                                        }}
                                        disabled={!processedBuffer}
                                    >
                                        A/B
                                    </button>

                                    <span className="playback-hint">
                                        {isPlaying
                                            ? `Playing ${
                                                  playMode === "input"
                                                      ? "original"
                                                      : "processed"
                                              } audio`
                                            : analyzed
                                              ? "Analysis cursor active"
                                              : "Upload and analyze to begin"}
                                    </span>
                                </div>
                            </>
                        )}
                    </div>

                    {!analyzed && (
                        <aside className="controls-panel">
                                                    <div className="panel-label">PROCESSING CONTROLS</div>

                        <div className="controls-list">
                            {Object.keys(MODULE_META).map((key) => (
                                <ModuleControl
                                    key={key}
                                    moduleKey={key}
                                    config={modules[key]}
                                    onToggle={toggleModule}
                                    onChange={updateModule}
                                />
                            ))}
                        </div>

                        <div className="controls-actions">
                            <button
                                type="button"
                                className="bypass-button"
                                onClick={toggleBypass}
                                disabled={!audioBuffer}
                            >
                                BYPASS / ENABLE
                            </button>

                            <button
                                type="button"
                                className="process-button"
                                onClick={() => processAndAnalyze()}
                                disabled={!audioBuffer || processing}
                            >
                                {processing
                                    ? "Processing…"
                                    : analyzed
                                      ? "Re-process & Analyze"
                                      : "Analyze & Process"}
                            </button>
                        </div>

                        <div className="meter-grid">
                            <LevelMeter
                                label="INPUT LEVEL"
                                stats={activeStats.input}
                                clipped={0}
                            />
                            <LevelMeter
                                label="OUTPUT LEVEL"
                                stats={activeStats.output}
                                clipped={stats.clipped}
                            />
                        </div>
                    

                        </aside>
                    )}
                </section>

                <section className="analysis-section">
                    <div className="analysis-heading">
                        <div>
                            <div className="studio-eyebrow">SIGNAL ANALYSIS</div>
                            <h2>Input / Output Analysis</h2>
                            <p>
                                Compare the original signal with the processed
                                output while the playback cursor moves through
                                the file.
                            </p>
                        </div>

                        <div className="analysis-status">
                            <span
                                className={
                                    analyzed ? "status-dot active" : "status-dot"
                                }
                            />
                            {analyzed ? "ANALYZED" : "WAITING"}
                        </div>
                    </div>

                    {!analyzed ? (
                        <div className="analysis-empty">
                            <div className="empty-orb">∿</div>
                            <strong>
                                Your four analysis views will appear here
                            </strong>
                            <span>
                                Upload an audio file, choose your modules and
                                press Analyze &amp; Process.
                            </span>
                        </div>
                    ) : (
                        <>
                            <div className="signal-legend">
                                <span>
                                    <i className="legend-dot input" />
                                    ORIGINAL INPUT
                                </span>
                                <span>
                                    <i className="legend-dot output" />
                                    PROCESSED OUTPUT
                                </span>
                                <span className="legend-note">
                                    Scroll over a graph to zoom · click to
                                    reposition · double-click to reset
                                </span>
                            </div>

                            <div className="analysis-grid">
                                {graphDefinitions.map((graph) => (
                                    <GraphCanvas
                                        key={graph.key}
                                        title={graph.title}
                                        subtitle={graph.subtitle}
                                        badge={graph.badge}
                                        data={graph.data}
                                        type={graph.type}
                                        color={graph.color}
                                        cursorProgress={currentProgress}
                                        zoom={getGraphView(graph.key).zoom}
                                        viewCenter={getGraphView(graph.key).center}
                                        onZoomChange={(value) =>
                                            updateGraphView(graph.key, {
                                                zoom: value
                                            })
                                        }
                                        onCenterChange={(value) =>
                                            updateGraphView(graph.key, {
                                                center: value
                                            })
                                        }
                                        duration={duration}
                                        playMode={graph.playMode}
                                        playActive={
                                            isPlaying &&
                                            playMode === graph.playMode
                                        }
                                        onPlay={
                                            graph.playMode === "input"
                                                ? handlePlayInput
                                                : handlePlayOutput
                                        }
                                        onOpen={() =>
                                            setExpandedGraph(graph.key)
                                        }
                                    />
                                ))}
                            </div>

                            <div className="analysis-footer">
                                <span>
                                    Cursor{" "}
                                    <strong>
                                        {formatTime(
                                            currentProgress * duration
                                        )}
                                    </strong>
                                </span>

                                <span>
                                    Peak output{" "}
                                    <strong>
                                        {Math.round(
                                            stats.output.peak * 100
                                        )}
                                        %
                                    </strong>
                                </span>

                                <span>
                                    Clipped samples{" "}
                                    <strong>
                                        {stats.clipped.toLocaleString()}
                                    </strong>
                                </span>
                            </div>
                        </>
                    )}
                </section>

                {analyzed && (
                    <section
                        className="controls-panel"
                        style={{
                            marginTop: "14px"
                        }}
                    >
                        
                        <div className="panel-label">PROCESSING CONTROLS</div>

                        <div className="controls-list">
                            {Object.keys(MODULE_META).map((key) => (
                                <ModuleControl
                                    key={key}
                                    moduleKey={key}
                                    config={modules[key]}
                                    onToggle={toggleModule}
                                    onChange={updateModule}
                                />
                            ))}
                        </div>

                        <div className="controls-actions">
                            <button
                                type="button"
                                className="bypass-button"
                                onClick={toggleBypass}
                                disabled={!audioBuffer}
                            >
                                BYPASS / ENABLE
                            </button>

                            <button
                                type="button"
                                className="process-button"
                                onClick={() => processAndAnalyze()}
                                disabled={!audioBuffer || processing}
                            >
                                {processing
                                    ? "Processing…"
                                    : analyzed
                                      ? "Re-process & Analyze"
                                      : "Analyze & Process"}
                            </button>
                        </div>

                        <div className="meter-grid">
                            <LevelMeter
                                label="INPUT LEVEL"
                                stats={activeStats.input}
                                clipped={0}
                            />
                            <LevelMeter
                                label="OUTPUT LEVEL"
                                stats={activeStats.output}
                                clipped={stats.clipped}
                            />
                        </div>
                    
                    </section>
                )}

                <GraphModal
                    graph={expandedGraphData}
                    onClose={() => setExpandedGraph(null)}
                    zoom={
                        expandedGraphData
                            ? getGraphView(expandedGraphData.key).zoom
                            : 1
                    }
                    viewCenter={
                        expandedGraphData
                            ? getGraphView(expandedGraphData.key).center
                            : 0.5
                    }
                    onZoomChange={(value) => {
                        if (expandedGraphData) {
                            updateGraphView(expandedGraphData.key, {
                                zoom: value
                            });
                        }
                    }}
                    onCenterChange={(value) => {
                        if (expandedGraphData) {
                            updateGraphView(expandedGraphData.key, {
                                center: value
                            });
                        }
                    }}
                    cursorProgress={currentProgress}
                    duration={duration}
                    playActive={
                        isPlaying &&
                        expandedGraphData?.playMode === playMode
                    }
                    onPlay={
                        expandedGraphData?.playMode === "input"
                            ? handlePlayInput
                            : expandedGraphData?.playMode === "output"
                              ? handlePlayOutput
                              : null
                    }
                />
            </div>
        </main>
    );
}
