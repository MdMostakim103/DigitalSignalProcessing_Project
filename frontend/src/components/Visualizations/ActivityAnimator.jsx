import { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/time-domain.css";

function clamp(v, a = 0, b = 1) {
    return Math.min(b, Math.max(a, v));
}

// A crop long enough that the min-speech/min-silence cleanup pass (tens of
// ms each) has more than a couple of demo frames to actually operate on —
// at 100 raw samples/frame this is roughly 250ms of real audio, ~119 demo
// frames. Too short a crop (an earlier version used 3000 samples, ~29
// frames) makes 60ms-scale cleanup swallow the entire demo instead of
// visibly merging/removing a few segments.
const CROP_SAMPLES = 12000;
const WINDOW_SIZE = 200;
const HOP_SIZE = 100;

const NOISE_FLOOR_PERCENTILE = 20;
const SIGNAL_LEVEL_PERCENTILE = 95;
const HYSTERESIS_RATIO = 0.5;
const MIN_SPEECH_MS = 60;
const MIN_SILENCE_MS = 60;

// A real, UNMODIFIED excerpt — not a downsampled/averaged version of the
// whole file. This matters more here than it does for convolution: bucket-
// averaging thousands of raw samples into one point was tried first, and it
// destroyed the very thing ZCR measures. Averaging thousands of raw samples
// into a single point acts like a crude filter that changes what "crossing
// zero" even means — measured on real files, it pushed EVERY signal (clean
// speech included) up to 55-74% ZCR, nowhere near the 1-5% clean speech
// actually has at full resolution. A short raw crop has no such distortion.
//
// The crop is taken from the loudest region of the signal (a coarse energy
// scan), so a quiet file doesn't hand the demo a silent excerpt to frame.
function findLoudestCrop(buffer, cropLen) {
    if (!buffer || buffer.length === 0) return new Float32Array(0);
    if (buffer.length <= cropLen) return Float32Array.from(buffer);

    const scanStep = Math.max(1, Math.floor(cropLen / 3));
    let bestStart = 0;
    let bestEnergy = -1;
    for (let s = 0; s + cropLen <= buffer.length; s += scanStep) {
        let sumSq = 0;
        for (let j = s; j < s + cropLen; j += 4) sumSq += buffer[j] * buffer[j];
        if (sumSq > bestEnergy) {
            bestEnergy = sumSq;
            bestStart = s;
        }
    }

    const crop = new Float32Array(cropLen);
    for (let i = 0; i < cropLen; i++) crop[i] = buffer[bestStart + i];

    let peak = 1e-9;
    for (let i = 0; i < crop.length; i++) peak = Math.max(peak, Math.abs(crop[i]));
    for (let i = 0; i < crop.length; i++) crop[i] /= peak;
    return crop;
}

function frameEnergy(window) {
    let sumSq = 0;
    for (let i = 0; i < window.length; i++) sumSq += window[i] * window[i];
    return Math.sqrt(sumSq / window.length);
}

function frameZcr(window) {
    if (window.length < 2) return 0;
    let crossings = 0;
    let prevSign = window[0] >= 0 ? 1 : -1;
    for (let i = 1; i < window.length; i++) {
        const sign = window[i] >= 0 ? 1 : -1;
        if (sign !== prevSign) crossings++;
        prevSign = sign;
    }
    return crossings / (window.length - 1);
}

// Linear-interpolation percentile, matching numpy's default so the demo's
// noise-floor/signal-level estimates use the same definition the backend
// does — not just "a robust estimate," the SAME one.
function percentile(values, p) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    const frac = idx - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

// Bridge short silence gaps, then drop speech islands still too short to be
// a real syllable — the same two-pass cleanup the backend runs, over the
// demo's own (much shorter) frame sequence.
function applyMinDuration(active, hopSeconds, minSpeechMs, minSilenceMs) {
    const n = active.length;
    if (!n) return active;
    const minSpeechFrames = Math.max(1, Math.round(minSpeechMs / 1000 / hopSeconds));
    const minSilenceFrames = Math.max(1, Math.round(minSilenceMs / 1000 / hopSeconds));
    const out = active.slice();

    let i = 0;
    while (i < n) {
        if (!out[i]) {
            let j = i;
            while (j < n && !out[j]) j++;
            const gapLen = j - i;
            const flanked = i > 0 && j < n;
            if (flanked && gapLen < minSilenceFrames) {
                for (let k = i; k < j; k++) out[k] = true;
            }
            i = j;
        } else {
            i++;
        }
    }

    i = 0;
    while (i < n) {
        if (out[i]) {
            let j = i;
            while (j < n && out[j]) j++;
            if (j - i < minSpeechFrames) {
                for (let k = i; k < j; k++) out[k] = false;
            }
            i = j;
        } else {
            i++;
        }
    }

    return out;
}

// The "process" view for Module 4. Mirrors ConvolutionAnimator's approach
// rather than replaying a precomputed backend array: it crops a real,
// unmodified excerpt of the decoded audio, then runs the ACTUAL pipeline —
// framing, energy (+ ZCR), noise-floor-relative hysteresis, minimum-duration
// cleanup — live, in the browser, on that excerpt. Same algorithm as
// apply_activity_gate() on the backend, just on far fewer samples.
export default function ActivityAnimator({
    inputBuffer,
    sampleRate = 44100,
    method = "energy",
    energyThresholdRatio = 0.15,
    zcrThreshold = 0.05,
    onComplete,
    loop = false,
}) {
    const isZcr = method === "energy_zcr";

    const {
        waveform, energy, zcr, active, enterThreshold, exitThreshold, noiseFloor, frameStarts,
    } = useMemo(() => {
        const wave = findLoudestCrop(inputBuffer, CROP_SAMPLES);
        const n = wave.length;
        if (n < WINDOW_SIZE) {
            return {
                waveform: wave, energy: [], zcr: [], active: [],
                enterThreshold: 0, exitThreshold: 0, noiseFloor: 0, frameStarts: [],
            };
        }

        const starts = [];
        for (let start = 0; start + WINDOW_SIZE <= n; start += HOP_SIZE) starts.push(start);

        const e = starts.map((s) => frameEnergy(wave.slice(s, s + WINDOW_SIZE)));
        const z = starts.map((s) => frameZcr(wave.slice(s, s + WINDOW_SIZE)));

        const nFloor = percentile(e, NOISE_FLOOR_PERCENTILE);
        const signalLevel = percentile(e, SIGNAL_LEVEL_PERCENTILE);
        const headroom = Math.max(0, signalLevel - nFloor);
        const enterThresh = nFloor + energyThresholdRatio * headroom;
        const exitThresh = nFloor + energyThresholdRatio * HYSTERESIS_RATIO * headroom;

        const act = new Array(e.length).fill(false);
        let state = false;
        for (let i = 0; i < e.length; i++) {
            if (!state) {
                let enters = e[i] > enterThresh;
                if (isZcr) enters = enters && z[i] < zcrThreshold;
                state = enters;
            } else {
                state = e[i] >= exitThresh;
            }
            act[i] = state;
        }

        const hopSeconds = HOP_SIZE / sampleRate;
        const cleaned = applyMinDuration(act, hopSeconds, MIN_SPEECH_MS, MIN_SILENCE_MS);

        return {
            waveform: wave, energy: e, zcr: z, active: cleaned,
            enterThreshold: enterThresh, exitThreshold: exitThresh, noiseFloor: nFloor, frameStarts: starts,
        };
    }, [inputBuffer, isZcr, energyThresholdRatio, zcrThreshold, sampleRate]);

    const count = frameStarts.length;

    const [progress, setProgress] = useState(0);
    const onCompleteRef = useRef(onComplete);
    useEffect(() => { onCompleteRef.current = onComplete; });

    useEffect(() => {
        if (!count) return undefined;
        let raf;
        let start = performance.now();
        let completedOnce = false;
        const duration = 6200;

        const tick = (now) => {
            const p = clamp((now - start) / duration);
            setProgress(p);

            if (p >= 1) {
                if (!completedOnce) {
                    completedOnce = true;
                    onCompleteRef.current?.();
                }
                if (loop) {
                    start = now + 900;
                    completedOnce = false;
                    setProgress(0);
                    raf = requestAnimationFrame(tick);
                }
                return;
            }
            raf = requestAnimationFrame(tick);
        };

        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [count, loop]);

    if (!count) {
        return <div className="spectrum-empty">Upload a signal first — the process view needs decoded audio to frame.</div>;
    }

    const revealed = Math.max(1, Math.floor(progress * count));
    const currentFrame = Math.min(count - 1, revealed - 1);

    // Layout
    const width = 900;
    const waveHeight = 110;
    const barHeight = 110;
    const pad = { left: 46, right: 20 };
    const plotLeft = pad.left;
    const plotRight = width - pad.right;
    const waveMidY = waveHeight / 2;

    const toX = (i, n) => plotLeft + (i / Math.max(1, n - 1)) * (plotRight - plotLeft);

    const wavePoints = Array.from(waveform)
        .map((v, i) => `${toX(i, waveform.length).toFixed(1)},${(waveMidY - v * waveMidY * 0.85).toFixed(1)}`)
        .join(" ");

    const barX = (i) => plotLeft + ((i + 0.5) / count) * (plotRight - plotLeft);
    const barWidth = Math.max(2, (plotRight - plotLeft) / count - 2);

    const maxEnergy = Math.max(...energy, 1e-9);
    const energyBarY = (v) => barHeight - clamp(v / maxEnergy) * (barHeight - 10);
    const enterThresholdY = barHeight - clamp(enterThreshold / maxEnergy) * (barHeight - 10);
    const exitThresholdY = barHeight - clamp(exitThreshold / maxEnergy) * (barHeight - 10);
    const noiseFloorY = barHeight - clamp(noiseFloor / maxEnergy) * (barHeight - 10);

    const zcrBarY = (v) => barHeight - clamp(v / 1) * (barHeight - 10); // ZCR is already 0..1
    const zcrThresholdY = barHeight - clamp(zcrThreshold) * (barHeight - 10);

    const windowStart = frameStarts[currentFrame] ?? 0;
    const windowEnd = windowStart + WINDOW_SIZE;
    const windowX1 = toX(windowStart, waveform.length);
    const windowX2 = toX(windowEnd - 1, waveform.length);

    const activeCount = active.filter(Boolean).length;

    return (
        <div className="activity-animator">
            <div className="animator-warning-banner">
                <div className="warning-icon">⚠</div>
                <div>
                    <strong>DEMONSTRATION MODE — REAL SAMPLES, SAME PIPELINE</strong>
                    <p>
                        A real, unmodified {waveform.length}-sample excerpt from the loudest part of the signal,
                        framed into {count} windows of {WINDOW_SIZE} samples ({HOP_SIZE}-sample hop). Energy
                        {isZcr ? " and ZCR are" : " is"} measured, then classified with the same noise-floor-relative
                        hysteresis and minimum-duration cleanup the backend uses — not a simplified per-frame rule.
                        The actual gated audio uses the full signal from the backend.
                    </p>
                </div>
            </div>

            <div className="effect-stage" style={{ gridColumn: "1 / -1" }}>
                <div className="effect-stage-label">
                    <span>WAVEFORM — SLIDING WINDOW</span>
                    <small>the {WINDOW_SIZE}-sample window currently being measured</small>
                </div>
                <svg viewBox={`0 0 ${width} ${waveHeight}`} preserveAspectRatio="none">
                    <line x1={plotLeft} x2={plotRight} y1={waveMidY} y2={waveMidY} className="conv-axis" />
                    <rect
                        x={windowX1} y={2}
                        width={Math.max(4, windowX2 - windowX1)} height={waveHeight - 4}
                        fill="rgba(247,184,1,.16)" stroke="rgba(247,184,1,.6)" strokeWidth="1.5"
                    />
                    <polyline points={wavePoints} fill="none" stroke="#4ea1ff" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                </svg>
            </div>

            <div className="effect-stage" style={{ gridColumn: "1 / -1", marginTop: "10px" }}>
                <div className="effect-stage-label">
                    <span>SHORT-TIME ENERGY per WINDOW</span>
                    <small>solid line = enter threshold · dashed = exit threshold (hysteresis)</small>
                </div>
                <svg viewBox={`0 0 ${width} ${barHeight}`} preserveAspectRatio="none">
                    <line x1={plotLeft} x2={plotRight} y1={barHeight} y2={barHeight} className="conv-axis" />
                    <line
                        x1={plotLeft} x2={plotRight} y1={noiseFloorY} y2={noiseFloorY}
                        stroke="rgba(255,255,255,.3)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke"
                    />
                    <line
                        x1={plotLeft} x2={plotRight} y1={exitThresholdY} y2={exitThresholdY}
                        stroke="rgba(255,93,108,.4)" strokeDasharray="5 5" vectorEffect="non-scaling-stroke"
                    />
                    <line
                        x1={plotLeft} x2={plotRight} y1={enterThresholdY} y2={enterThresholdY}
                        stroke="rgba(255,93,108,.75)" vectorEffect="non-scaling-stroke"
                    />
                    {energy.map((v, i) => {
                        if (i > currentFrame) return null;
                        const y = energyBarY(v);
                        return (
                            <rect
                                key={i}
                                x={barX(i) - barWidth / 2} y={y}
                                width={barWidth} height={Math.max(1, barHeight - y)}
                                fill={active[i] ? "#3ddc84" : "rgba(255,255,255,.18)"}
                                opacity={i === currentFrame ? 1 : 0.85}
                            />
                        );
                    })}
                </svg>
            </div>

            {isZcr && (
                <div className="effect-stage" style={{ gridColumn: "1 / -1", marginTop: "10px" }}>
                    <div className="effect-stage-label">
                        <span>ZERO-CROSSING RATE per WINDOW</span>
                        <small>only gates ENTERING active — not required to stay active</small>
                    </div>
                    <svg viewBox={`0 0 ${width} ${barHeight}`} preserveAspectRatio="none">
                        <line x1={plotLeft} x2={plotRight} y1={barHeight} y2={barHeight} className="conv-axis" />
                        <line
                            x1={plotLeft} x2={plotRight} y1={zcrThresholdY} y2={zcrThresholdY}
                            stroke="rgba(182,108,255,.65)" strokeDasharray="5 5" vectorEffect="non-scaling-stroke"
                        />
                        {zcr.map((v, i) => {
                            if (i > currentFrame) return null;
                            const y = zcrBarY(v);
                            return (
                                <rect
                                    key={i}
                                    x={barX(i) - barWidth / 2} y={y}
                                    width={barWidth} height={Math.max(1, barHeight - y)}
                                    fill={active[i] ? "#3ddc84" : "#b66cff"}
                                    opacity={i === currentFrame ? 1 : 0.7}
                                />
                            );
                        })}
                    </svg>
                </div>
            )}

            <div className="effect-operation" style={{ marginTop: "12px" }}>
                <span>{isZcr ? "ENERGY + ZCR VAD, WITH HYSTERESIS" : "SHORT-TIME ENERGY VAD, WITH HYSTERESIS"}</span>
                <strong>
                    {isZcr
                        ? "enter: STE > E_enter AND ZCR < Z_thresh  ·  stay active until STE < E_exit"
                        : "enter: STE > E_enter  ·  stay active until STE < E_exit"}
                </strong>
                <small style={{ color: "rgba(255,255,255,.5)", fontSize: "10px", letterSpacing: ".5px" }}>
                    {count} demo windows · {activeCount}/{count} active after hysteresis + min-duration cleanup
                </small>
            </div>

            <div className="frequency-progress">
                <div>
                    <span>{isZcr ? "Measuring energy and zero-crossings per window…" : "Framing the signal and measuring energy…"}</span>
                    <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="frequency-progress-track"><span style={{ width: `${progress * 100}%` }} /></div>
            </div>
        </div>
    );
}
