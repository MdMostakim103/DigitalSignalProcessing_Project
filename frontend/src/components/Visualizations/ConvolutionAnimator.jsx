import React, { useState, useEffect, useMemo, useRef } from 'react';
import '../../styles/time-domain.css';

function extractSignal(buffer, stride, maxPoints) {
    if (!buffer || buffer.length === 0) return new Float32Array(0);
    const len = Math.min(maxPoints, Math.ceil(buffer.length / stride));
    const out = new Float32Array(len);
    let globalPeak = 0.0001;
    
    for (let i = 0; i < len; i++) {
        let maxVal = 0;
        const start = Math.floor(i * stride);
        const end = Math.min(Math.floor((i + 1) * stride), buffer.length);
        
        for (let j = start; j < end; j++) {
            const absVal = Math.abs(buffer[j]);
            if (absVal > maxVal) maxVal = absVal;
        }
        out[i] = maxVal;
        if (maxVal > globalPeak) globalPeak = maxVal;
    }
    
    for (let i = 0; i < out.length; i++) out[i] = out[i] / globalPeak;
    return out;
}

export default function ConvolutionAnimator({ inputBuffer, irBuffer, onComplete }) {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const timerRef = useRef(null);

    const TARGET_N = 80;
    const TARGET_M = 40;
    const MIN_X = -40;
    const MAX_X = 120;
    const TOTAL_GRID_POINTS = MAX_X - MIN_X;

    const { x, h, N, M } = useMemo(() => {
        if (!inputBuffer || !irBuffer) return { x: new Float32Array(), h: new Float32Array(), N: 0, M: 0 };
        const strideX = Math.max(1, Math.floor(inputBuffer.length / TARGET_N));
        const decX = extractSignal(inputBuffer, strideX, TARGET_N);
        const strideH = Math.max(1, Math.floor(irBuffer.length / TARGET_M));
        const decH = extractSignal(irBuffer, strideH, TARGET_M); 
        return { x: decX, h: decH, N: decX.length, M: decH.length };
    }, [inputBuffer, irBuffer]);

    const TOTAL_STEPS = N > 0 && M > 0 ? N + M - 1 : 0;

    const y = useMemo(() => {
        if (TOTAL_STEPS === 0) return new Float32Array();
        const out = new Float32Array(TOTAL_STEPS);
        let peakY = 0.0001;
        
        for (let n = 0; n < TOTAL_STEPS; n++) {
            let sum = 0;
            for (let k = 0; k < N; k++) {
                if (n - k >= 0 && n - k < M) sum += x[k] * h[n - k];
            }
            out[n] = sum;
            if (sum > peakY) peakY = sum;
        }
        for (let i = 0; i < out.length; i++) out[i] /= peakY;
        return out;
    }, [x, h, N, M, TOTAL_STEPS]);

    // Animation Loop + Trigger Completion Modal when it reaches the end
    useEffect(() => {
        if (isPlaying && TOTAL_STEPS > 0) {
            timerRef.current = setInterval(() => {
                setStep(prev => {
                    if (prev >= TOTAL_STEPS - 1) {
                        setIsPlaying(false);
                        if (onComplete) onComplete(); // Trigger modal pop up!
                        return prev;
                    }
                    return prev + 1; 
                });
            }, 60);
        } else {
            clearInterval(timerRef.current);
        }
        return () => clearInterval(timerRef.current);
    }, [isPlaying, TOTAL_STEPS, onComplete]);

    const svgWidth = 800;
    const svgHeight = 140; 
    const midY = 115;       
    const scaleY = 95; 
    
    const paddingLeft = 30; 
    const paddingRight = 20;
    const spacing = (svgWidth - paddingLeft - paddingRight) / TOTAL_GRID_POINTS;
    const getX = (index) => paddingLeft + (index - MIN_X) * spacing;

    const renderAxis = () => {
        const ticks = [];
        for (let i = MIN_X; i <= MAX_X; i += 10) {
            const cx = getX(i);
            ticks.push(
                <g key={`tick-${i}`}>
                    <line x1={cx} y1={midY - 4} x2={cx} y2={midY + 4} stroke="rgba(255,255,255,0.3)" />
                    <text x={cx} y={midY + 16} fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="monospace" textAnchor="middle">
                        {i}
                    </text>
                </g>
            );
        }
        return (
            <g>
                <line x1="0" y1={midY} x2={svgWidth} y2={midY} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                {ticks}
                <line x1={getX(0)} y1="10" x2={getX(0)} y2={midY} stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="4 4" />
            </g>
        );
    };

    if (N === 0) return null;

    return (
        <div className="convolution-animator-container">
            <div className="animator-warning-banner">
                <div className="warning-icon">⚠</div>
                <div>
                    <strong>DEMONSTRATION MODE ONLY</strong>
                    <p>Animation running with <strong>{N} input points</strong> and <strong>{M} IR points</strong>. Audio computed via backend FFT.</p>
                </div>
            </div>

            {/* INPUT SIGNAL x[k] */}
            <div className="animator-plot">
                <div className="animator-plot-header">INPUT SIGNAL x[k]</div>
                <svg width="100%" height={`${svgHeight}px`} viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="none">
                    {renderAxis()}
                    {Array.from(x).map((val, k) => {
                        const cx = getX(k);
                        const cy = midY - val * scaleY;
                        const isOverlapping = k >= step - M + 1 && k <= step;
                        const color = isOverlapping ? "#ffffff" : "#4ea1ff"; 
                        return (
                            <g key={`x-${k}`}>
                                <line x1={cx} y1={midY} x2={cx} y2={cy} stroke={color} strokeWidth={isOverlapping ? "2" : "1.5"} opacity={isOverlapping ? "1" : "0.5"} />
                                <circle cx={cx} cy={cy} r={isOverlapping ? "3" : "2"} fill={color} />
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* REVERSED IR h[n-k] */}
            <div className="animator-plot">
                <div className="animator-plot-header">REVERSED IMPULSE RESPONSE h[n-k]</div>
                <svg width="100%" height={`${svgHeight}px`} viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="none">
                    {renderAxis()}
                    {Array.from(h).map((val, m) => {
                        const k = step - m;
                        if (k < MIN_X || k > MAX_X) return null;
                        const cx = getX(k);
                        const cy = midY - val * scaleY;
                        const isOverlapping = k >= 0 && k < N;
                        const color = isOverlapping ? "#ffffff" : "#f7b801"; 
                        return (
                            <g key={`h-${m}`}>
                                <line x1={cx} y1={midY} x2={cx} y2={cy} stroke={color} strokeWidth={isOverlapping ? "2" : "1.5"} opacity={isOverlapping ? "1" : "0.5"} />
                                <circle cx={cx} cy={cy} r={isOverlapping ? "3" : "2"} fill={color} />
                            </g>
                        );
                    })}
                </svg>
            </div>
            
            {/* OUTPUT y[n] */}
            <div className="animator-plot">
                <div className="animator-plot-header">OUTPUT y[n]</div>
                <svg width="100%" height={`${svgHeight}px`} viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="none">
                    {renderAxis()}
                    {Array.from(y).map((val, n) => {
                        if (n > step) return null; 
                        const cx = getX(n);
                        const cy = midY - val * scaleY;
                        const isCurrent = n === step;
                        const color = isCurrent ? "#ffffff" : "#ff5d6c"; 
                        return (
                            <g key={`y-${n}`}>
                                <line x1={cx} y1={midY} x2={cx} y2={cy} stroke={color} strokeWidth={isCurrent ? "2" : "1.5"} opacity={isCurrent ? "1" : "0.7"} />
                                <circle cx={cx} cy={cy} r={isCurrent ? "3" : "2"} fill={color} />
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* TRANSPORT CONTROLS */}
            <div className="animator-transport">
                <button className="anim-btn" onClick={() => setIsPlaying(!isPlaying)}>
                    {isPlaying ? '⏸ PAUSE' : '▶ PLAY'}
                </button>
                <button className="anim-btn" onClick={() => { setStep(0); setIsPlaying(false); }}>
                    ↺ RESET
                </button>
                <div className="anim-scrubber">
                    <input 
                        type="range" min="0" max={TOTAL_STEPS - 1} value={step} 
                        onChange={(e) => { setStep(Number(e.target.value)); setIsPlaying(false); }} 
                        className="range-input"
                    />
                    <span style={{ minWidth: '80px', textAlign: 'right' }}>n = {step}</span>
                </div>
            </div>
        </div>
    );
}