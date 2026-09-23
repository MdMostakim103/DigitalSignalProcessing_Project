import { useEffect, useRef, useState } from "react";
import "../../styles/time-domain.css";
import { getFilterResponse } from "../../services/api";

// A standalone |H(f)| line — the filter's own shape, independent of any
// audio. Debounced-fetches the backend's real freqz() response every time a
// slider changes, so it updates live as the user tunes the filter, well
// before they ever press "run".
export default function FilterResponseGraph({ filterFamily, bandType, cutoff, cutoff2, order, sampleRate }) {
    const [data, setData] = useState(null);
    const [failed, setFailed] = useState(false);
    const debounceRef = useRef(null);
    const requestIdRef = useRef(0);

    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        const thisRequest = ++requestIdRef.current;

        debounceRef.current = setTimeout(async () => {
            try {
                const result = await getFilterResponse({ filterFamily, bandType, cutoff, cutoff2, order, sampleRate });
                if (requestIdRef.current === thisRequest) {
                    setData(result);
                    setFailed(false);
                }
            } catch (err) {
                console.error(err);
                if (requestIdRef.current === thisRequest) setFailed(true);
            }
        }, 150);

        return () => clearTimeout(debounceRef.current);
    }, [filterFamily, bandType, cutoff, cutoff2, order, sampleRate]);

    if (failed) {
        return <div className="spectrum-empty">Could not reach the backend for the filter preview.</div>;
    }
    if (!data || !data.frequencies?.length) {
        return <div className="spectrum-empty">Loading filter response…</div>;
    }

    const { frequencies, magnitude, maxFrequency } = data;

    const width = 900;
    const height = 220;
    const pad = { left: 50, right: 20, top: 16, bottom: 28 };
    const plotTop = pad.top;
    const plotBottom = height - pad.bottom;
    const plotLeft = pad.left;
    const plotRight = width - pad.right;

    const toX = (f) => plotLeft + (f / Math.max(1, maxFrequency)) * (plotRight - plotLeft);
    const toY = (m) => plotBottom - Math.min(1, Math.max(0, m)) * (plotBottom - plotTop);

    const path = frequencies
        .map((f, i) => `${i === 0 ? "M" : "L"} ${toX(f).toFixed(1)} ${toY(magnitude[i]).toFixed(1)}`)
        .join(" ");

    const cutoffMarkers = bandType === "bandpass" || bandType === "bandstop" ? [cutoff, cutoff2] : [cutoff];

    return (
        <div className="filter-response-graph" style={{ width: "100%", height: `${height}px` }}>
            <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                    <line key={f} x1={plotLeft} x2={plotRight} y1={toY(f)} y2={toY(f)} className="result-grid" />
                ))}
                {cutoffMarkers.map((c, i) => (
                    <line
                        key={i}
                        x1={toX(c)} x2={toX(c)} y1={plotTop} y2={plotBottom}
                        stroke="rgba(255,255,255,.35)" strokeDasharray="4 4" vectorEffect="non-scaling-stroke"
                    />
                ))}
                <path d={path} fill="none" stroke="#f7b801" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
                {[0, 0.5, 1].map((f) => (
                    <text
                        key={f} x={plotLeft - 8} y={toY(f)} fill="rgba(255,255,255,.45)" fontSize="10"
                        fontFamily="ui-monospace, monospace" textAnchor="end" dominantBaseline="middle"
                    >
                        {f.toFixed(1)}
                    </text>
                ))}
                <text x={plotLeft} y={height - 8} fill="rgba(255,255,255,.45)" fontSize="10" fontFamily="ui-monospace, monospace" textAnchor="start">
                    0 Hz
                </text>
                <text x={plotRight} y={height - 8} fill="rgba(255,255,255,.45)" fontSize="10" fontFamily="ui-monospace, monospace" textAnchor="end">
                    {Math.round(maxFrequency).toLocaleString()} Hz
                </text>
            </svg>
        </div>
    );
}
