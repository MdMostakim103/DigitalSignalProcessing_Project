import { useEffect, useRef } from "react";

function ProcessedWaveform({ audio, currentTime, onSeek, audioElement }) {
    const canvasRef = useRef(null);
    const waveformDataRef = useRef(null);
    const durationRef = useRef(0);
    const animationRef = useRef(null);

    useEffect(() => {
        if (!audio) return;
        let cancelled = false;

        const loadAudio = async () => {
            try {
                const response = await fetch(audio.src);
                const arrayBuffer = await response.arrayBuffer();
                const context = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await context.decodeAudioData(arrayBuffer);

                if (cancelled) {
                    await context.close();
                    return;
                }

                waveformDataRef.current = audioBuffer.getChannelData(0);
                durationRef.current = audioBuffer.duration;
                await context.close();
                drawStatic();
            } catch (error) {
                console.error("Could not decode processed audio:", error);
            }
        };

        loadAudio();
        return () => {
            cancelled = true;
            cancelAnimationFrame(animationRef.current);
        };
    }, [audio?.src]);

    useEffect(() => {
        const animate = () => {
            if (audioElement && !audioElement.paused && !audioElement.ended) {
                drawLiveWindow(audioElement.currentTime);
            } else {
                drawStatic();
            }
            animationRef.current = requestAnimationFrame(animate);
        };

        cancelAnimationFrame(animationRef.current);
        animationRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animationRef.current);
    }, [audioElement]);

    useEffect(() => {
        if (!audioElement || !audioElement.paused) return;
        drawStatic();
    }, [currentTime, audioElement]);

    const drawStatic = () => {
        const data = waveformDataRef.current;
        const canvas = canvasRef.current;
        if (!data || !canvas) return;
        const ctx = canvas.getContext("2d");
        drawBackground(ctx, canvas);
        drawWaveform(data, ctx, canvas);
        drawPlayhead(ctx, canvas, currentTime, durationRef.current);
    };

    const drawLiveWindow = (time) => {
        const data = waveformDataRef.current;
        const canvas = canvasRef.current;
        const duration = durationRef.current;
        if (!data || !canvas || !duration) return;

        const ctx = canvas.getContext("2d");
        drawBackground(ctx, canvas);

        const samplesPerSecond = data.length / duration;
        const windowSamples = Math.max(256, Math.floor(samplesPerSecond * 0.08));
        const centerSample = Math.floor(time * samplesPerSecond);
        const start = Math.max(0, Math.min(data.length - windowSamples, centerSample - Math.floor(windowSamples / 2)));

        ctx.beginPath();
        for (let x = 0; x < canvas.width; x += 2) {
            const index = start + Math.floor((x / canvas.width) * windowSamples);
            const value = data[Math.min(index, data.length - 1)] || 0;
            const y = canvas.height / 2 - value * canvas.height * 0.42;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.strokeStyle = "#9b5cff";
        ctx.lineWidth = 2;
        ctx.stroke();
        drawPlayhead(ctx, canvas, time, duration);
    };

    const handleCanvasClick = (event) => {
        const canvas = canvasRef.current;
        if (!canvas || !durationRef.current) return;

        const rect = canvas.getBoundingClientRect();
        const progress = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        onSeek(progress * durationRef.current);
    };

    return (
        <div className="waveform-container">
            <div className="waveform-header">
                <span>{audioElement ? "LIVE PROCESSED SIGNAL" : "PROCESSED SIGNAL"}</span>
                <span>{formatTime(currentTime)}</span>
            </div>

            <canvas
                ref={canvasRef}
                width={1000}
                height={300}
                className="waveform-canvas"
                onClick={handleCanvasClick}
            />

            <div className="waveform-hint">
                {audioElement
                    ? "PLAY THE RESULT • processed waveform moves with the signal"
                    : "CLICK THE WAVEFORM TO SEEK"}
            </div>
        </div>
    );
}

function drawBackground(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.stroke();
}

function drawWaveform(data, ctx, canvas) {
    const step = Math.max(1, Math.ceil(data.length / canvas.width));
    ctx.beginPath();

    for (let x = 0; x < canvas.width; x++) {
        const start = x * step;
        let min = 1;
        let max = -1;

        for (let i = 0; i < step && start + i < data.length; i++) {
            const value = data[start + i];
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        const center = canvas.height / 2;
        const displayHeight = canvas.height * 0.8;
        const yMin = center - min * displayHeight / 2;
        const yMax = center - max * displayHeight / 2;

        if (x === 0) ctx.moveTo(x, yMin);
        else ctx.lineTo(x, yMin);
        ctx.lineTo(x, yMax);
    }

    ctx.strokeStyle = "#9b5cff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

function drawPlayhead(ctx, canvas, currentTime, duration) {
    if (!duration) return;
    const x = Math.max(0, Math.min(1, currentTime / duration)) * canvas.width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 2;
    ctx.stroke();
}

function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const minutes = Math.floor(seconds / 60);
    const remaining = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${remaining}`;
}

export default ProcessedWaveform;
