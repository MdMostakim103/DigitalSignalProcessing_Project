import { useEffect, useRef, useState } from "react";
import { encodeWav } from "../../utils/wavEncoder";

// Manually start/stop microphone capture, entirely in the browser: a
// ScriptProcessor tap collects raw PCM as it arrives (no lossy codec in
// between — the same technique the Bird Sound Detector uses for its fixed-
// length capture, generalized here into a start/stop toggle any module can
// offer as a live alternative to picking a file). On stop, the captured
// samples are merged and encoded to a plain WAV File via encodeWav, then
// handed to onRecordingComplete exactly like a chosen file would be.
function MicRecordButton({ onRecordingComplete, onError, disabled = false, className = "" }) {
    const [state, setState] = useState("idle"); // idle | requesting | recording
    const [elapsed, setElapsed] = useState(0);

    const streamRef = useRef(null);
    const audioCtxRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const chunksRef = useRef([]);
    const timerRef = useRef(null);

    const cleanup = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = null;
        processorRef.current?.disconnect();
        sourceRef.current?.disconnect();
        streamRef.current?.getTracks().forEach((track) => track.stop());
        if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
            audioCtxRef.current.close();
        }
        streamRef.current = null;
        audioCtxRef.current = null;
        processorRef.current = null;
        sourceRef.current = null;
    };

    useEffect(() => () => cleanup(), []);

    const startRecording = async () => {
        if (disabled || state !== "idle") return;
        setState("requesting");

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error(err);
            onError?.("Microphone access was blocked or is unavailable in this browser.");
            setState("idle");
            return;
        }

        streamRef.current = stream;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        sourceRef.current = source;

        const bufferSize = 4096;
        const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
        processorRef.current = processor;
        chunksRef.current = [];

        processor.onaudioprocess = (e) => {
            chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);

        setElapsed(0);
        timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
        setState("recording");
    };

    const stopRecording = () => {
        if (state !== "recording") return;

        const audioCtx = audioCtxRef.current;
        const sampleRate = audioCtx.sampleRate;
        const chunks = chunksRef.current;

        cleanup();
        setState("idle");

        const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
        if (totalLength === 0) {
            onError?.("No audio was captured — try recording again.");
            return;
        }

        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }

        const wavBlob = encodeWav(merged, sampleRate);
        const file = new File([wavBlob], `mic-recording-${Date.now()}.wav`, { type: "audio/wav" });
        onRecordingComplete(file);
    };

    const handleClick = () => {
        if (state === "recording") stopRecording();
        else startRecording();
    };

    const formatElapsed = (s) => `0:${String(s).padStart(2, "0")}`;

    return (
        <button
            type="button"
            className={`mic-record-button ${state === "recording" ? "is-recording" : ""} ${className}`}
            onClick={handleClick}
            disabled={disabled || state === "requesting"}
        >
            {state === "recording" ? (
                <>
                    <span className="mic-record-dot" />
                    STOP ({formatElapsed(elapsed)})
                </>
            ) : state === "requesting" ? (
                "REQUESTING MIC…"
            ) : (
                <>
                    <span className="mic-record-icon">●</span>
                    RECORD AUDIO
                </>
            )}
        </button>
    );
}

export default MicRecordButton;
