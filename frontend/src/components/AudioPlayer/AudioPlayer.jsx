import { useEffect, useRef } from "react";

function AudioPlayer({
    audio,
    onTimeUpdate,
    onAudioReady,
}) {
    const audioRef = useRef(null);

    useEffect(() => {
        if (audioRef.current) {
            onAudioReady?.(audioRef.current);
        }
    }, [audio?.src]);

    return (
        <div className="audio-player">
            <div className="audio-player-info">
                <span className="audio-player-label">
                    SELECTED SIGNAL
                </span>
                <h3>{audio.name}</h3>
            </div>

            <audio
                ref={audioRef}
                controls
                src={audio.src}
                className="audio-controls"
                onTimeUpdate={() => {
                    if (audioRef.current) {
                        onTimeUpdate?.(audioRef.current.currentTime);
                    }
                }}
            />
        </div>
    );
}

export default AudioPlayer;
