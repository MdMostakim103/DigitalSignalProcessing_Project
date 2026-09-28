import { useRef, useState } from "react";
import MicRecordButton from "./MicRecordButton";

function UploadBox({ onAudioSelect }) {
    const fileInputRef = useRef(null);
    const [error, setError] = useState("");

    const handleClick = () => {
        fileInputRef.current.click();
    };

    const selectFile = (file) => {
        if (!file) return;

        const audioUrl = URL.createObjectURL(file);
        setError("");

        onAudioSelect({
            name: file.name,
            src: audioUrl,
            file: file,
        });
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        event.target.value = "";
        selectFile(file);
    };

    return (
        <div className="upload-box">

            <div className="upload-icon">
                ↑
            </div>

            <h3>
                DROP YOUR AUDIO FILE
            </h3>

            <p>
                Upload an audio file to begin signal
                processing, or record one live.
            </p>

            <button
                className="upload-button"
                onClick={handleClick}
            >
                Upload Audio
            </button>

            <MicRecordButton
                onRecordingComplete={selectFile}
                onError={setError}
            />

            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                hidden
            />

            <span className="upload-hint">
                Any audio format works
            </span>

            {error && (
                <p className="upload-error">
                    {error}
                </p>
            )}

        </div>
    );
}

export default UploadBox;
