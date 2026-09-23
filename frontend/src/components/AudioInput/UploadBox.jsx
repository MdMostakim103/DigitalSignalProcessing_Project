import { useRef } from "react";

function UploadBox({ onAudioSelect }) {
    const fileInputRef = useRef(null);

    const handleClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];

        if (!file) {
            return;
        }

        const audioUrl = URL.createObjectURL(file);

        onAudioSelect({
            name: file.name,
            src: audioUrl,
            file: file,
        });
    };

    return (
        <div className="upload-box">

            <div className="upload-icon">
                ↑
            </div>

            <h3>
                DROP YOUR WAV FILE
            </h3>

            <p>
                Upload an audio file to begin signal
                processing.
            </p>

            <button
                className="upload-button"
                onClick={handleClick}
            >
                Upload Audio
            </button>

            <input
                ref={fileInputRef}
                type="file"
                accept=".wav,audio/wav"
                onChange={handleFileChange}
                hidden
            />

            <span className="upload-hint">
                WAV files only
            </span>

        </div>
    );
}

export default UploadBox;