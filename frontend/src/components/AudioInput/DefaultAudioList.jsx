const samples = [
    {
        name: "Speech",
        file: "/audio/speech.wav",
    },
    {
        name: "Piano",
        file: "/audio/piano.wav",
    },
    {
        name: "Music",
        file: "/audio/music.wav",
    },
    {
        name: "Noise",
        file: "/audio/noise.wav",
    },
];

function DefaultAudioList({ onAudioSelect }) {

    const selectSample = (sample) => {
        onAudioSelect({
            name: sample.name,
            src: sample.file,
        });
    };

    return (
        <div className="sample-list">

            <p className="sample-label">
                TRY A SAMPLE
            </p>

            <div className="sample-buttons">

                {samples.map((sample) => (
                    <button
                        key={sample.name}
                        className="sample-button"
                        onClick={() =>
                            selectSample(sample)
                        }
                    >
                        {sample.name}
                    </button>
                ))}

            </div>

        </div>
    );
}

export default DefaultAudioList;