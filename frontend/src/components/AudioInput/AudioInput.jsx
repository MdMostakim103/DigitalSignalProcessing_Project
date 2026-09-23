import { useRef, useState } from "react";
import { processAudio  } from "../../services/api";

import UploadBox from "./UploadBox";
import DefaultAudioList from "./DefaultAudioList";

import AudioPlayer from "../AudioPlayer/AudioPlayer";
import Waveform from "../Visualizations/Waveform";
import ProcessedWaveform from "../Visualizations/ProcessedWaveform";

function AudioInput() {
    const [selectedAudio, setSelectedAudio] = useState(null);
    const [currentTime, setCurrentTime] = useState(0);

    const [processing, setProcessing] = useState(false);
    const [processedAudio, setProcessedAudio] = useState(null);
    const [processedCurrentTime, setProcessedCurrentTime] = useState(0);
    const [error, setError] = useState("");

    const audioElementRef = useRef(null);
    const processedAudioElementRef = useRef(null);

    const [selectedEffect, setSelectedEffect] = useState("amplify");
    const [amplifyFactor, setAmplifyFactor] = useState(2);

    const [eqLow, setEqLow] = useState(5);
    const [eqMid, setEqMid] = useState(5);
    const [eqHigh, setEqHigh] = useState(5);

    const handleAudioReady = (audioElement) => {
        audioElementRef.current = audioElement;
    };

    const effectNames = {
        amplify: "Amplified",
        reverb: "Reverb",
        echo: "Echo",
        noise: "Noise Reduced",
        equalizer: "Equalized",
    };

    const handleSeek = (time) => {
        if (audioElementRef.current) {
            audioElementRef.current.currentTime = time;
            setCurrentTime(time);
        }
    };
    const handleProcessedSeek = (time) => {
        if (processedAudioElementRef.current){
            processedAudioElementRef.current.currentTime = time;
            setProcessedCurrentTime(time);
        }
    };


    const handleProcess = async () => {
        if (!selectedAudio?.file) {
            setError("Please upload a WAV file first.");
            return;
        }

        try {
            setProcessing(true);
            setError("");

            const result = await processAudio(
                selectedAudio.file,
                selectedEffect,
                amplifyFactor,
                eqLow,
                eqMid,
                eqHigh
            );

            console.log("Backend response:", result);

        setProcessedAudio({
            name: `${effectNames[selectedEffect]} - ${selectedAudio.name}`,
            src: result.audio_url,
            plotUrl: result.plot_url,
            effect: effectNames[selectedEffect],
            inputDuration: result.input_duration_seconds,
            processedDuration: result.processed_duration_seconds,
            status: result.status,
        });

            setProcessedCurrentTime(0);

        } catch (error) {
            console.error(error);
            setError("Could not process the audio.");
        } finally {
            setProcessing(false);
        }
    };

    return (
        <section className="audio-lab" id="audio-lab">

            <div className="audio-lab-header">
                <p className="section-label">
                    AUDIO LAB
                </p>

                <h2>
                    BRING YOUR
                    <br />
                    <span>SIGNAL TO LIFE</span>
                </h2>

                <p>
                    Upload a WAV file or explore one of the
                    provided sample signals.
                </p>
            </div>

            <div className="audio-input-area">

                <UploadBox
                    onAudioSelect={(audio) => {
                        setCurrentTime(0);
                        setSelectedAudio(audio);
                    }}
                />

                <div className="audio-divider">
                    <span>OR</span>
                </div>

                <DefaultAudioList
                    onAudioSelect={(audio) => {
                        setCurrentTime(0);
                        setSelectedAudio(audio);
                    }}
                />

            </div>

            {selectedAudio && (
                <div className="selected-audio">

                    <AudioPlayer
                        audio={selectedAudio}
                        onTimeUpdate={setCurrentTime}
                        onAudioReady={handleAudioReady}
                    />

                    <Waveform
                        audio={selectedAudio}
                        currentTime={currentTime}
                        onSeek={handleSeek}
                        audioElement={audioElementRef.current}
                    />

                    <div className="processing-controls">

                        <div className="effect-selector">

                            <label htmlFor="effect">
                                DSP EFFECT
                            </label>

                            <select
                                id="effect"
                                value={selectedEffect}
                                onChange={(event) =>
                                    setSelectedEffect(event.target.value)
                                }
                            >
                                <option value="amplify">
                                    Amplify
                                </option>

                                <option value="reverb">
                                    Reverb
                                </option>

                                <option value="echo">
                                    Echo
                                </option>

                                <option value="noise">
                                    Noise Reduction
                                </option>

                                <option value="equalizer">
                                    Equalizer
                                </option>
                            </select>

                        </div>
                        {selectedEffect === "amplify" && (
                            <div className="effect-parameter">
                                <label>AMPLIFICATION</label>

                                <input
                                    type="text"
                                    inputMode="decimal"
                                    value={amplifyFactor}
                                    onChange={(e) => {
                                        const value = e.target.value;

                                        // Allow empty field while editing
                                        if (value === "") {
                                            setAmplifyFactor("");
                                            return;
                                        }

                                        // Only digits and at most one decimal point
                                        if (!/^\d*\.?\d*$/.test(value)) {
                                            return;
                                        }

                                        if(value == "0" || value == "0."){
                                            setAmplifyFactor(value);
                                            return;
                                        }

                                        const number = Number(value);

                                        // Only allow 0.01 to 10
                                        if (number >= 0.01 && number <= 10) {
                                            setAmplifyFactor(value);
                                        }
                                    }}
                                />

                                <span>0.01 — 10</span>
                            </div>
                        )}
                        {selectedEffect === "equalizer" && (
                            <div className="effect-parameter equalizer-controls">

                                <label>EQUALIZER</label>

                                <div className="eq-control">
                                    <div className="eq-label">
                                        <span>LOW</span>
                                        <strong>{eqLow}</strong>
                                    </div>

                                    <input
                                        type="range"
                                        min="0"
                                        max="10"
                                        step="1"
                                        value={eqLow}
                                        onChange={(e) => setEqLow(Number(e.target.value))}
                                    />
                                </div>

                                <div className="eq-control">
                                    <div className="eq-label">
                                        <span>MID</span>
                                        <strong>{eqMid}</strong>
                                    </div>

                                    <input
                                        type="range"
                                        min="0"
                                        max="10"
                                        step="1"
                                        value={eqMid}
                                        onChange={(e) => setEqMid(Number(e.target.value))}
                                    />
                                </div>

                                <div className="eq-control">
                                    <div className="eq-label">
                                        <span>HIGH</span>
                                        <strong>{eqHigh}</strong>
                                    </div>

                                    <input
                                        type="range"
                                        min="0"
                                        max="10"
                                        step="1"
                                        value={eqHigh}
                                        onChange={(e) => setEqHigh(Number(e.target.value))}
                                    />
                                </div>

                                <span>0 = REDUCE &nbsp; | &nbsp; 5 = NEUTRAL &nbsp; | &nbsp; 10 = BOOST</span>

                            </div>
                        )}


                        <button
                            onClick={handleProcess}
                            disabled={processing}
                            className="process-button"
                        >
                            {processing ? (
                                <>
                                    <span className="processing-spinner"></span>
                                    PROCESSING...
                                </>
                            ) : (
                                "PROCESS AUDIO"
                            )}
                        </button>

                    </div>

                </div>
            )}

            {processedAudio && (
                <div className="processed-audio">

                    <div className="processed-audio-header">
                        <span>PROCESSED SIGNAL</span>
                        <h3>{processedAudio.name}</h3>
                    </div>

                    <AudioPlayer
                        audio={processedAudio}
                        onTimeUpdate={setProcessedCurrentTime}
                        onAudioReady={(element) => {
                            processedAudioElementRef.current = element;
                        }}
                    />

                    <ProcessedWaveform
                        audio={processedAudio}
                        currentTime={processedCurrentTime}
                        onSeek={handleProcessedSeek}
                        audioElement={processedAudioElementRef.current}
                    />

                    <div className="processing-result">
                        <div className="processing-result-header">
                            <span>PROCESSING RESULT</span>
                            <h3>Processing Complete</h3>
                        </div>

                        <div className="result-grid">
                            <div className="result-item">
                                <span>EFFECT</span>
                                <strong>{processedAudio.effect}</strong>
                            </div>

                            <div className="result-item">
                                <span>INPUT</span>
                                <strong>{selectedAudio.name}</strong>
                            </div>

                            <div className="result-item">
                                <span>INPUT DURATION</span>
                                <strong>{processedAudio.inputDuration.toFixed(2)} s</strong>
                            </div>

                            <div className="result-item">
                                <span>PROCESSED DURATION</span>
                                <strong>{processedAudio.processedDuration.toFixed(2)} s</strong>
                            </div>

                            <div className="result-item">
                                <span>STATUS</span>
                                <strong>SUCCESS</strong>
                            </div>
                        </div>
                    </div>
                    {/* --- NEW 4-PANEL GRAPH VISUALZER --- */}
                    {processedAudio.plotUrl && (
                        <div className = "plot-container" style = {{ marginTop : "30px", textAlign : "center" }}>
                            <h3 style = {{ marginBottom : "15px"  }}>DSP Signal Analysis</h3>
                            <img 
                                src = {processedAudio.plotUrl} 
                                alt = "4-Panel DSP Graph"
                                style = {{ width : "100%", maxWidth : "900px", borderRadius : "10px", border : "1px solid #333" }}
                            />
                        </div>
                    )}

                </div>
            )}

            {error && (
                <p className="process-error">
                    {error}
                </p>
            )}

        </section>
    );
}

export default AudioInput;