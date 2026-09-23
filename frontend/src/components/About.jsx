function About() {
    return (
        <section className="about-section" id="about">
            <div className="section-heading">
                <p className="section-label">THE ARCHITECTURE</p>
                <h2>
                    UNDERSTANDING
                    <br />
                    <span>DSP STUDIO</span>
                </h2>
                <p className="section-description">
                    An interactive audio workstation bridging raw mathematical signal theory 
                    with real-time web audio manipulation and high-performance processing.
                </p>
            </div>

            <div className="about-grid">
                <div className="about-card">
                    <span className="about-tag">CORE THEORY</span>
                    <h3>Signals & Transforms</h3>
                    <p>
                        Digital signal processing converts analog physics into discrete numbers. 
                        From Fast Fourier Transforms (FFT) splitting frequencies to discrete 
                        convolutions modeling physical acoustics, every effect reflects mathematical algorithms.
                    </p>
                </div>

                <div className="about-card">
                    <span className="about-tag">BACKEND ENGINE</span>
                    <h3>FastAPI & SciPy Pipeline</h3>
                    <p>
                        Audio files are transmitted via asynchronous REST endpoints to Python, 
                        where NumPy, SciPy, and Librosa run vector math on PCM samples, synthesize impulse 
                        responses, and compute waveform plots on the fly.
                    </p>
                </div>

                <div className="about-card">
                    <span className="about-tag">FRONTEND INTERACTION</span>
                    <h3>Visual React Studio</h3>
                    <p>
                        Built with Vite and modular CSS tokens, the interface provides parametric 
                        visualizations, real-time waveform inspections, and modular routing to 
                        explore acoustic concepts effortlessly.
                    </p>
                </div>
            </div>

            <div className="about-tech-stack">
                <div className="tech-item"><span>ENGINE</span> Python • FastAPI • Librosa • NumPy</div>
                <div className="tech-item"><span>CLIENT</span> React • Vite • Modern CSS3</div>
                <div className="tech-item"><span>DSP METHODS</span> FFT Equalization • Convolution Reverb • Delay Mixers</div>
            </div>
        </section>
    );
}

export default About;