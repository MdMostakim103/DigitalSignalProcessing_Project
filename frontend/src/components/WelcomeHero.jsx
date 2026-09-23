import RotatingVector from "./RotatingVector";

function WelcomeHero() {
    return (
        <section className="hero" id="home">

            <div className="hero-content">

                <p className="hero-label">
                    DIGITAL SIGNAL PROCESSING
                </p>

                <h1>
                    WELCOME TO THE
                    <br />
                    <span>WORLD OF SIGNALS</span>
                </h1>

                <p className="hero-description">
                    Explore sound through mathematics.
                    <br />
                    Transform, analyze, and visualize audio signals.
                </p>

                <button
                    className="hero-button"
                    onClick={() => {
                        window.location.hash = "studio";
                    }}
                >
                    ENTER STUDIO
                    <span>→</span>
                </button>

            </div>

            <div className="hero-visual">

                <div className="hero-grid"></div>

                <RotatingVector />

            </div>

        </section>
    );
}

export default WelcomeHero;