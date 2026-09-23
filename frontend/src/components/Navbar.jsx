function Navbar() {
    return (
        <nav className="navbar">
            <div className="navbar-logo">DSP<span>/</span>STUDIO</div>

            <div className="navbar-links">
                <a href="#home">Home</a>
                <a href="#modules">DSP Modules</a>
                <a href="#about">About</a>
            </div>

            <button
                className="navbar-button"
                onClick={() => {
                    window.location.hash = "studio";
                }}
            >
                Enter Studio
            </button>
        </nav>
    );
}

export default Navbar;
