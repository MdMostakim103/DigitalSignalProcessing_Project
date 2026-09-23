function SignalWave() {
    const points = [];

    for (let i = 0; i <= 500; i++) {
        const x = i;

        // Combination of two sine waves
        const y =
            50 +
            18 * Math.sin(i * 0.06) +
            8 * Math.sin(i * 0.17);

        points.push(`${x},${y}`);
    }

    return (
        <svg
            className="signal-wave"
            viewBox="0 0 500 100"
            preserveAspectRatio="none"
        >
            <polyline
                points={points.join(" ")}
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
            />
        </svg>
    );
}

export default SignalWave;