import { useState } from "react";

function generateHeart() {
    const points = [];

    for (let i = 0; i <= 360; i++) {
        const t = (i * Math.PI) / 180;

        const x = 16 * Math.pow(Math.sin(t), 3);

        const y =
            13 * Math.cos(t) -
            5 * Math.cos(2 * t) -
            2 * Math.cos(3 * t) -
            Math.cos(4 * t);

        points.push(`${x * 10},${-y * 10}`);
    }

    return points;
}

function generateLissajous() {
    const points = [];

    for (let i = 0; i <= 360; i++) {
        const t = (i * Math.PI) / 180;

        const x = 140 * Math.sin(3 * t + Math.PI / 2);
        const y = 140 * Math.sin(2 * t);

        points.push(`${x},${y}`);
    }

    return points;
}

function generateFlower() {
    const points = [];

    for (let i = 0; i <= 360; i++) {
        const t = (i * Math.PI) / 180;

        const radius = 110 + 35 * Math.sin(5 * t);

        const x = radius * Math.cos(t);
        const y = radius * Math.sin(t);

        points.push(`${x},${y}`);
    }

    return points;
}

function generateSpiral() {
    const points = [];

    for (let i = 0; i <= 720; i++) {
        const t = (i * Math.PI) / 180;

        const radius = 0.35 * i;

        const x = radius * Math.cos(t);
        const y = radius * Math.sin(t);

        points.push(`${x},${y}`);
    }

    return points;
}

function RotatingVector() {
    const shapes = [
        generateHeart,
        generateLissajous,
        generateFlower,
        generateSpiral,
    ];

    const [points] = useState(() => {
        const randomIndex = Math.floor(
            Math.random() * shapes.length
        );

        return shapes[randomIndex]();
    });

    return (
        <div className="rotating-vector">

            <svg
                viewBox="-180 -180 360 360"
                className="heart-svg"
            >
                <polyline
                    points={points.join(" ")}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                />
            </svg>

        </div>
    );
}

export default RotatingVector;