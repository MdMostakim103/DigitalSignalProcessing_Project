function DSPModuleCard({ module }) {
    const handleOpen = () => {
        window.location.hash = `module/${module.number}`;
    };

    return (
        <article
            className="module-card"
            onClick={handleOpen}
            onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") handleOpen();
            }}
            role="button"
            tabIndex={0}
        >
            <div className="module-number">{module.number}</div>
            <div className="module-content">
                <h3>{module.title}</h3>
                <p>{module.description}</p>
            </div>
            <div className="module-arrow">↗</div>
        </article>
    );
}

export default DSPModuleCard;
