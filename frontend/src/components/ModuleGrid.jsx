import { modules } from "../data/modules";
import DSPModuleCard from "./DSPModuleCard";

function ModuleGrid() {
    return (
        <section className="modules-section" id="modules">
            <div className="section-heading">
                <p className="section-label">FEEL YOUR SIGNAL</p>
                <h2>
                    SEE WHAT
                    <br />
                    <span>YOUR SIGNAL DOES.</span>
                </h2>
                <p className="section-description">
                    Enter a DSP module and watch the transformation happen — from the original signal,
                    through the processing steps, to the final result.
                </p>
            </div>

            <div className="modules-grid">
                {modules.map((module) => (
                    <DSPModuleCard key={module.number} module={module} />
                ))}
            </div>
        </section>
    );
}

export default ModuleGrid;
