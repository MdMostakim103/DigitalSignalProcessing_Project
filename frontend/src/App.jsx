import { useEffect, useState } from "react";
import Home from "./pages/Home";
import Studio from "./pages/Studio";
import AmplitudeDynamics from "./pages/AmplitudeDynamics";
import TimeDomain from "./pages/TimeDomain";
import FrequencyFiltering from "./pages/FrequencyFiltering";
import SpeechActivity from "./pages/SpeechActivity";
import SpectralDetection from "./pages/SpectralDetection";
import VoiceMorphing from "./pages/VoiceMorphing";
import BirdDetector from "./pages/BirdDetector";

function getRoute() {
    const hash = window.location.hash;
    if (hash === "#studio") return "studio";
    if (hash === "#module/01") return "amplitude";
    if (hash === "#module/02") return "frequencyFiltering";
    if (hash === "#module/03") return "timeDomain";
    if (hash === "#module/04") return "speechActivity";
    if (hash === "#module/05") return "spectralDetection";
    if (hash === "#module/06") return "voiceMorphing";
    if (hash === "#module/BONUS") return "birdDetector";

    return "home";
}

function App() {
    const [route, setRoute] = useState(getRoute);

    useEffect(() => {
        const handleHashChange = () => setRoute(getRoute());
        window.addEventListener("hashchange", handleHashChange);
        return () => window.removeEventListener("hashchange", handleHashChange);
    }, []);

    if (route === "studio") return <Studio />;
    if (route === "amplitude") return <AmplitudeDynamics />;
    if (route === "frequencyFiltering") return <FrequencyFiltering />;
    if (route === "timeDomain") return <TimeDomain />;
    if (route === "speechActivity") return <SpeechActivity />;
    if (route === "spectralDetection") return <SpectralDetection />;
    if (route === "voiceMorphing") return <VoiceMorphing />;
    if (route === "birdDetector") return <BirdDetector />;

    return <Home />;
}

export default App;