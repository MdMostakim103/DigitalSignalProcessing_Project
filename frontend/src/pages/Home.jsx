import Navbar from "../components/Navbar";
import WelcomeHero from "../components/WelcomeHero";
import ModuleGrid from "../components/ModuleGrid";
import About from "../components/About";

function Home() {
    return (
        <>
            <Navbar />
            <main id="home">
                <WelcomeHero />
                <ModuleGrid />
                <About />
            </main>
        </>
    );
}

export default Home;
