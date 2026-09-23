import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

import "./styles/variables.css";
import "./styles/global.css";
import "./styles/navbar.css";
import "./styles/hero.css";
import "./styles/modules.css";
import "./styles/module-page.css";
import "./styles/about.css";
import "./styles/studio.css";

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
