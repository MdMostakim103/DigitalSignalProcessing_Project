const API_BASE_URL = "http://127.0.0.1:8000";

export async function processAudio(
    file,
    effect,
    value = 2,
    eqLow = 5,
    eqMid = 5,
    eqHigh = 5,
    irFile = null,
    extraParams = {}
) {
    const formData = new FormData();

    formData.append("file", file);
    formData.append("effect", effect);
    formData.append("value", value);

    if (effect === "equalizer") {
        formData.append("eq_low", eqLow);
        formData.append("eq_mid", eqMid);
        formData.append("eq_high", eqHigh);
    }

    if (effect === "convolution" && irFile) {
        formData.append("ir_file", irFile);
    }

    // Effect-specific params (filter cutoff/order, echo/delay timing, etc.)
    // that don't have a dedicated argument above.
    Object.entries(extraParams || {}).forEach(([key, val]) => {
        if (val !== undefined && val !== null) formData.append(key, val);
    });

    const response = await fetch(`${API_BASE_URL}/process-audio`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const message = await response.text().catch(() => "");
        throw new Error(message || "Audio processing failed");
    }

    return await response.json();
}

// Sends a short recorded clip (already encoded as a WAV Blob in the
// browser) to the traditional-DSP Bird Sound Detector. Nothing about this
// clip is saved by the backend; it's decoded, measured, and discarded in
// the same request.
export async function detectBirdSound(blob) {
    const formData = new FormData();
    formData.append("file", blob, "bird_clip.wav");

    const response = await fetch(`${API_BASE_URL}/detect-bird`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const message = await response.text().catch(() => "");
        throw new Error(message || "Bird detection failed");
    }

    return await response.json();
}

// Fetches just the filter's own |H(f)| curve — no audio file needed — so a
// live preview graph can update as the user drags cutoff/order sliders.
export async function getFilterResponse({ filterFamily, bandType, cutoff, cutoff2, order, sampleRate }) {
    const params = new URLSearchParams({
        filter_family: filterFamily,
        band_type: bandType,
        cutoff: String(cutoff),
        cutoff2: String(cutoff2),
        order: String(order),
        sr: String(sampleRate),
    });

    const response = await fetch(`${API_BASE_URL}/filter-response?${params.toString()}`);
    if (!response.ok) {
        throw new Error("Failed to fetch filter response");
    }
    return await response.json();
}
