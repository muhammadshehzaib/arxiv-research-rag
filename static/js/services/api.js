let accessToken = "";
let pendingLogin = null;

function requestAccessToken() {
    if (pendingLogin) return pendingLogin;
    pendingLogin = new Promise((resolve, reject) => {
        const dialog = document.createElement("dialog");
        dialog.innerHTML = `<form><h3>API access</h3><p>Enter the access token configured on your server.</p>
            <input type="password" name="token" aria-label="API access token" autocomplete="off" required>
            <button type="submit">Connect</button><button type="button" id="cancel-auth">Cancel</button></form>`;
        document.body.appendChild(dialog);
        const close = () => { dialog.remove(); pendingLogin = null; };
        dialog.querySelector("form").onsubmit = event => {
            event.preventDefault();
            const token = dialog.querySelector("input").value;
            close(); resolve(token);
        };
        const cancel = event => { event.preventDefault(); close(); reject(new Error("Authentication cancelled.")); };
        dialog.querySelector("#cancel-auth").onclick = cancel;
        dialog.oncancel = cancel;
        dialog.showModal();
    });
    return pendingLogin;
}

async function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    let response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        accessToken = await requestAccessToken();
        headers.set("Authorization", `Bearer ${accessToken}`);
        response = await fetch(url, { ...options, headers });
    }
    return response;
}

/**
 * API services for the arXiv RAG application.
 */

/**
 * Fetch the list of indexed research papers.
 * @returns {Promise<Array>} List of paper objects containing id, title, etc.
 */
export async function fetchPapers() {
    console.log("📡 API: Fetching research papers corpus...");
    const response = await apiFetch("/api/papers");
    if (!response.ok) {
        console.error("❌ API: Failed to fetch papers list");
        throw new Error("Failed to fetch papers list");
    }
    const data = await response.json();
    console.log("✅ API: Papers list retrieved. Count:", data.length);
    return data;
}

/**
 * Fetch database stats (paper and chunk count).
 * @returns {Promise<Object>} Stats object.
 */
export async function fetchStats() {
    console.log("📡 API: Fetching database statistics...");
    const response = await apiFetch("/api/stats");
    if (!response.ok) {
        console.error("❌ API: Failed to fetch database stats");
        throw new Error("Failed to fetch database stats");
    }
    const data = await response.json();
    console.log("✅ API: Database stats retrieved:", data);
    return data;
}

/**
 * Submit a search query to the RAG backend.
 * @param {Object} queryParams Query payload including query text and filters.
 * @returns {Promise<Object>} Object containing target answer and source citations.
 */
export async function queryRAG(queryParams, onProgress = () => {}) {
    console.log("📡 API: Submitting RAG query:", queryParams);
    const response = await apiFetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(queryParams)
    });

    if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        console.error("❌ API: RAG query execution failed:", errBody);
        throw new Error(errBody.detail || "Server failed to query RAG database");
    }

    if (!response.body) {
        throw new Error("Your browser cannot show live search progress. Please try an updated browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function handleLine(line) {
        if (!line.trim()) return null;
        const event = JSON.parse(line);
        if (event.type === "status") onProgress(event.stage);
        if (event.type === "error") throw new Error(event.message || "The search could not be completed.");
        if (event.type === "result") return { answer: event.answer, sources: event.sources || [] };
        return null;
    }

    try {
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            const lines = buffer.split("\n");
            buffer = lines.pop();
            for (const line of lines) {
                const result = handleLine(line);
                if (result) return result;
            }
            if (done) {
                const result = handleLine(buffer);
                if (result) return result;
                throw new Error("The search ended before an answer was ready. Please try again.");
            }
        }
    } finally {
        reader.releaseLock();
    }
}

/**
 * Fetch the latest RAG Triad evaluation results.
 * @returns {Promise<Object>} Latest evaluation summary and details.
 */
export async function fetchLatestEval() {
    console.log("📡 API: Requesting latest evaluation report...");
    const response = await apiFetch("/api/eval/latest");
    if (!response.ok) {
        console.error("❌ API: Failed to fetch latest evaluation results");
        throw new Error("Failed to fetch latest evaluation results");
    }
    const data = await response.json();
    console.log("✅ API: Latest eval report received:", data);
    return data;
}

/**
 * Fetch historical RAG Triad evaluation runs.
 * @returns {Promise<Array>} List of historical summaries.
 */
export async function fetchEvalHistory() {
    console.log("📡 API: Requesting evaluation history timeline...");
    const response = await apiFetch("/api/eval/history");
    if (!response.ok) {
        console.error("❌ API: Failed to fetch evaluation history");
        throw new Error("Failed to fetch evaluation history");
    }
    const data = await response.json();
    console.log("✅ API: Eval history timeline received. Runs count:", data.length);
    return data;
}

/**
 * Trigger a new RAG Triad evaluation run.
 * @returns {Promise<Object>} Newly calculated evaluation results.
 */
export async function runEval() {
    console.log("📡 API: Triggering live RAG Triad Evaluation Suite (approx 20-30s)...");
    const response = await apiFetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    });
    if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        console.error("❌ API: Evaluation suite execution failed:", errBody);
        throw new Error(errBody.detail || "Failed to execute evaluation suite");
    }
    const data = await response.json();
    console.log("✅ API: Live evaluation suite completed. Summary scores:", {
        faithfulness: data.faithfulness,
        answer_relevancy: data.answer_relevancy,
        context_precision: data.context_precision,
        context_recall: data.context_recall,
        average_score: data.average_score
    });
    return data;
}

