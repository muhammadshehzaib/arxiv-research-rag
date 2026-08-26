/**
 * API services for the arXiv RAG application.
 */

/**
 * Fetch the list of indexed research papers.
 * @returns {Promise<Array>} List of paper objects containing id, title, etc.
 */
export async function fetchPapers() {
    console.log("📡 API: Fetching research papers corpus...");
    const response = await fetch("/api/papers");
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
    const response = await fetch("/api/stats");
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
export async function queryRAG(queryParams) {
    console.log("📡 API: Submitting RAG query:", queryParams);
    const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(queryParams)
    });

    if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        console.error("❌ API: RAG query execution failed:", errBody);
        throw new Error(errBody.detail || "Server failed to query RAG database");
    }

    const data = await response.json();
    console.log("✅ API: RAG response received. Sources count:", data.sources ? data.sources.length : 0);
    return data;
}

/**
 * Fetch the latest RAG Triad evaluation results.
 * @returns {Promise<Object>} Latest evaluation summary and details.
 */
export async function fetchLatestEval() {
    console.log("📡 API: Requesting latest evaluation report...");
    const response = await fetch("/api/eval/latest");
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
    const response = await fetch("/api/eval/history");
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
    const response = await fetch("/api/eval/run", {
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

