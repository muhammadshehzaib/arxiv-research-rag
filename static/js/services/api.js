/**
 * API services for the arXiv RAG application.
 */

/**
 * Fetch the list of indexed research papers.
 * @returns {Promise<Array>} List of paper objects containing id, title, etc.
 */
export async function fetchPapers() {
    const response = await fetch("/api/papers");
    if (!response.ok) {
        throw new Error("Failed to fetch papers list");
    }
    return await response.json();
}

/**
 * Submit a search query to the RAG backend.
 * @param {Object} queryParams Query payload including query text and filters.
 * @returns {Promise<Object>} Object containing target answer and source citations.
 */
export async function queryRAG(queryParams) {
    const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(queryParams)
    });

    if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || "Server failed to query RAG database");
    }

    return await response.json();
}
