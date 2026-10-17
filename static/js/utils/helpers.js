/**
 * Utility helpers for formatting and UI alerts.
 */

/** Display only explicitly scored relevance; never infer confidence from distance. */
export function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
}

export function safeExternalUrl(value) {
    try {
        const url = new URL(String(value));
        return ["http:", "https:"].includes(url.protocol) ? url.href : "#";
    } catch {
        return "#";
    }
}

export function formatRelevanceScore(source) {
    const score = source.relevance_score;
    if (typeof score === "number" && Number.isFinite(score) && score >= 0 && score <= 1) {
        return `Relevance: ${score.toFixed(3)}`;
    }
    return source.is_live_mcp ? "Live source · unscored" : "Relevance not scored";
}

/** Format markdown-like text as escaped HTML with citation binding. */
export function formatMarkdown(text) {
    // Escape HTML tags to prevent XSS
    let escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Replace bold tags: **text** -> <strong>text</strong>
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Replace bulleted lists
    escaped = escaped.replace(/((?:\n\s*[\*\-]\s*[^\n]*(?:\n\s*\n)?)+)/g, (match) => {
        const items = match.replace(/\n\s*[\*\-]\s*(.*)/g, "<li>$1</li>");
        return `<ul>${items.replace(/\n/g, "")}</ul>`;
    });

    // Replace numbered lists
    escaped = escaped.replace(/((?:\n\s*\d+\.\s*[^\n]*(?:\n\s*\n)?)+)/g, (match) => {
        const items = match.replace(/\n\s*\d+\.\s*(.*)/g, "<li>$1</li>");
        return `<ol>${items.replace(/\n/g, "")}</ol>`;
    });

    // Replace linebreaks
    escaped = escaped.replace(/\n/g, "<br>");

    // Format Citations: [Source X] or [Source X, Source Y] or [X]
    // 1. Matches [Source X] -> creates clickable span
    escaped = escaped.replace(/\[Source\s+(\d+)\]/g, (match, num) => {
        return `<span class="citation-link" onclick="highlightCitation(${num})">[Source ${num}]</span>`;
    });

    // 2. Matches [X] (bracket numbers) -> creates clickable span
    escaped = escaped.replace(/\[(\d+)\]/g, (match, num) => {
        return `<span class="citation-link" onclick="highlightCitation(${num})">[Source ${num}]</span>`;
    });

    return escaped;
}

/**
 * Display a temporary toast notification in the application.
 * @param {string} message Text message to display.
 * @param {string} type Theme selector ('success', 'warning', 'error').
 */
export function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.style.position = "absolute";
    toast.style.bottom = "20px";
    toast.style.left = "50%";
    toast.style.transform = "translateX(-50%)";
    toast.style.background = type === "error" ? "#ef4444" : "rgba(30,30,50,0.9)";
    toast.style.color = "#fff";
    toast.style.padding = "10px 20px";
    toast.style.borderRadius = "8px";
    toast.style.boxShadow = "0 4px 16px rgba(0,0,0,0.5)";
    toast.style.zIndex = "100";
    toast.style.fontSize = "13px";
    toast.style.backdropFilter = "blur(10px)";
    toast.style.border = "1px solid rgba(255,255,255,0.1)";
    toast.style.transition = "opacity 0.3s ease-out";
    
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}
