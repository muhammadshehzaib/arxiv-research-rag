// State Variables
let currentSources = [];

// Initialize Page Elements
document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // Fetch and populate paper select list & statistics
    fetchPapers();

    // Setup Event Listeners
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    const minPagesSlider = document.getElementById("min-pages");
    const minPagesVal = document.getElementById("min-pages-val");
    const clearFiltersBtn = document.getElementById("clear-filters-btn");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const toggleSourcesBtn = document.getElementById("toggle-sources-btn");
    const closeSourcesBtn = document.getElementById("close-sources-btn");
    const sourcesPanel = document.getElementById("sources-panel");

    // Slider value synchronization
    minPagesSlider.addEventListener("input", (e) => {
        minPagesVal.textContent = e.target.value;
    });

    // Reset filters handler
    clearFiltersBtn.addEventListener("click", () => {
        document.getElementById("paper-select").value = "";
        document.getElementById("published-after").value = "";
        minPagesSlider.value = 0;
        minPagesVal.textContent = 0;
        showToast("Filters reset successfully");
    });

    // Sidebar panels toggle logic
    toggleSourcesBtn.addEventListener("click", () => {
        sourcesPanel.classList.toggle("open");
    });

    closeSourcesBtn.addEventListener("click", () => {
        sourcesPanel.classList.remove("open");
    });

    // Handle example buttons click
    document.querySelectorAll(".example-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            chatInput.value = btn.textContent;
            chatInput.focus();
        });
    });

    // Chat submit event
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleSubmit();
    });

    // Textarea autosize and enter key submit handler
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    });
}

// Fetch Papers List from API
async function fetchPapers() {
    try {
        const response = await fetch("/api/papers");
        if (!response.ok) throw new Error("Failed to fetch papers");
        
        const papers = await response.ok ? await response.json() : [];
        const select = document.getElementById("paper-select");
        const statsPaperCount = document.getElementById("stats-paper-count");
        const statsChunkCount = document.getElementById("stats-chunk-count");

        // Clear existing options except first
        select.innerHTML = '<option value="">All Papers</option>';

        // Populate dropdown
        papers.forEach(paper => {
            const opt = document.createElement("option");
            opt.value = paper.paper_id;
            const truncatedTitle = paper.title.length > 55 ? paper.title.substring(0, 52) + "..." : paper.title;
            opt.textContent = `[${paper.paper_id}] ${truncatedTitle}`;
            select.appendChild(opt);
        });

        // Set corpus statistics
        statsPaperCount.textContent = `${papers.length} Papers`;
        
        // Estimate chunks count (assume average of ~15-20 chunks per paper if not computed)
        let totalChunks = 916; // Hardcoded default based on database build logs
        statsChunkCount.textContent = `${totalChunks} Chunks`;

    } catch (err) {
        console.error("Error loading papers list:", err);
        showToast("Error loading indexed corpus", "error");
    }
}

// Handle chat query submission
async function handleSubmit() {
    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    // Reset input height & value
    input.value = "";
    input.style.height = "auto";

    // Hide welcome card on first query
    const welcomeCard = document.getElementById("welcome-card");
    if (welcomeCard) {
        welcomeCard.style.display = "none";
    }

    // Append user bubble to chat feed
    appendMessage(query, "user");

    // Gather active search filters
    const paperId = document.getElementById("paper-select").value;
    const publishedAfter = document.getElementById("published-after").value;
    const minPagesVal = document.getElementById("min-pages").value;
    const minPages = minPagesVal > 0 ? parseInt(minPagesVal) : null;

    // Append AI skeleton loading bubble
    const loadingBubbleId = appendLoadingBubble();
    scrollToBottom();

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                paper_id: paperId || null,
                published_after: publishedAfter || null,
                min_pages: minPages
            })
        });

        if (!response.ok) {
            const errBody = await response.json();
            throw new Error(errBody.detail || "Server failed to query RAG database");
        }

        const data = await response.json();
        
        // Remove loading state bubble
        removeBubble(loadingBubbleId);

        // Process citations and append AI bubble
        appendMessage(data.answer, "ai", data.sources);
        
        // Render sources in the slide-out panel
        currentSources = data.sources || [];
        renderSources(currentSources);

    } catch (err) {
        console.error("Query Error:", err);
        removeBubble(loadingBubbleId);
        appendMessage(`❌ **Error querying pipeline**: ${err.message}`, "ai");
    }
    
    scrollToBottom();
}

// Append bubble to chat feed
function appendMessage(text, sender, sources = []) {
    const feed = document.getElementById("message-feed");
    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (sender === "user") {
        bubble.textContent = text;
    } else {
        // Format markdown-like text elements (citations, bolding, lists)
        bubble.innerHTML = formatMarkdown(text);
    }

    wrapper.appendChild(bubble);
    feed.appendChild(wrapper);
}

// Append skeleton load bubble
function appendLoadingBubble() {
    const feed = document.getElementById("message-feed");
    const bubbleId = "loading-" + Date.now();

    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper ai";
    wrapper.id = bubbleId;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.style.width = "60%";

    bubble.innerHTML = `
        <div class="skeleton-msg">
            <div class="skeleton-line long"></div>
            <div class="skeleton-line medium"></div>
            <div class="skeleton-line short"></div>
        </div>
    `;

    wrapper.appendChild(bubble);
    feed.appendChild(wrapper);
    return bubbleId;
}

// Remove bubble from chat feed by id
function removeBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Format simple Markdown tags (citations, headers, lists)
function formatMarkdown(text) {
    // Escape HTML tags to prevent XSS
    let escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Replace bold tags: **text** -> <strong>text</strong>
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Replace lists: \n* item -> \n<li>item</li>
    escaped = escaped.replace(/\n\s*[\*\-]\s*(.*?)/g, "\n<li>$1</li>");
    escaped = escaped.replace(/(<li>.*?<\/li>)+/gs, "<ul>$&</ul>");

    // Replace numbered lists: \n1. item -> \n<li>item</li>
    escaped = escaped.replace(/\n\s*\d+\.\s*(.*?)/g, "\n<li class='num-list'>$1</li>");
    escaped = escaped.replace(/(<li class='num-list'>.*?<\/li>)+/gs, "<ol>$&</ol>");

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

// Highlight Citation in Slide-out panel
window.highlightCitation = function(index) {
    // Open the sources panel if closed
    const panel = document.getElementById("sources-panel");
    panel.classList.add("open");

    // Clear previous highlights
    document.querySelectorAll(".source-card").forEach(c => c.classList.remove("highlighted"));

    // Find and highlight card
    const targetCard = document.getElementById(`source-card-${index}`);
    if (targetCard) {
        targetCard.classList.add("highlighted");
        targetCard.scrollIntoView({ behavior: "smooth", block: "center" });

        // Flash effect
        setTimeout(() => {
            targetCard.classList.remove("highlighted");
        }, 3000);
    } else {
        showToast(`Source details [${index}] not found in this response.`, "warning");
    }
};

// Render citations list in sidebar container
function renderSources(sources) {
    const container = document.getElementById("sources-container");
    const badge = document.getElementById("source-badge");

    // Set header badge counter
    badge.textContent = sources.length;

    if (!sources || sources.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i data-lucide="library"></i>
                <p>No sources retrieved for this query.</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    container.innerHTML = "";
    sources.forEach(src => {
        const card = document.createElement("div");
        card.className = "source-card";
        card.id = `source-card-${src.index}`;

        // Compute similarity confidence (1 - distance)
        const confidence = Math.max(0, (1 - src.distance) * 100).toFixed(1);

        card.innerHTML = `
            <span class="source-idx">Source [${src.index}]</span>
            <h4>${src.title}</h4>
            <div class="source-authors">Authors: ${src.authors}</div>
            <div class="source-footer">
                <div class="score-badge">
                    <i data-lucide="compass" style="width: 13px; height: 13px;"></i>
                    Similarity: ${confidence}%
                </div>
                <a href="${src.pdf_url}" target="_blank" class="pdf-link">
                    <i data-lucide="external-link" style="width: 13px; height: 13px;"></i> arXiv PDF
                </a>
            </div>
        `;
        container.appendChild(card);
    });

    lucide.createIcons();
}

// Auto-scroll message feed to bottom
function scrollToBottom() {
    const feed = document.getElementById("message-feed");
    feed.scrollTop = feed.scrollHeight;
}

// Show temporary toast message/notifications
function showToast(message, type = "success") {
    // Create toast element
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
