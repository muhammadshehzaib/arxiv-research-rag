import { fetchPapers, fetchStats, queryRAG } from "./services/api.js";
import { showToast } from "./utils/helpers.js";

// Import Web Components to register them
import "./components/RagSidebar.js";
import "./components/RagChat.js";
import "./components/RagSources.js";

document.addEventListener("DOMContentLoaded", () => {
    // Select Web Components
    const sidebar = document.querySelector("rag-sidebar");
    const chat = document.querySelector("rag-chat");
    const sourcesPanel = document.querySelector("rag-sources");

    // Initial Fetch & Render
    loadCorpus(sidebar);

    // Coordinate Filter Changes
    sidebar.addEventListener("filter-change", (e) => {
        // Active filters can be retrieved directly via event details or from sidebar API
        console.log("Active search parameters changed:", e.detail);
    });

    sidebar.addEventListener("filter-reset", () => {
        showToast("Filters reset successfully");
    });

    // Coordinate Sources Toggle
    chat.addEventListener("toggle-sources", () => {
        sourcesPanel.classList.toggle("open");
    });

    sourcesPanel.addEventListener("close-sources", () => {
        sourcesPanel.toggle(false);
    });

    // Coordinate Query Submissions
    chat.addEventListener("query-submit", async (e) => {
        const queryText = e.detail.query;
        const activeFilters = sidebar.getFilters();

        // 1. Append User Bubble & AI Loading Bubble
        chat.appendMessage(queryText, "user");
        const loadingId = chat.appendLoadingBubble();

        try {
            // 2. Fetch Query Answer from RAG system
            const response = await queryRAG({
                query: queryText,
                paper_id: activeFilters.paperId,
                published_after: activeFilters.publishedAfter,
                min_pages: activeFilters.minPages
            });

            // 3. Remove Loading Bubble
            chat.removeBubble(loadingId);

            // 4. Render AI response and associated sources
            chat.appendMessage(response.answer, "ai");
            chat.updateSourceBadgeCount(response.sources ? response.sources.length : 0);
            sourcesPanel.setSources(response.sources || []);

        } catch (err) {
            console.error("Query Error:", err);
            chat.removeBubble(loadingId);
            chat.appendMessage(`❌ **Error querying pipeline**: ${err.message}`, "ai");
            showToast("Failed to run RAG query", "error");
        }
    });

    // Global Citation Link Hook (defined on window since inline anchors/spans call window.highlightCitation)
    window.highlightCitation = function(index) {
        if (sourcesPanel) {
            sourcesPanel.highlightSource(index, showToast);
        }
    };
});

/**
 * Fetch and load the initial paper corpus into the sidebar selector.
 * @param {HTMLElement} sidebarComponent The sidebar custom element.
 */
async function loadCorpus(sidebarComponent) {
    try {
        const papers = await fetchPapers();
        sidebarComponent.setPapers(papers);
        
        // Fetch and display dynamic stats from Chroma DB
        try {
            const stats = await fetchStats();
            sidebarComponent.setStats(stats.papers, stats.chunks);
        } catch (statsErr) {
            console.error("Error loading stats:", statsErr);
        }
    } catch (err) {
        console.error("Error loading papers list:", err);
        showToast("Error loading indexed corpus", "error");
    }
}
