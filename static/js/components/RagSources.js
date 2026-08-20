/**
 * Web Component for slide-out Sources panel.
 */
class RagSources extends HTMLElement {
    connectedCallback() {
        this.render();
        this.setupListeners();
    }

    render() {
        this.innerHTML = `
            <div class="panel-header">
                <h3>Retrieved Sources</h3>
                <button id="close-sources-btn" class="btn btn-icon">
                    <i data-lucide="x"></i>
                </button>
            </div>
            <div class="panel-content" id="sources-container">
                <div class="empty-state">
                    <i data-lucide="library"></i>
                    <p>Sources retrieved for your last query will appear here.</p>
                </div>
            </div>
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    setupListeners() {
        const closeBtn = this.querySelector("#close-sources-btn");
        closeBtn.addEventListener("click", () => {
            this.dispatchEvent(new CustomEvent("close-sources", {
                bubbles: true
            }));
        });
    }

    /**
     * Populate and render sources metadata.
     * @param {Array} sources List of sources from the backend.
     */
    setSources(sources) {
        const container = this.querySelector("#sources-container");
        if (!container) return;

        if (!sources || sources.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="library"></i>
                    <p>No sources retrieved for this query.</p>
                </div>
            `;
            if (window.lucide) {
                window.lucide.createIcons();
            }
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

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    /**
     * Open or close the sidebar.
     * @param {boolean} open True to open, false to close.
     */
    toggle(open) {
        if (open) {
            this.classList.add("open");
        } else {
            this.classList.remove("open");
        }
    }

    /**
     * Find and scroll to a specific source card, highlighting it temporarily.
     * @param {number} index Source index.
     * @param {Function} showToastCallback Callback to show feedback toast.
     */
    highlightSource(index, showToastCallback) {
        // First ensure panel is open
        this.toggle(true);

        // Clear previous highlights
        this.querySelectorAll(".source-card").forEach(c => c.classList.remove("highlighted"));

        // Find card
        const targetCard = this.querySelector(`#source-card-${index}`);
        if (targetCard) {
            targetCard.classList.add("highlighted");
            targetCard.scrollIntoView({ behavior: "smooth", block: "center" });

            // Flash effect
            setTimeout(() => {
                targetCard.classList.remove("highlighted");
            }, 3000);
        } else if (showToastCallback) {
            showToastCallback(`Source details [${index}] not found in this response.`, "warning");
        }
    }
}

customElements.define("rag-sources", RagSources);
export default RagSources;
