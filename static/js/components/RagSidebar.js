/**
 * Web Component for RAG Sidebar: Handles search filters and repository statistics.
 */
class RagSidebar extends HTMLElement {
    connectedCallback() {
        this.render();
        this.setupListeners();
    }

    render() {
        this.innerHTML = `
            <div class="sidebar-header">
                <div class="logo-icon">
                    <i data-lucide="brain-circuit"></i>
                </div>
                <div>
                    <h1>arXiv RAG</h1>
                    <span class="subtitle">Research Explorer</span>
                </div>
            </div>

            <div class="sidebar-content">
                <div class="filter-card">
                    <div class="filter-header">
                        <i data-lucide="sliders-horizontal"></i>
                        <h2>Search Filters</h2>
                    </div>

                    <div class="control-group">
                        <label for="paper-select">Target Research Paper</label>
                        <select id="paper-select">
                            <option value="">All Papers</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label for="published-after">Published On/After</label>
                        <input type="date" id="published-after" min="2010-01-01" max="2027-12-31">
                    </div>

                    <div class="control-group">
                        <div class="slider-label-row">
                            <label for="min-pages">Min Page Count</label>
                            <span id="min-pages-val">0</span>
                        </div>
                        <input type="range" id="min-pages" min="0" max="100" value="0">
                    </div>

                    <button id="clear-filters-btn" class="btn btn-secondary">
                        <i data-lucide="rotate-ccw"></i> Reset Filters
                    </button>
                </div>
                
                <div class="stats-card">
                    <div class="stat-row">
                        <span class="stat-label">Indexed Corpus</span>
                        <span class="stat-value" id="stats-paper-count">0 Papers</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Text Chunks</span>
                        <span class="stat-value" id="stats-chunk-count">0 Chunks</span>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-footer">
                <div class="footer-badge">
                    <span class="indicator green"></span>
                    <span id="backend-status">Connected to Gemini API</span>
                </div>
            </div>
        `;
        
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    setupListeners() {
        const select = this.querySelector("#paper-select");
        const dateInput = this.querySelector("#published-after");
        const slider = this.querySelector("#min-pages");
        const sliderVal = this.querySelector("#min-pages-val");
        const resetBtn = this.querySelector("#clear-filters-btn");

        // Sync slider display value and dispatch event
        slider.addEventListener("input", (e) => {
            sliderVal.textContent = e.target.value;
            this.dispatchFilterChange();
        });

        select.addEventListener("change", () => this.dispatchFilterChange());
        dateInput.addEventListener("change", () => this.dispatchFilterChange());

        resetBtn.addEventListener("click", () => {
            select.value = "";
            dateInput.value = "";
            slider.value = 0;
            sliderVal.textContent = 0;
            
            this.dispatchEvent(new CustomEvent("filter-reset"));
            this.dispatchFilterChange();
        });
    }

    dispatchFilterChange() {
        this.dispatchEvent(new CustomEvent("filter-change", {
            detail: this.getFilters(),
            bubbles: true
        }));
    }

    /**
     * Retrieve the current state of search filters.
     * @returns {Object} Filters object
     */
    getFilters() {
        return {
            paperId: this.querySelector("#paper-select").value || null,
            publishedAfter: this.querySelector("#published-after").value || null,
            minPages: parseInt(this.querySelector("#min-pages").value) || null
        };
    }

    /**
     * Populate the dropdown select input with paper metadata.
     * @param {Array} papers List of papers.
     */
    setPapers(papers) {
        const select = this.querySelector("#paper-select");
        if (!select) return;
        
        select.innerHTML = '<option value="">All Papers</option>';
        papers.forEach(paper => {
            const opt = document.createElement("option");
            opt.value = paper.paper_id;
            const truncatedTitle = paper.title.length > 55 ? paper.title.substring(0, 52) + "..." : paper.title;
            opt.textContent = truncatedTitle;
            select.appendChild(opt);
        });
    }

    /**
     * Set stats fields values.
     * @param {number} paperCount 
     * @param {number} chunkCount 
     */
    setStats(paperCount, chunkCount) {
        const statsPaperCount = this.querySelector("#stats-paper-count");
        const statsChunkCount = this.querySelector("#stats-chunk-count");
        if (statsPaperCount) statsPaperCount.textContent = `${paperCount} Papers`;
        if (statsChunkCount) statsChunkCount.textContent = `${chunkCount} Chunks`;
    }
}

customElements.define("rag-sidebar", RagSidebar);
export default RagSidebar;
