import { fetchLatestEval, fetchEvalHistory, runEval } from "../services/api.js";
import { showToast } from "../utils/helpers.js";

/**
 * Web Component for RAG Triad Evaluation Dashboard.
 */
class RagEvaluation extends HTMLElement {
    connectedCallback() {
        this.renderInitialState();
        this.loadData();
    }

    renderInitialState() {
        this.innerHTML = `
            <header class="chat-header">
                <div class="chat-title-info">
                    <h2>RAG Triad Quantitative Evaluation</h2>
                    <p>Automated regression testing and LLM-as-a-judge metric monitoring</p>
                </div>
                <button id="run-eval-btn" class="btn btn-primary" style="background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%); color: #fff; box-shadow: 0 4px 16px rgba(109, 40, 217, 0.4);">
                    <i data-lucide="play-circle"></i> Run Evaluation Suite
                </button>
            </header>

            <section class="message-feed eval-feed" id="eval-container">
                <div class="skeleton-dashboard">
                    <div class="skeleton-card" style="height: 140px;"></div>
                    <div class="skeleton-card" style="height: 300px;"></div>
                </div>
            </section>
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }

        // Setup listener for Run Evaluation
        const runBtn = this.querySelector("#run-eval-btn");
        if (runBtn) {
            runBtn.addEventListener("click", () => this.handleRunEval());
        }
    }

    async loadData() {
        const container = this.querySelector("#eval-container");
        try {
            const [latest, history] = await Promise.all([
                fetchLatestEval(),
                fetchEvalHistory()
            ]);

            if (latest.status === "no_runs_yet") {
                this.renderEmptyState();
            } else {
                this.renderDashboard(latest, history);
            }
        } catch (err) {
            console.error("Error loading evaluation data:", err);
            container.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="alert-triangle" style="color: #ef4444;"></i>
                    <p>Failed to load evaluation data: ${err.message}</p>
                </div>
            `;
            if (window.lucide) window.lucide.createIcons();
        }
    }

    renderEmptyState() {
        const container = this.querySelector("#eval-container");
        container.innerHTML = `
            <div class="welcome-card" id="eval-welcome-card" style="max-width: 650px; margin: 40px auto;">
                <div class="welcome-icon" style="background: rgba(109, 40, 217, 0.15); border-color: rgba(109, 40, 217, 0.3);">
                    <i data-lucide="gauge"></i>
                </div>
                <h2>Establish Your Quantitative Baseline</h2>
                <p>Verify if modifications to chunking strategies, prompt templates, or retrieval architectures improve or regress system accuracy. Running this suite will assess the RAG pipeline on a standardized 5-question synthetic test set using <strong>Gemini</strong> LLM-as-a-judge.</p>
                
                <div class="triad-explain-grid">
                    <div class="triad-explain-card">
                        <h4>Faithfulness</h4>
                        <p>Checks if generated answers are strictly grounded only in the retrieved contexts (preventing hallucinations).</p>
                    </div>
                    <div class="triad-explain-card">
                        <h4>Answer Relevance</h4>
                        <p>Checks if the response directly addresses the question without containing noise or irrelevant fluff.</p>
                    </div>
                    <div class="triad-explain-card">
                        <h4>Context Recall & Precision</h4>
                        <p>Checks if retriever fetched all the required details (Recall) and filtered out noisy text chunks (Precision).</p>
                    </div>
                </div>

                <button id="start-first-eval-btn" class="btn btn-primary" style="margin: 24px auto 0 auto; background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%); color: #fff; padding: 12px 24px;">
                    <i data-lucide="play-circle"></i> Run Baseline Evaluation
                </button>
            </div>
        `;

        if (window.lucide) window.lucide.createIcons();

        const startBtn = this.querySelector("#start-first-eval-btn");
        if (startBtn) {
            startBtn.addEventListener("click", () => this.handleRunEval());
        }
    }

    renderDashboard(latest, history) {
        const container = this.querySelector("#eval-container");
        
        // Format timestamp
        const runDate = new Date(latest.timestamp);
        const formattedDate = runDate.toLocaleString(undefined, { 
            dateStyle: 'medium', 
            timeStyle: 'short' 
        });

        // Compute history entries
        let historyHtml = "";
        if (history && history.length > 0) {
            // Take up to 5 latest runs, reverse to show newest first
            const recentHistory = [...history].reverse().slice(0, 5);
            historyHtml = recentHistory.map((h, i) => {
                const date = new Date(h.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
                return `
                    <div class="history-item">
                        <div class="history-meta">
                            <span class="history-date">${date}</span>
                            <span class="history-score">Score: <strong>${h.average_score.toFixed(3)}</strong></span>
                        </div>
                        <div class="history-bar-row">
                            <div class="history-bar-fill" style="width: ${h.average_score * 100}%"></div>
                        </div>
                    </div>
                `;
            }).join("");
        } else {
            historyHtml = `<p class="text-muted" style="font-size: 13px; text-align: center; padding: 10px;">No historical runs recorded.</p>`;
        }

        // Generate details cards
        const detailsHtml = latest.details.map((item, idx) => {
            return `
                <div class="eval-case-card">
                    <div class="eval-case-header" onclick="this.parentElement.classList.toggle('expanded')">
                        <div class="eval-case-title">
                            <span class="case-number">Test Case #${idx+1}</span>
                            <h4>${item.question}</h4>
                        </div>
                        <div class="eval-case-badges">
                            <span class="eval-badge ${this.getScoreClass(item.faithfulness)}">Faithfulness: ${item.faithfulness.toFixed(2)}</span>
                            <span class="eval-badge ${this.getScoreClass(item.answer_relevancy)}">Relevance: ${item.answer_relevancy.toFixed(2)}</span>
                            <span class="eval-badge ${this.getScoreClass(item.context_precision)}">Precision: ${item.context_precision.toFixed(2)}</span>
                            <span class="eval-badge ${this.getScoreClass(item.context_recall)}">Recall: ${item.context_recall.toFixed(2)}</span>
                            <i data-lucide="chevron-down" class="expand-icon"></i>
                        </div>
                    </div>
                    <div class="eval-case-body">
                        <div class="eval-body-section">
                            <h5><i data-lucide="file-check" class="section-icon text-green"></i> RAG Generated Answer</h5>
                            <div class="eval-text-block generated-answer">${item.answer}</div>
                        </div>
                        <div class="eval-body-section">
                            <h5><i data-lucide="bookmark" class="section-icon text-primary"></i> Ground Truth Reference</h5>
                            <div class="eval-text-block ground-truth">${item.ground_truth}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join("");

        container.innerHTML = `
            <div class="dashboard-grid">
                <!-- Summary Metrics -->
                <div class="summary-card-container">
                    <div class="metric-card main-score">
                        <div class="metric-score-label">Overall RAG Score</div>
                        <div class="metric-big-score">${latest.average_score.toFixed(3)}</div>
                        <div class="score-footer-text">Average of RAG Triad Metrics</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title-row">
                            <span>Faithfulness</span>
                            <span class="metric-score-val ${this.getScoreClass(latest.faithfulness)}">${latest.faithfulness.toFixed(2)}</span>
                        </div>
                        <div class="progress-bar-row">
                            <div class="progress-bar-fill bg-purple" style="width: ${latest.faithfulness * 100}%"></div>
                        </div>
                        <p class="metric-desc">Are the facts grounded strictly in retrieved context?</p>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title-row">
                            <span>Answer Relevance</span>
                            <span class="metric-score-val ${this.getScoreClass(latest.answer_relevancy)}">${latest.answer_relevancy.toFixed(2)}</span>
                        </div>
                        <div class="progress-bar-row">
                            <div class="progress-bar-fill bg-cyan" style="width: ${latest.answer_relevancy * 100}%"></div>
                        </div>
                        <p class="metric-desc">Does the answer address the question directly?</p>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title-row">
                            <span>Context Precision</span>
                            <span class="metric-score-val ${this.getScoreClass(latest.context_precision)}">${latest.context_precision.toFixed(2)}</span>
                        </div>
                        <div class="progress-bar-row">
                            <div class="progress-bar-fill bg-green" style="width: ${latest.context_precision * 100}%"></div>
                        </div>
                        <p class="metric-desc">Did retriever fetch clean chunks free of noise?</p>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title-row">
                            <span>Context Recall</span>
                            <span class="metric-score-val ${this.getScoreClass(latest.context_recall)}">${latest.context_recall.toFixed(2)}</span>
                        </div>
                        <div class="progress-bar-row">
                            <div class="progress-bar-fill bg-indigo" style="width: ${latest.context_recall * 100}%"></div>
                        </div>
                        <p class="metric-desc">Did retriever fetch all necessary information?</p>
                    </div>
                </div>

                <div class="dashboard-split-row">
                    <!-- Historical Timeline -->
                    <div class="timeline-panel card">
                        <div class="card-header">
                            <i data-lucide="history"></i>
                            <h3>Evaluation History</h3>
                        </div>
                        <div class="card-content history-list-container">
                            ${historyHtml}
                        </div>
                        <div class="card-footer">
                            <span>Last run: ${formattedDate}</span>
                        </div>
                    </div>
                    
                    <!-- Run Details -->
                    <div class="details-panel">
                        <div class="details-panel-header">
                            <i data-lucide="clipboard-list"></i>
                            <h3>Detailed Test Case Logs</h3>
                        </div>
                        <div class="details-list-container">
                            ${detailsHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;

        if (window.lucide) window.lucide.createIcons();
    }

    getScoreClass(score) {
        if (score >= 0.8) return "score-green";
        if (score >= 0.5) return "score-yellow";
        return "score-red";
    }

    async handleRunEval() {
        const runBtn = this.querySelector("#run-eval-btn");
        const container = this.querySelector("#eval-container");

        // 1. Set UI into loading state
        if (runBtn) {
            runBtn.disabled = true;
            runBtn.innerHTML = `<i data-lucide="loader" class="spin"></i> Running Evaluation...`;
        }

        container.innerHTML = `
            <div class="welcome-card eval-loading-state" style="max-width: 600px; margin: 60px auto; padding: 40px 30px;">
                <div class="eval-loading-orb">
                    <div class="orb-ring pulse-1"></div>
                    <div class="orb-ring pulse-2"></div>
                    <div class="orb-core">
                        <i data-lucide="cpu" style="width: 32px; height: 32px; color: var(--accent-cyan);"></i>
                    </div>
                </div>
                
                <h3 style="margin-top: 30px; font-size: 18px; color: #fff;">LLM-as-a-Judge Evaluating Pipeline</h3>
                <p style="font-size: 13.5px; color: var(--text-muted); line-height: 1.6; margin-bottom: 24px;">
                    Running Ragas validation across the 5 synthetic test cases. This makes API calls to Gemini and calculates the RAG Triad scores. Please wait about 20-30 seconds...
                </p>

                <div class="loader-steps-list">
                    <div class="loader-step active" id="step-1">
                        <span class="step-indicator"></span>
                        <span>Querying Chroma & BM25 retrieval engines...</span>
                    </div>
                    <div class="loader-step" id="step-2">
                        <span class="step-indicator"></span>
                        <span>Generating responses from Gemini 2.5 Flash...</span>
                    </div>
                    <div class="loader-step" id="step-3">
                        <span class="step-indicator"></span>
                        <span>Scoring faithfulness (groundedness check)...</span>
                    </div>
                    <div class="loader-step" id="step-4">
                        <span class="step-indicator"></span>
                        <span>Calculating Answer Relevance & Context accuracy...</span>
                    </div>
                </div>
            </div>
        `;

        if (window.lucide) window.lucide.createIcons();

        // Animate fake loading steps
        let currentStep = 1;
        const stepInterval = setInterval(() => {
            currentStep++;
            if (currentStep <= 4) {
                const prev = this.querySelector(`#step-${currentStep-1}`);
                const curr = this.querySelector(`#step-${currentStep}`);
                if (prev) {
                    prev.classList.remove("active");
                    prev.classList.add("done");
                }
                if (curr) {
                    curr.classList.add("active");
                }
            }
        }, 5000);

        try {
            // 2. Trigger evaluation run
            const newResults = await runEval();
            clearInterval(stepInterval);

            showToast("RAG Triad Evaluation completed successfully!");
            
            // 3. Reload dashboard
            this.renderInitialState();
            await this.loadData();

        } catch (err) {
            clearInterval(stepInterval);
            console.error("Evaluation run failed:", err);
            showToast("Failed to run evaluation suite: " + err.message, "error");
            
            // Revert state
            this.renderInitialState();
            await this.loadData();
        }
    }
}

customElements.define("rag-evaluation", RagEvaluation);
export default RagEvaluation;
