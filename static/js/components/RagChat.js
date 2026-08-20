import { formatMarkdown } from "../utils/helpers.js";

/**
 * Web Component for RAG Chat interface.
 */
class RagChat extends HTMLElement {
    connectedCallback() {
        this.render();
        this.setupListeners();
    }

    render() {
        this.innerHTML = `
            <header class="chat-header">
                <div class="chat-title-info">
                    <h2>Interactive RAG Chat</h2>
                    <p>Grounding answers strictly in the scientific literature database</p>
                </div>
                <button id="toggle-sources-btn" class="btn btn-icon" title="View Retrieved Sources">
                    <i data-lucide="file-text"></i>
                    <span class="badge" id="source-badge">0</span>
                </button>
            </header>

            <section class="message-feed" id="message-feed">
                <!-- Welcome Card -->
                <div class="welcome-card" id="welcome-card">
                    <div class="welcome-icon">
                        <i data-lucide="sparkles"></i>
                    </div>
                    <h2>Welcome to the arXiv RAG Explorer!</h2>
                    <p>Ask queries about Retrieval-Augmented Generation, and the model will retrieve the relevant sections from the 50 downloaded arXiv papers to generate an answer with inline citations.</p>
                    
                    <div class="example-questions">
                        <h3>Try asking:</h3>
                        <button class="example-btn">What is Autoregressive Retrieval Augmentation (AR-RAG)?</button>
                        <button class="example-btn">How does the Ragas framework evaluate RAG pipelines?</button>
                        <button class="example-btn">What are the main security threats in retrieval stages?</button>
                    </div>
                </div>
            </section>

            <footer class="input-panel">
                <form id="chat-form" class="input-form">
                    <div class="input-wrapper">
                        <textarea id="chat-input" placeholder="Type your query about arXiv papers..." rows="1" required></textarea>
                        <button type="submit" class="send-btn" id="send-btn">
                            <i data-lucide="send"></i>
                        </button>
                    </div>
                </form>
            </footer>
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    setupListeners() {
        const chatForm = this.querySelector("#chat-form");
        const chatInput = this.querySelector("#chat-input");
        const toggleSourcesBtn = this.querySelector("#toggle-sources-btn");

        // Toggle sources event
        toggleSourcesBtn.addEventListener("click", () => {
            this.dispatchEvent(new CustomEvent("toggle-sources", {
                bubbles: true
            }));
        });

        // Click on suggestion buttons
        this.querySelectorAll(".example-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                chatInput.value = btn.textContent;
                chatInput.focus();
            });
        });

        // Submit query event
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            this.handleQuerySubmit();
        });

        // Key down submit on Enter key without shift key
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this.handleQuerySubmit();
            }
        });
    }

    handleQuerySubmit() {
        const chatInput = this.querySelector("#chat-input");
        const query = chatInput.value.trim();
        if (!query) return;

        // Reset input
        this.resetInput();

        // Hide welcome card
        const welcomeCard = this.querySelector("#welcome-card");
        if (welcomeCard) {
            welcomeCard.style.display = "none";
        }

        // Dispatch submit event
        this.dispatchEvent(new CustomEvent("query-submit", {
            detail: { query },
            bubbles: true
        }));
    }

    resetInput() {
        const chatInput = this.querySelector("#chat-input");
        chatInput.value = "";
        chatInput.style.height = "auto";
    }

    /**
     * Append a message bubble to the chat feed.
     * @param {string} text Markdown or plaintext of the message.
     * @param {string} sender 'user' or 'ai'
     */
    appendMessage(text, sender) {
        const feed = this.querySelector("#message-feed");
        const wrapper = document.createElement("div");
        wrapper.className = `message-wrapper ${sender}`;

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (sender === "user") {
            bubble.textContent = text;
        } else {
            bubble.innerHTML = formatMarkdown(text);
        }

        wrapper.appendChild(bubble);
        feed.appendChild(wrapper);
        this.scrollToBottom();
    }

    /**
     * Append the loading placeholder bubble.
     * @returns {string} Id of the loading bubble.
     */
    appendLoadingBubble() {
        const feed = this.querySelector("#message-feed");
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
        this.scrollToBottom();
        return bubbleId;
    }

    /**
     * Remove bubble by ID (used for clearing the loading bubble).
     * @param {string} id 
     */
    removeBubble(id) {
        const el = this.querySelector(`#${id}`);
        if (el) el.remove();
    }

    /**
     * Scroll the feed container to the bottom.
     */
    scrollToBottom() {
        const feed = this.querySelector("#message-feed");
        if (feed) {
            feed.scrollTop = feed.scrollHeight;
        }
    }

    /**
     * Update the number of sources shown in the header badge.
     * @param {number} count 
     */
    updateSourceBadgeCount(count) {
        const badge = this.querySelector("#source-badge");
        if (badge) {
            badge.textContent = count;
        }
    }
}

customElements.define("rag-chat", RagChat);
export default RagChat;
