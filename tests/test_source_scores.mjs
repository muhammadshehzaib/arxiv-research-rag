import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// Load the browser ES module without changing the Node package's module type.
const source = await readFile(new URL("../static/js/utils/helpers.js", import.meta.url), "utf8");
const helpersUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { formatRelevanceScore } = await import(helpersUrl);

assert.equal(formatRelevanceScore({ relevance_score: 0.9 }), "Relevance: 0.900");
assert.equal(formatRelevanceScore({ relevance_score: 0 }), "Relevance: 0.000");
assert.equal(formatRelevanceScore({ relevance_score: null, is_live_mcp: true }), "Live source · unscored");
assert.equal(formatRelevanceScore({ confidence: 0.95, distance: 0.05 }), "Relevance not scored");
for (const score of [null, undefined, NaN, Infinity, -0.1, 1.1, "0.95"]) {
    assert.equal(formatRelevanceScore({ relevance_score: score }), "Relevance not scored");
}

// Render the real source component with a minimal DOM; no browser packages needed.
globalThis.HTMLElement = class {};
globalThis.window = {};
globalThis.document = { createElement: () => ({}) };
globalThis.customElements = { define() {} };
const component = (await readFile(new URL("../static/js/components/RagSources.js", import.meta.url), "utf8"))
    .replace('"../utils/helpers.js"', JSON.stringify(helpersUrl));
const { default: RagSources } = await import(`data:text/javascript;base64,${Buffer.from(component).toString("base64")}`);
const container = { innerHTML: "", cards: [], appendChild(card) { this.cards.push(card); } };
const panel = new RagSources();
panel.querySelector = () => container;
panel.setSources([
    { index: 1, title: "Local paper", relevance_score: 0.9 },
    { index: 2, title: "Live paper", relevance_score: null, is_live_mcp: true },
    { index: 3, title: "Legacy paper", confidence: 0.95, distance: 0.05 },
]);
assert.match(container.cards[0].innerHTML, /Relevance: 0\.900/);
assert.match(container.cards[1].innerHTML, /Live source · unscored/);
assert.match(container.cards[2].innerHTML, /Relevance not scored/);
assert.doesNotMatch(container.cards.map(card => card.innerHTML).join(""), /95%|NaN|Similarity:/);
console.log("Source relevance labels and component rendering: all checks passed.");
