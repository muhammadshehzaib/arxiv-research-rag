import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const moduleUrl = text => `data:text/javascript;base64,${Buffer.from(text).toString("base64")}`;
const helpers = moduleUrl(await readFile(new URL("../static/js/utils/helpers.js", import.meta.url), "utf8"));
const api = moduleUrl("export const fetchLatestEval=()=>{}, fetchEvalHistory=()=>{}, runEval=()=>{};");
let source = await readFile(new URL("../static/js/components/RagEvaluation.js", import.meta.url), "utf8");
source = source.replaceAll('"../utils/helpers.js"', JSON.stringify(helpers))
    .replaceAll('"../services/api.js"', JSON.stringify(api));
globalThis.HTMLElement = class {};
globalThis.window = {};
globalThis.customElements = { define() {} };
const { default: RagEvaluation } = await import(moduleUrl(source));
const panel = new RagEvaluation();
const container = { innerHTML: "" };
panel.querySelector = () => container;
const metrics = { faithfulness: null, answer_relevancy: null, context_precision: null, context_recall: null, average_score: null };
const latest = { ...metrics, status: "incomplete", timestamp: "2026-10-18T00:00:00Z",
    details: [{ ...metrics, question: '<img onerror="attack">', answer: "<script>attack</script>", ground_truth: "Truth" }] };
panel.renderDashboard(latest, [{ ...latest, details: [] }]);
assert.match(container.innerHTML, /N\/A/);
assert.match(container.innerHTML, /Incomplete evaluation/);
assert.doesNotMatch(container.innerHTML, /<img|<script>|NaN/);
panel.renderDashboard({ ...latest, status: "legacy_unverified" }, []);
assert.match(container.innerHTML, /unverified/);
console.log("Evaluation failure labels and escaped rendering: all checks passed.");
