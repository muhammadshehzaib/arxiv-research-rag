import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai

# Ensure stdout/stderr use UTF-8 encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Set up API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Import project's query logic
from query_rag import init_services, query_rag, get_bm25_index

# Define fixed test evaluation set matching downloaded papers
EVAL_QUESTIONS = [
    {
        "question": "What is AR-RAG and what are its two parallel frameworks?",
        "ground_truth": "AR-RAG (Autoregressive Retrieval Augmentation for Image Generation) is a paradigm that enhances image generation by autoregressively incorporating patch-level retrievals. Its two parallel frameworks are: (1) Distribution-Augmentation in Decoding (DAiD), a training-free decoding strategy, and (2) Feature-Augmentation in Decoding (FAiD), a parameter-efficient fine-tuning method."
    },
    {
        "question": "What is Factually and how does it alert users to potential falsehoods?",
        "ground_truth": "Factually is a proactive, wearable fact-checking system integrated into devices like smartwatches or rings. It discreetly alerts users to potential falsehoods via vibrotactile feedback."
    },
    {
        "question": "What is EHR-RAGp and how does its retrieval module work?",
        "ground_truth": "EHR-RAGp is a retrieval-augmented foundation model for Electronic Health Records (EHR). It uses a prototype-guided retrieval module that acts as an alignment mechanism to estimate the relevance of retrieved historical chunks with respect to a given prediction task."
    },
    {
        "question": "What was the highest ROUGE-1 score achieved in the automated literature review research, and by which model?",
        "ground_truth": "The highest ROUGE-1 score achieved in the research was 0.364, which was accomplished by the Large Language Model GPT-3.5-turbo."
    },
    {
        "question": "What distinction does the paper on AI systems and critical thinking propose?",
        "ground_truth": "The position paper proposes the distinction between demonstrated and performed critical thinking in the era of generative AI."
    }
]

def parse_json_array(text):
    """
    Safely parse a JSON array from LLM output.
    """
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception:
        # Fallback split
        lines = [line.strip("- *1234567890. \"'") for line in text.split("\n") if line.strip()]
        return lines

def generate_content_with_retry(model, prompt, max_retries=5):
    """
    Call Gemini generate_content with exponential backoff on 429 Rate Limits.
    """
    delay = 6  # Start with a 6-second delay as typical free tier windows reset every minute
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "limit" in err_msg or "resource_exhausted" in err_msg:
                if attempt < max_retries - 1:
                    print(f"      ⚠️ API Rate limit hit. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    continue
            raise e

def get_embedding_with_retry(model_name, content, max_retries=5):
    """
    Call Gemini embed_content with exponential backoff on 429 Rate Limits.
    """
    delay = 6
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            return genai.embed_content(model=model_name, content=content, task_type="retrieval_query")
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "limit" in err_msg or "resource_exhausted" in err_msg:
                if attempt < max_retries - 1:
                    print(f"      ⚠️ API Rate limit hit for embedding. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    continue
            raise e

def calculate_faithfulness(model, answer, contexts):
    """
    Evaluate if the answer is derived strictly from contexts (no hallucinations).
    """
    if not answer or not contexts:
        return 1.0
    
    # 1. Extract statements from answer
    prompt_extract = f"""Given a text, extract all atomic statements/factual claims from it.
Each statement should be a single standalone sentence containing exactly one fact.
Output the result as a JSON array of strings. Do not include any formatting other than the JSON array.

Text: {answer}
Statements:"""
    try:
        res = generate_content_with_retry(model, prompt_extract).text
        statements = parse_json_array(res)
    except Exception as e:
        print(f"      ⚠️ Faithfulness statement extraction failed: {e}")
        return 0.5
        
    if not statements:
        return 1.0
        
    # 2. Verify each statement against joined contexts
    joined_context = "\n---\n".join(contexts)
    supported_count = 0
    
    for statement in statements:
        prompt_verify = f"""Analyze if the given Statement is factually supported by the Context.
Output either 'YES' if it is supported, or 'NO' if it is not supported or if the context does not contain enough information.
Do not explain, output only 'YES' or 'NO'.

Context:
{joined_context}

Statement: {statement}
Supported:"""
        try:
            res_verify = generate_content_with_retry(model, prompt_verify).text.strip().upper()
            if "YES" in res_verify:
                supported_count += 1
        except Exception:
            pass
            
    return float(supported_count / len(statements))

def calculate_answer_relevance(model, question, answer, emb_model):
    """
    Evaluate how pertinent the answer is to the user question.
    """
    if not answer:
        return 0.0
        
    print(f"      [Step 1] Querying Gemini LLM to extract 3 questions answered by the generation...")
    # 1. Generate 3 questions based on generated answer
    prompt_gen = f"""Based on the given Answer, generate exactly 3 distinct, specific questions that this answer directly and fully answers.
Output the result as a JSON array of 3 strings. Do not include any formatting other than the JSON array.

Answer: {answer}
Questions:"""
    try:
        res = generate_content_with_retry(model, prompt_gen).text
        gen_questions = parse_json_array(res)
        print(f"      [Step 1] Gemini generated synthetic questions:")
        for idx, g_q in enumerate(gen_questions):
            print(f"         ├─ Question {idx+1}: \"{g_q}\"")
    except Exception as e:
        print(f"      ⚠️ Relevance question generation failed: {e}")
        return 0.5
        
    if not gen_questions:
        return 0.0
        
    # 2. Compute similarity
    print(f"      [Step 2] Querying Gemini Embeddings and computing Cosine Similarities against original query...")
    try:
        res_q = get_embedding_with_retry(emb_model, question)
        v_q = np.array(res_q['embedding'])
        
        similarities = []
        for idx, g_q in enumerate(gen_questions):
            res_g = get_embedding_with_retry(emb_model, g_q)
            v_g = np.array(res_g['embedding'])
            
            dot = np.dot(v_q, v_g)
            norm_q = np.linalg.norm(v_q)
            norm_g = np.linalg.norm(v_g)
            if norm_q > 0 and norm_g > 0:
                sim = dot / (norm_q * norm_g)
                similarities.append(float(sim))
                print(f"         ├─ CosineSimilarity(Q_original, Q_{idx+1}): {sim:.4f}")
                
        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        print(f"      [Step 2] Final relevance score (average): {avg_sim:.4f}")
        return avg_sim
    except Exception as e:
        print(f"      ⚠️ Relevance embedding calculation failed: {e}")
        return 0.5

def calculate_context_recall(model, ground_truth, contexts):
    """
    Evaluate if the retriever fetched all necessary information to answer the question.
    """
    if not ground_truth or not contexts:
        return 1.0
        
    # 1. Extract statements from ground truth
    prompt_extract = f"""Given a text, extract all atomic statements/factual claims from it.
Each statement should be a single standalone sentence containing exactly one fact.
Output the result as a JSON array of strings. Do not include any formatting other than the JSON array.

Text: {ground_truth}
Statements:"""
    try:
        res = generate_content_with_retry(model, prompt_extract).text
        gt_statements = parse_json_array(res)
    except Exception as e:
        print(f"      ⚠️ Recall statement extraction failed: {e}")
        return 0.5
        
    if not gt_statements:
        return 1.0
        
    # 2. Verify each statement
    joined_context = "\n---\n".join(contexts)
    supported_count = 0
    
    for statement in gt_statements:
        prompt_verify = f"""Analyze if the given Statement is factually supported by the Context.
Output either 'YES' if it is supported, or 'NO' if it is not supported or if the context does not contain enough information.
Do not explain, output only 'YES' or 'NO'.

Context:
{joined_context}

Statement: {statement}
Supported:"""
        try:
            res_verify = generate_content_with_retry(model, prompt_verify).text.strip().upper()
            if "YES" in res_verify:
                supported_count += 1
        except Exception:
            pass
            
    return float(supported_count / len(gt_statements))

def calculate_context_precision(model, question, contexts):
    """
    Evaluate if the retrieved contexts are clean of noise.
    """
    if not contexts:
        return 0.0
        
    relevant_count = 0
    for chunk in contexts:
        prompt_verify = f"""Analyze if the given Context Chunk is relevant and useful to answer the Question.
Output either 'YES' if it is relevant, or 'NO' if it is not relevant.
Do not explain, output only 'YES' or 'NO'.

Question: {question}
Context Chunk: {chunk}
Relevant:"""
        try:
            res_verify = generate_content_with_retry(model, prompt_verify).text.strip().upper()
            if "YES" in res_verify:
                relevant_count += 1
        except Exception:
            pass
            
    return float(relevant_count / len(contexts))

def run_evaluation():
    print("🚀 Initializing Evaluation...")
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY is not configured.")
        return None

    # Initialize RAG collections
    try:
        collection = init_services()
    except Exception as e:
        print(f"❌ Failed to connect to RAG DB: {e}")
        return None
        
    bm25, bm25_chunks = get_bm25_index()
    
    llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    emb_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    
    print(f"🤖 Evaluation LLM: {llm_model}")
    print(f"🌐 Evaluation Embeddings: {emb_model}")
    
    model = genai.GenerativeModel(llm_model)
    
    evaluation_results = []
    
    # Loop over test questions
    for idx, test_case in enumerate(EVAL_QUESTIONS):
        q = test_case["question"]
        gt = test_case["ground_truth"]
        
        print(f"\n📝 Running Test Case [{idx+1}/{len(EVAL_QUESTIONS)}]: '{q}'")
        
        # 1. Run query on RAG system
        try:
            answer, sources = query_rag(
                collection,
                q,
                num_results=3,
                bm25=bm25,
                bm25_chunks=bm25_chunks
            )
            contexts = [src["text"] for src in sources if "text" in src]
        except Exception as e:
            print(f"   ❌ Query execution failed: {e}")
            answer = f"Error: {e}"
            contexts = []
            
        # 2. Compute metrics
        print("   🧠 Running LLM-as-a-judge scorers...")
        f_score = calculate_faithfulness(model, answer, contexts)
        ar_score = calculate_answer_relevance(model, q, answer, emb_model)
        cp_score = calculate_context_precision(model, q, contexts)
        cr_score = calculate_context_recall(model, gt, contexts)
        
        print(f"   ✅ Faithfulness: {f_score:.2f} | Relevance: {ar_score:.2f} | Precision: {cp_score:.2f} | Recall: {cr_score:.2f}")
        
        evaluation_results.append({
            "question": q,
            "answer": answer,
            "ground_truth": gt,
            "faithfulness": f_score,
            "answer_relevancy": ar_score,
            "context_precision": cp_score,
            "context_recall": cr_score
        })
        
        # Rate limit delay between queries
        time.sleep(2)
        
    # Calculate overall averages
    avg_f = float(np.mean([r["faithfulness"] for r in evaluation_results]))
    avg_ar = float(np.mean([r["answer_relevancy"] for r in evaluation_results]))
    avg_cp = float(np.mean([r["context_precision"] for r in evaluation_results]))
    avg_cr = float(np.mean([r["context_recall"] for r in evaluation_results]))
    avg_total = float(np.mean([avg_f, avg_ar, avg_cp, avg_cr]))
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": avg_f,
        "answer_relevancy": avg_ar,
        "context_precision": avg_cp,
        "context_recall": avg_cr,
        "average_score": avg_total,
        "details": evaluation_results
    }
    
    save_results(summary)
    return summary

def save_results(summary):
    os.makedirs("data", exist_ok=True)
    
    # Save latest results
    results_path = os.path.join("data", "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved latest evaluation results to: {results_path}")
        
    # Save history
    history_path = os.path.join("data", "eval_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
            
    history.append({
        "timestamp": summary["timestamp"],
        "faithfulness": summary["faithfulness"],
        "answer_relevancy": summary["answer_relevancy"],
        "context_precision": summary["context_precision"],
        "context_recall": summary["context_recall"],
        "average_score": summary["average_score"]
    })
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"💾 Updated evaluation history in: {history_path}")

def print_summary_table(summary):
    if not summary:
        return
    print("\n" + "="*80)
    print(f"🏆 RAG TRIAD QUANTITATIVE EVALUATION SUMMARY ({summary['timestamp'][:10]})")
    print("="*80)
    print(f"  Faithfulness:       {summary['faithfulness']:.4f} (Groundedness of generation)")
    print(f"  Answer Relevance:   {summary['answer_relevancy']:.4f} (Directness of response)")
    print(f"  Context Precision:  {summary['context_precision']:.4f} (Noise filtering quality)")
    print(f"  Context Recall:     {summary['context_recall']:.4f} (Completeness of retrieval)")
    print("-"*80)
    print(f"  OVERALL RAG SCORE:  {summary['average_score']:.4f}")
    print("="*80 + "\n")
    
    print("Detailed Log Cases:")
    for idx, case in enumerate(summary["details"]):
        print(f"\n[{idx+1}] Question: {case['question']}")
        print(f"    Faithfulness: {case['faithfulness']:.2f} | Relevancy: {case['answer_relevancy']:.2f} | Precision: {case['context_precision']:.2f} | Recall: {case['context_recall']:.2f}")

if __name__ == "__main__":
    print("🎬 Starting Automated RAG Triad Evaluation Suite...")
    summary = run_evaluation()
    print_summary_table(summary)
