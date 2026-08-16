import fs from 'fs';
import path from 'path';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';

/**
 * PDF Text Extractor & Sliding Window Chunker for RAG
 * - Extracts raw text from arXiv PDFs
 * - Cleans and normalizes formatting
 * - Splits into sliding window chunks (~500 words with 50 words overlap)
 * - Attaches metadata for payload indexing
 */

const CHUNK_SIZE = 500;  // Target words per chunk (~500 tokens)
const OVERLAP = 50;      // Overlap between adjacent chunks (~50 tokens)

function cleanText(text) {
  return text
    // Replace hyphenated line breaks (e.g. "retriev-\nal" -> "retrieval")
    .replace(/(\w+)-\s*\n\s*(\w+)/g, '$1$2')
    // Normalize newlines to spaces
    .replace(/\r?\n|\r/g, ' ')
    // Replace multiple spaces with a single space
    .replace(/\s+/g, ' ')
    .trim();
}

function chunkText(text, chunkSize = CHUNK_SIZE, overlap = OVERLAP) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const chunks = [];

  if (words.length === 0) return chunks;

  let i = 0;
  let index = 0;

  while (i < words.length) {
    const end = Math.min(i + chunkSize, words.length);
    const chunkWords = words.slice(i, end);
    const chunkContent = chunkWords.join(' ');

    chunks.push({
      chunk_index: index,
      text: chunkContent,
      word_count: chunkWords.length,
      start_word: i,
      end_word: end
    });

    index++;
    // Move window forward by chunkSize - overlap
    i += (chunkSize - overlap);

    if (i >= words.length || (words.length - i < overlap && chunks.length > 0)) {
      break;
    }
  }

  return chunks;
}

async function processAllPapers() {
  const dataDir = path.join(process.cwd(), 'data');
  const papersDir = path.join(dataDir, 'papers');
  const metadataPath = path.join(dataDir, 'papers_metadata.json');

  if (!fs.existsSync(metadataPath)) {
    console.error(`❌ Metadata file not found at ${metadataPath}. Run arxiv_downloader.js first!`);
    return;
  }

  const papersMetadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
  console.log(`📚 Found metadata for ${papersMetadata.length} papers. Processing PDFs...\n`);

  const allChunks = [];
  let processedCount = 0;
  let totalChunksCount = 0;

  for (const paper of papersMetadata) {
    const safeTitle = paper.title.replace(/[^a-zA-Z0-9 _-]/g, '').trim().substring(0, 40);
    const filename = `${paper.paperId.replace('/', '_')}_${safeTitle}.pdf`;
    const pdfPath = path.join(papersDir, filename);

    if (!fs.existsSync(pdfPath)) {
      console.warn(`⚠️ PDF file not found: ${pdfPath}. Skipping.`);
      continue;
    }

    try {
      console.log(`📄 Extracting text from: "${paper.title}"`);
      const dataBuffer = fs.readFileSync(pdfPath);
      const parse = typeof pdfParse === 'function' ? pdfParse : pdfParse.default || pdfParse;
      const pdfData = await parse(dataBuffer);

      const cleanedText = cleanText(pdfData.text);
      const rawChunks = chunkText(cleanedText, CHUNK_SIZE, OVERLAP);

      console.log(`   └─ Extracted ${pdfData.numpages} pages | Total Words: ${cleanedText.split(/\s+/).length} | Chunks Created: ${rawChunks.length}`);

      for (const chunk of rawChunks) {
        const enrichedChunk = {
          chunk_id: `${paper.paperId}_c${chunk.chunk_index}`,
          paper_id: paper.paperId,
          title: paper.title,
          authors: paper.authors,
          published: paper.published,
          pdf_url: paper.pdfUrl,
          categories: paper.categories || [],
          total_pages: pdfData.numpages,
          ...chunk
        };
        allChunks.push(enrichedChunk);
      }

      processedCount++;
      totalChunksCount += rawChunks.length;

    } catch (err) {
      console.error(`❌ Failed to parse PDF for ${paper.title}: ${err.message}`);
    }
  }

  // Save all enriched chunks to JSON
  const chunksOutputPath = path.join(dataDir, 'paper_chunks.json');
  fs.writeFileSync(chunksOutputPath, JSON.stringify(allChunks, null, 2));

  console.log(`\n==================================================`);
  console.log(`🎉 PDF Processing Complete!`);
  console.log(`✔ Papers Processed: ${processedCount}/${papersMetadata.length}`);
  console.log(`✔ Total RAG Chunks Generated: ${totalChunksCount}`);
  console.log(`💾 Saved chunks dataset to: ${chunksOutputPath}`);
  console.log(`==================================================\n`);
}

processAllPapers();
