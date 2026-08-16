import fs from 'fs';
import path from 'path';

/**
 * Node.js arXiv Downloader (No API Key Required)
 * Uses arXiv's public API to query papers and save metadata / PDFs
 */
async function fetchArxivPapers(query = "Retrieval Augmented Generation", maxResults = 10, downloadPdfs = true) {
  console.log(`🔍 Searching arXiv for: "${query}" (Max results: ${maxResults})...\n`);

  const apiUrl = `http://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=${maxResults}&sortBy=relevance&sortOrder=descending`;

  try {
    const response = await fetch(apiUrl);
    const xmlText = await response.text();

    // Simple regex parser for Atom XML entries
    const entries = xmlText.split('<entry>').slice(1);
    const papers = [];

    const pdfDir = path.join(process.cwd(), 'data', 'papers');
    if (downloadPdfs && !fs.existsSync(pdfDir)) {
      fs.mkdirSync(pdfDir, { recursive: true });
    }

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const title = (entry.match(/<title>([\s\S]*?)<\/title>/) || [])[1]?.replace(/\s+/g, ' ').trim() || 'Untitled';
      const idUrl = (entry.match(/<id>([\s\S]*?)<\/id>/) || [])[1]?.trim() || '';
      const paperId = idUrl.split('/abs/')[1] || `paper_${i + 1}`;
      const published = (entry.match(/<published>([\s\S]*?)<\/published>/) || [])[1]?.substring(0, 10) || '';
      const summary = (entry.match(/<summary>([\s\S]*?)<\/summary>/) || [])[1]?.replace(/\s+/g, ' ').trim() || '';
      
      const authorMatches = [...entry.matchAll(/<name>([\s\S]*?)<\/name>/g)];
      const authors = authorMatches.map(m => m[1].trim());

      const pdfUrl = `https://arxiv.org/pdf/${paperId}.pdf`;

      const paperData = { paperId, title, authors, published, summary, pdfUrl };
      papers.push(paperData);

      console.log(`[${i + 1}] Title: ${title}`);
      console.log(`    Authors: ${authors.slice(0, 3).join(', ')}${authors.length > 3 ? ' et al.' : ''}`);
      console.log(`    Published: ${published}`);
      console.log(`    PDF URL: ${pdfUrl}`);

      if (downloadPdfs) {
        const safeTitle = title.replace(/[^a-zA-Z0-9 _-]/g, '').trim().substring(0, 40);
        const filename = `${paperId.replace('/', '_')}_${safeTitle}.pdf`;
        const filePath = path.join(pdfDir, filename);

        if (!fs.existsSync(filePath)) {
          console.log(`    📥 Downloading PDF to: ${filePath}`);
          try {
            const pdfRes = await fetch(pdfUrl);
            if (pdfRes.ok) {
              const arrayBuffer = await pdfRes.arrayBuffer();
              fs.writeFileSync(filePath, Buffer.from(arrayBuffer));
            } else {
              console.log(`    ⚠️ Could not download PDF (Status: ${pdfRes.status})`);
            }
          } catch (err) {
            console.log(`    ⚠️ Error downloading PDF: ${err.message}`);
          }
        } else {
          console.log(`    ✅ PDF already downloaded.`);
        }
      }
      console.log('-'.repeat(75));
    }

    const dataDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

    const metadataPath = path.join(dataDir, 'papers_metadata.json');
    fs.writeFileSync(metadataPath, JSON.stringify(papers, null, 2));

    console.log(`\n✨ Successfully fetched ${papers.length} papers.`);
    console.log(`📁 Metadata saved to: ${metadataPath}`);
    if (downloadPdfs) console.log(`📁 PDFs saved to: ${pdfDir}`);

  } catch (error) {
    console.error('❌ Failed to fetch arXiv papers:', error);
  }
}

fetchArxivPapers("Retrieval Augmented Generation", 10, true);
