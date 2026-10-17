/**
 * Book Discovery & Recommendation Frontend Logic
 * ===============================================
 * Handles hybrid search queries, multi-modal autocomplete,
 * recommendation drawer modals, and real-time latency telemetry.
 */

const API_BASE = "http://localhost:8000";

// State
let currentTab = "all";
let debounceTimer = null;
let selectedAcIndex = -1;
let currentAcItems = [];

// DOM Elements
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const clearSearchBtn = document.getElementById("clearSearchBtn");
const autocompleteDropdown = document.getElementById("autocompleteDropdown");
const acResults = document.getElementById("acResults");
const acTabs = document.querySelectorAll(".ac-tab");
const booksGrid = document.getElementById("booksGrid");
const resultsCount = document.getElementById("resultsCount");
const toggleCache = document.getElementById("toggleCache");
const toggleReranker = document.getElementById("toggleReranker");

// Telemetry Elements
const telemetryStatus = document.getElementById("telemetryStatus");
const telTotal = document.getElementById("telTotal");
const telCache = document.getElementById("telCache");
const telDense = document.getElementById("telDense");
const telBM25 = document.getElementById("telBM25");
const telRerank = document.getElementById("telRerank");
const telCandidates = document.getElementById("telCandidates");

// Recommendation Modal Elements
const recommendationModal = document.getElementById("recommendationModal");
const closeRecModal = document.getElementById("closeRecModal");
const recModalTitle = document.getElementById("recModalTitle");
const recModalGrid = document.getElementById("recModalGrid");

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  fetchSystemHealth();
  setupEventListeners();

  // Run an initial search to populate high-relevance books
  performSearch("dystopian surveillance state and psychological control");
});

function setupEventListeners() {
  // Search button
  searchBtn.addEventListener("click", () => {
    const q = searchInput.value.trim();
    if (q) performSearch(q);
  });

  // Enter key in input
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      if (selectedAcIndex >= 0 && currentAcItems[selectedAcIndex]) {
        applyAutocompleteItem(currentAcItems[selectedAcIndex]);
      } else {
        closeAutocomplete();
        const q = searchInput.value.trim();
        if (q) performSearch(q);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      navigateAutocomplete(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      navigateAutocomplete(-1);
    } else if (e.key === "Escape") {
      closeAutocomplete();
    }
  });

  // Debounced input for autocomplete
  searchInput.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    clearSearchBtn.style.display = val ? "block" : "none";

    clearTimeout(debounceTimer);
    if (val.length >= 2) {
      debounceTimer = setTimeout(() => fetchAutocomplete(val), 200);
    } else {
      closeAutocomplete();
    }
  });

  // Clear button
  clearSearchBtn.addEventListener("click", () => {
    searchInput.value = "";
    clearSearchBtn.style.display = "none";
    closeAutocomplete();
    searchInput.focus();
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-box-wrapper")) {
      closeAutocomplete();
    }
  });

  // Autocomplete Tab Switch
  acTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      acTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentTab = tab.getAttribute("data-type");
      const q = searchInput.value.trim();
      if (q.length >= 2) fetchAutocomplete(q);
    });
  });

  // Quick Prompt Chips
  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      searchInput.value = q;
      clearSearchBtn.style.display = "block";
      closeAutocomplete();
      performSearch(q);
    });
  });

  // Modal Close
  closeRecModal.addEventListener("click", () => {
    recommendationModal.style.display = "none";
  });

  recommendationModal.addEventListener("click", (e) => {
    if (e.target === recommendationModal) {
      recommendationModal.style.display = "none";
    }
  });
}

// Check Backend Health
async function fetchSystemHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return;
    const data = await res.json();

    const pillEs = document.getElementById("pill-es");
    if (data.elasticsearch?.connected) {
      pillEs.innerHTML = `<span class="status-dot"></span><span>ES Index: <b>${data.vector_catalog?.indexed_count || 50000} docs</b></span>`;
    }

    const pillRedis = document.getElementById("pill-redis");
    if (data.redis?.connected) {
      pillRedis.innerHTML = `<span class="status-dot"></span><span>Redis: <b>Active (0ms)</b></span>`;
    }
  } catch (err) {
    console.warn("Backend API not reachable at", API_BASE, err);
  }
}

// Fetch Autocomplete Suggestions
async function fetchAutocomplete(query) {
  try {
    const res = await fetch(`${API_BASE}/api/autocomplete?q=${encodeURIComponent(query)}&type=${currentTab}&limit=8`);
    if (!res.ok) return;
    const data = await res.json();
    renderAutocomplete(data);
  } catch (err) {
    console.error("Autocomplete error:", err);
  }
}

function renderAutocomplete(data) {
  acResults.innerHTML = "";
  currentAcItems = [];
  selectedAcIndex = -1;

  const { titles = [], authors = [], genres = [] } = data;

  // Render Authors
  authors.forEach((a) => {
    const item = { type: "author", text: a.name, icon: "👤", sub: "Author" };
    currentAcItems.push(item);
  });

  // Render Titles
  titles.forEach((t) => {
    const authorStr = t.authors?.length ? `by ${t.authors.join(", ")}` : "";
    const yearStr = t.year ? `(${t.year})` : "";
    const item = {
      type: "title",
      text: t.title,
      icon: "📚",
      sub: `${authorStr} ${yearStr}`.trim(),
      id: t.book_id,
    };
    currentAcItems.push(item);
  });

  // Render Genres
  genres.forEach((g) => {
    const item = { type: "genre", text: g.genre, icon: "🏷️", sub: "Genre / Shelf Tag" };
    currentAcItems.push(item);
  });

  if (currentAcItems.length === 0) {
    closeAutocomplete();
    return;
  }

  currentAcItems.forEach((item, idx) => {
    const el = document.createElement("div");
    el.className = "ac-item";
    el.innerHTML = `
      <span class="ac-icon">${item.icon}</span>
      <div>
        <div class="ac-item-title">${escapeHtml(item.text)}</div>
        <div class="ac-item-sub">${escapeHtml(item.sub)}</div>
      </div>
    `;
    el.addEventListener("click", () => applyAutocompleteItem(item));
    acResults.appendChild(el);
  });

  autocompleteDropdown.style.display = "block";
}

function navigateAutocomplete(direction) {
  const items = acResults.querySelectorAll(".ac-item");
  if (!items.length) return;

  items.forEach((it) => it.classList.remove("selected"));

  selectedAcIndex += direction;
  if (selectedAcIndex < 0) selectedAcIndex = items.length - 1;
  if (selectedAcIndex >= items.length) selectedAcIndex = 0;

  items[selectedAcIndex].classList.add("selected");
  items[selectedAcIndex].scrollIntoView({ block: "nearest" });
}

function applyAutocompleteItem(item) {
  searchInput.value = item.text;
  closeAutocomplete();
  performSearch(item.text);
}

function closeAutocomplete() {
  autocompleteDropdown.style.display = "none";
  selectedAcIndex = -1;
  currentAcItems = [];
}

// Perform Hybrid Search
async function performSearch(query) {
  showLoading();

  const useCache = toggleCache.checked;
  const useReranker = toggleReranker.checked;

  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        top_k: 12,
        use_cache: useCache,
        use_reranker: useReranker,
      }),
    });

    if (!res.ok) throw new Error("Search request failed");
    const data = await res.json();
    renderSearchResults(data);
    updateTelemetry(data);
  } catch (err) {
    console.error("Search failed:", err);
    booksGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <h3>⚠️ Search Error</h3>
        <p>Could not connect to the search engine backend at ${API_BASE}.</p>
      </div>
    `;
  }
}

function showLoading() {
  booksGrid.innerHTML = `
    <div class="loading-state" style="grid-column: 1 / -1;">
      <div class="spinner"></div>
      <h3>Retrieving candidates & neural reranking...</h3>
      <p style="font-size: 0.9rem; color: var(--text-dim);">Querying Dense BGE Embeddings, BM25 Index, and FP16 Cross-Encoder</p>
    </div>
  `;
}

function renderSearchResults(data) {
  const results = data.results || [];
  resultsCount.innerHTML = `Found <b>${results.length}</b> top matches for "<b>${escapeHtml(data.query)}</b>" (from ${data.total_candidates_retrieved || results.length} candidates)`;

  if (!results.length) {
    booksGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <h3>No books found</h3>
        <p>Try searching for a broader reading premise, different author, or genre.</p>
      </div>
    `;
    return;
  }

  booksGrid.innerHTML = "";
  results.forEach((book, idx) => {
    const card = document.createElement("div");
    card.className = "book-card";

    const authors = book.authors?.length ? book.authors.join(", ") : "Unknown Author";
    const year = book.publication_year ? `• ${book.publication_year}` : "";
    const rating = book.average_rating ? `★ ${book.average_rating.toFixed(2)}` : "★ --";
    const shelves = (book.popular_shelves || []).slice(0, 4);

    // Cover Image with fallback
    const coverUrl = book.image_url || `https://images.gr-assets.com/books/1304100136m/${book.book_id}.jpg`;

    // Score Pills
    const scores = [];
    if (book.rerank_score !== undefined && book.rerank_score !== null) {
      scores.push(`<span class="score-badge score-rerank" title="Cross-Encoder Rerank Score">🎯 Rerank: ${book.rerank_score >= 0 ? "+" : ""}${book.rerank_score.toFixed(3)}</span>`);
    }
    if (book.dense_score !== undefined && book.dense_score !== null) {
      scores.push(`<span class="score-badge score-dense" title="Dense Vector Similarity">⚡ Dense: ${book.dense_score.toFixed(3)}</span>`);
    }
    if (book.bm25_score !== undefined && book.bm25_score !== null) {
      scores.push(`<span class="score-badge score-bm25" title="Elasticsearch BM25 Keyword Score">🔍 BM25: ${book.bm25_score.toFixed(1)}</span>`);
    }

    card.innerHTML = `
      <div class="card-top">
        <div class="cover-wrapper">
          <img class="book-cover" src="${coverUrl}" alt="${escapeHtml(book.title)}" onerror="this.src='https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042a9c91a29c1d680899eff700a03.png';" />
        </div>
        <div class="card-info">
          <span class="book-rank-badge">#${idx + 1}</span>
          <h3 class="book-title" title="${escapeHtml(book.title)}">${escapeHtml(book.title || "Untitled")}</h3>
          <div class="book-author">by ${escapeHtml(authors)}</div>
          <div class="book-meta-row">
            <span class="rating-star">${rating}</span>
            <span>${year}</span>
          </div>
        </div>
      </div>

      <div class="shelves-container">
        ${shelves.map((s) => `<span class="shelf-tag">${escapeHtml(s)}</span>`).join("")}
      </div>

      <div class="score-pill-row">
        ${scores.join("")}
      </div>

      <p class="book-desc">${escapeHtml(book.description || "No description provided.")}</p>

      <div class="card-actions">
        <button class="recommend-btn" data-bid="${book.book_id}" data-btitle="${escapeHtml(book.title)}">
          <span>✨ Similar Books</span>
        </button>
      </div>
    `;

    // Attach Recommendation Click
    const recBtn = card.querySelector(".recommend-btn");
    recBtn.addEventListener("click", () => {
      openRecommendations(book.book_id, book.title);
    });

    booksGrid.appendChild(card);
  });
}

// Update Bottom Telemetry Bar
function updateTelemetry(data) {
  const lat = data.latency || {};
  const isCache = !!data.from_cache;

  telemetryStatus.textContent = isCache ? "⚡ CACHE HIT (0ms)" : "🔄 PIPELINE COMPUTE";
  telemetryStatus.className = `latency-badge ${isCache ? "cache-hit" : "compute-run"}`;

  telTotal.textContent = (lat.total_ms || 0).toFixed(1);
  telCache.textContent = (lat.cache_lookup_ms || 0).toFixed(1);
  telDense.textContent = (lat.dense_search_ms || 0).toFixed(1);
  telBM25.textContent = (lat.bm25_search_ms || 0).toFixed(1);
  telRerank.textContent = (lat.reranking_ms || 0).toFixed(1);
  telCandidates.textContent = data.total_candidates_retrieved || data.results?.length || "--";
}

// Open Vector Similarity Recommendations
async function openRecommendations(bookId, bookTitle) {
  recModalTitle.textContent = `Books Similar to "${bookTitle}"`;
  recModalGrid.innerHTML = `
    <div style="grid-column: 1 / -1; text-align: center; padding: 2rem;">
      <div class="spinner"></div>
      <p>Finding nearest neighbors in dense embedding vector space...</p>
    </div>
  `;
  recommendationModal.style.display = "flex";

  try {
    const res = await fetch(`${API_BASE}/api/recommend/${bookId}?top_k=6`);
    if (!res.ok) throw new Error("Could not retrieve recommendations");
    const data = await res.json();
    renderRecommendations(data.recommendations || []);
  } catch (err) {
    recModalGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted);">
        <p>⚠️ Unable to load recommendations for this book.</p>
      </div>
    `;
  }
}

function renderRecommendations(recs) {
  if (!recs.length) {
    recModalGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center;">No vector recommendations found.</p>`;
    return;
  }

  recModalGrid.innerHTML = "";
  recs.forEach((book) => {
    const card = document.createElement("div");
    card.className = "book-card";
    card.style.padding = "0.9rem";

    const authors = book.authors?.length ? book.authors.join(", ") : "Unknown Author";
    const coverUrl = book.image_url || `https://images.gr-assets.com/books/1304100136m/${book.book_id}.jpg`;

    card.innerHTML = `
      <div class="card-top" style="margin-bottom: 0.5rem;">
        <div class="cover-wrapper" style="width: 55px; height: 80px;">
          <img class="book-cover" src="${coverUrl}" onerror="this.src='https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042a9c91a29c1d680899eff700a03.png';" />
        </div>
        <div class="card-info">
          <h4 class="book-title" style="font-size: 0.95rem;">${escapeHtml(book.title)}</h4>
          <div class="book-author" style="font-size: 0.75rem;">${escapeHtml(authors)}</div>
          <div class="book-meta-row" style="font-size: 0.75rem;">★ ${book.average_rating || "--"}</div>
        </div>
      </div>
      <p class="book-desc" style="-webkit-line-clamp: 2; font-size: 0.75rem;">${escapeHtml(book.description || "")}</p>
    `;
    recModalGrid.appendChild(card);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
