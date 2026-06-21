import { API } from "./config.js";

document.addEventListener("DOMContentLoaded", () => {
  // ===== DOM cache =====
  const textBox = document.getElementById("text");
  const sendBtn = document.getElementById("send");
  const resultEl = document.getElementById("result");
  const numberEl = document.getElementById("number");
  const tabCheckText = document.getElementById("tabCheckText");
  const tabEvalText = document.getElementById("tabEvalText");
  const reviewTitle = document.getElementById("reviewTitle");
  const resultTitle = document.getElementById("resultTitle");
  const aiSummaryTitle = document.getElementById("aiSummaryTitle");
  const uploadBtn = document.getElementById("uploadBtn");
  const langToggle = document.getElementById("langToggle");
  const langToggleLabel = document.getElementById("langToggleLabel");

  // ===== i18n strings =====
  const STRINGS = {
    ko: {
      placeholder: "리뷰를 드래그해서 선택하세요.",
      loading: "처리 중...",
      selectTextFirst: "텍스트를 먼저 선택해 주세요.",
      serverError: "서버 오류가 발생했습니다.",
      noReply: "응답이 없습니다.",
      scoreError: "점수 오류",
      reviewTitle: "인식된 리뷰",
      tabCheck: "리뷰 검사하기",
      tabEval: "리뷰 평가하기",
      resultTitle: " 검사 결과 (광고일 확률)",
      aiSummary: "AI 요약",
      sendBtn: "검사하기",
      uploadBtn: "가게 추천받기",
      langToggleLabel: "언어 변경",
    },
    en: {
      placeholder: "Highlight a review to analyze.",
      loading: "Working...",
      selectTextFirst: "Please select text first.",
      serverError: "A server error occurred.",
      noReply: "No response received.",
      scoreError: "Score unavailable",
      reviewTitle: "Detected Review",
      tabCheck: "Check Review",
      tabEval: "Evaluate Reviews",
      resultTitle: " Result (Ad likelihood)",
      aiSummary: "AI Summary",
      sendBtn: "Analyze",
      uploadBtn: "Get Store Recommendations",
      langToggleLabel: "Switch language",
    },
  };

  let currentLang = "ko";
  let PLACEHOLDER = STRINGS[currentLang].placeholder;

  const currentStrings = () => STRINGS[currentLang];

  // ===== In-memory cache (popup session) =====
  // key: text string
  // val: { gemReply: string, gemCached?: boolean, score?: number, detectCached?: boolean, ts: number }
  const memCache = new Map();

  // ===== Request de-dupe (same text -> share promise) =====
  // key: text string
  // val: { gemP: Promise<any>, detP: Promise<any> }
  const inflightByText = new Map();

  // AbortController for the *current* click (if user clicks multiple times quickly)
  let currentAbort = null;

  // ===== Small helpers =====
  const sameAsPlaceholder = (t) => !t || t === PLACEHOLDER;

  const setPlaceholderIfNeeded = () => {
    if (sameAsPlaceholder(textBox.textContent) || textBox.classList.contains("placeholder")) {
      textBox.textContent = PLACEHOLDER;
      textBox.classList.add("placeholder");
    }
  };

  const setStatus = (msg) => {
    if (!resultEl.textContent || resultEl.textContent === PLACEHOLDER) {
      resultEl.textContent = msg;
    }
  };

  const setLoadingUI = () => {
    resultEl.textContent = currentStrings().loading;
    if (numberEl) numberEl.textContent = "";
    if (typeof window.updateProgress === "function") {
      window.updateProgress(0);
    }
  };

  const showGemini = (reply) => {
    resultEl.textContent = reply || currentStrings().noReply;
  };

  const showScore = (score) => {
    const safeScore = Number.isFinite(score) ? score : 0;

    if (typeof window.updateProgress === "function") {
      window.updateProgress(safeScore);
      return;
    }
    if (numberEl) numberEl.textContent = String(safeScore);
  };

  const normalizeSelectedText = (t) => (t || "").trim();

  // Cache policy: keep only recent (avoid memory blow-up if user selects tons of text)
  const CACHE_TTL_MS = 2 * 60 * 1000; // 2 min
  const CACHE_MAX = 20;

  const cleanupCache = () => {
    const now = Date.now();
    for (const [k, v] of memCache.entries()) {
      if (!v?.ts || now - v.ts > CACHE_TTL_MS) memCache.delete(k);
    }
    // if still too big, drop oldest
    if (memCache.size > CACHE_MAX) {
      const entries = [...memCache.entries()].sort((a, b) => a[1].ts - b[1].ts);
      const toRemove = memCache.size - CACHE_MAX;
      for (let i = 0; i < toRemove; i++) memCache.delete(entries[i][0]);
    }
  };

  const cachePut = (text, patch) => {
    const prev = memCache.get(text) || { ts: Date.now() };
    const merged = { ...prev, ...patch, ts: Date.now() };
    memCache.set(text, merged);
    cleanupCache();
    return merged;
  };

  // ===== Selection loading =====
  const loadSelectionIntoTextBox = () => {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab?.id) {
        textBox.textContent = PLACEHOLDER;
        textBox.classList.add("placeholder");
        return;
      }

      chrome.scripting.executeScript(
        {
          target: { tabId: tab.id },
          func: () => window.getSelection().toString(),
        },
        (results) => {
          const selectedText = results?.[0]?.result ?? "";
          const cleanText = normalizeSelectedText(selectedText);

          if (cleanText) {
            textBox.textContent = cleanText;
            textBox.classList.remove("placeholder");
            chrome.storage.local.set({ selectedText: cleanText });
          } else {
            chrome.storage.local.get("selectedText", ({ selectedText }) => {
              const fallback = normalizeSelectedText(selectedText) || PLACEHOLDER;
              textBox.textContent = fallback;
              textBox.classList.toggle("placeholder", fallback === PLACEHOLDER);
            });
          }
        }
      );
    });
  };

  // ===== Networking =====
  const postJSON = async (url, bodyStr, signal) => {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: bodyStr,
      signal,
    });

    if (!res.ok) {
      let detail = "";
      try {
        const t = await res.text();
        detail = t ? ` (${t.slice(0, 200)})` : "";
      } catch (_) {}
      throw new Error(`HTTP ${res.status}${detail}`);
    }
    return res.json();
  };

  const getOrStartInflight = (text, signal) => {
    const existing = inflightByText.get(text);
    if (existing) return existing;

    const payload = JSON.stringify({ text });

    const gemP = postJSON(API.GEMINI, payload, signal);
    const detP = postJSON(API.DETECT_AD, payload, signal);

    const pair = { gemP, detP };
    inflightByText.set(text, pair);

    Promise.allSettled([gemP, detP]).then(() => {
      const cur = inflightByText.get(text);
      if (cur === pair) inflightByText.delete(text);
    });

    return pair;
  };

  // ===== Language toggle =====
  const applyLanguage = (lang) => {
    if (!STRINGS[lang]) lang = "ko";
    currentLang = lang;
    PLACEHOLDER = STRINGS[currentLang].placeholder;
    const t = currentStrings();

    document.documentElement.lang = currentLang;
    if (tabCheckText) tabCheckText.innerHTML = `📝<b> ${t.tabCheck}</b>`;
    if (tabEvalText) tabEvalText.innerHTML = `📊<b> ${t.tabEval}</b>`;
    if (reviewTitle) reviewTitle.textContent = t.reviewTitle;
    if (resultTitle) resultTitle.textContent = t.resultTitle;
    if (aiSummaryTitle) aiSummaryTitle.textContent = t.aiSummary;
    if (sendBtn) sendBtn.textContent = t.sendBtn;
    if (uploadBtn) uploadBtn.textContent = t.uploadBtn;
    if (langToggleLabel) langToggleLabel.textContent = currentLang === "ko" ? "EN" : "KO";
    if (langToggle) langToggle.setAttribute("aria-label", t.langToggleLabel);

    setPlaceholderIfNeeded();
    chrome.storage.local.set({ uiLang: currentLang });
  };

  // ===== Click handler =====
  const onSend = async () => {
    const inputText = normalizeSelectedText(textBox.textContent);

    if (sameAsPlaceholder(inputText)) {
      resultEl.innerText = currentStrings().selectTextFirst;
      return;
    }

    if (currentAbort) currentAbort.abort();
    currentAbort = new AbortController();

    setLoadingUI();

    const cached = memCache.get(inputText);
    if (cached && Date.now() - cached.ts <= CACHE_TTL_MS) {
      if (cached.gemReply) showGemini(cached.gemReply);
      if (typeof cached.score === "number") showScore(cached.score);
      return;
    }

    try {
      const { gemP, detP } = getOrStartInflight(inputText, currentAbort.signal);

      gemP
        .then((gem) => {
          const reply = gem?.reply || currentStrings().noReply;
          cachePut(inputText, { gemReply: reply, gemCached: !!gem?.cached });
          showGemini(reply);
          if (gem?.cached) console.log("Gemini: cache hit");
        })
        .catch((e) => {
          if (e?.name === "AbortError") return;
          console.error("Gemini Error:", e);
          resultEl.textContent = currentStrings().noReply;
        });

      detP
        .then((rate) => {
          const score = Number(rate?.prob_ad ?? 0);
          cachePut(inputText, { score, detectCached: !!rate?.cached });
          showScore(score);
          if (rate?.cached) console.log("Detect-ad: cache hit");
        })
        .catch((e) => {
          if (e?.name === "AbortError") return;
          console.error("Detect-ad Error:", e);
          if (numberEl) numberEl.textContent = currentStrings().scoreError;
        });
    } catch (err) {
      if (err?.name === "AbortError") return;
      console.error("API Error:", err);
      resultEl.textContent = currentStrings().serverError;
    }
  };

  sendBtn.addEventListener("click", onSend);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onSend();
  });

  if (langToggle) {
    langToggle.addEventListener("click", () => {
      const next = currentLang === "ko" ? "en" : "ko";
      applyLanguage(next);
    });
  }

  const initializeLanguage = () => {
    chrome.storage.local.get("uiLang", ({ uiLang }) => {
      if (uiLang && STRINGS[uiLang]) currentLang = uiLang;
      PLACEHOLDER = STRINGS[currentLang].placeholder;
      applyLanguage(currentLang);
      loadSelectionIntoTextBox();
    });
  };

  window.addEventListener("beforeunload", () => {
    if (currentAbort) currentAbort.abort();
  });

  initializeLanguage();
});
