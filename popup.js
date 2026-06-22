import { API } from "./config.js";

document.addEventListener("DOMContentLoaded", () => {
  // ===== DOM cache =====
  const textBox = document.getElementById("text");
  const sendBtn = document.getElementById("send");
  const numberEl = document.getElementById("number");
  const messageEl = document.getElementById("message");
  const progressBox = document.getElementById("progressBox");
  const resultLoader = document.getElementById("resultLoader");
  const tabCheckText = document.getElementById("tabCheckText");
  const tabEvalText = document.getElementById("tabEvalText");
  const reviewTitle = document.getElementById("reviewTitle");
  const resultTitle = document.getElementById("resultTitle");
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
      scoreError: "점수 오류",
      scoreVeryHigh: "직접 작성하지 않은 광고 리뷰예요.",
      scoreHigh: "광고로 의심되는 문구가 많아요.",
      scoreMid: "부분적으로 광고성 문구가 있어요.",
      scoreLow: "광고 가능성이 낮아요.",
      scoreVeryLow: "직접 작성한 리뷰에 가까워요.",
      reviewTitle: "인식된 리뷰",
      tabCheck: "리뷰 검사하기",
      tabEval: "리뷰 평가하기",
      resultTitle: " 검사 결과 (광고일 확률)",
      sendBtn: "검사하기",
      uploadBtn: "가게 추천받기",
      langToggleLabel: "언어 변경",
    },
    en: {
      placeholder: "Highlight a review to analyze.",
      loading: "Working...",
      selectTextFirst: "Please select text first.",
      serverError: "A server error occurred.",
      scoreError: "Score unavailable",
      scoreVeryHigh: "Likely promotional or generated.",
      scoreHigh: "Contains many promotional signals.",
      scoreMid: "Contains some promotional language.",
      scoreLow: "Low ad likelihood.",
      scoreVeryLow: "Likely an organic review.",
      reviewTitle: "Detected Review",
      tabCheck: "Check Review",
      tabEval: "Evaluate Reviews",
      resultTitle: " Result (Ad likelihood)",
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
  // val: { score?: number, detectCached?: boolean, ts: number }
  const memCache = new Map();

  // ===== Request de-dupe (same text -> share promise) =====
  // key: text string
  // val: { detP: Promise<any> }
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

  const setLoadingUI = (isLoading) => {
    if (progressBox) progressBox.classList.toggle("is-loading", isLoading);
    if (resultLoader) resultLoader.hidden = !isLoading;
    if (sendBtn) sendBtn.disabled = isLoading;
    if (numberEl && isLoading) numberEl.textContent = "";
    if (messageEl && isLoading) messageEl.textContent = currentStrings().loading;
    if (isLoading && typeof window.updateProgress === "function") {
      window.updateProgress(0);
    }
  };

  const clearLoadingUI = () => {
    if (progressBox) progressBox.classList.remove("is-loading");
    if (resultLoader) resultLoader.hidden = true;
    if (sendBtn) sendBtn.disabled = false;
  };

  const normalizePredictionScore = (score, isRatio = false) => {
    const numericScore = Number(score);
    if (!Number.isFinite(numericScore)) return null;
    const percentScore = isRatio ? numericScore * 100 : numericScore;
    return Math.min(Math.max(percentScore, 0), 100);
  };

  const showScore = (score) => {
    const clampedScore = normalizePredictionScore(score);
    if (clampedScore === null) {
      if (numberEl) numberEl.textContent = currentStrings().scoreError;
      if (messageEl) messageEl.textContent = currentStrings().scoreError;
      return;
    }

    if (messageEl) {
      const t = currentStrings();
      if (clampedScore >= 80) messageEl.textContent = t.scoreVeryHigh;
      else if (clampedScore >= 60) messageEl.textContent = t.scoreHigh;
      else if (clampedScore >= 40) messageEl.textContent = t.scoreMid;
      else if (clampedScore >= 20) messageEl.textContent = t.scoreLow;
      else messageEl.textContent = t.scoreVeryLow;
    }

    if (typeof window.updateProgress === "function") {
      window.updateProgress(clampedScore);
      return;
    }
    if (numberEl) numberEl.textContent = `${clampedScore.toFixed(0)}%`;
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

    const detP = postJSON(API.DETECT_AD, payload, signal);

    const pair = { detP };
    inflightByText.set(text, pair);

    Promise.allSettled([detP]).then(() => {
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
      clearLoadingUI();
      if (numberEl) numberEl.textContent = "";
      if (messageEl) messageEl.textContent = currentStrings().selectTextFirst;
      return;
    }

    if (currentAbort) currentAbort.abort();
    currentAbort = new AbortController();

    setLoadingUI(true);

    const cached = memCache.get(inputText);
    if (cached && Date.now() - cached.ts <= CACHE_TTL_MS) {
      if (typeof cached.score === "number") showScore(cached.score);
      clearLoadingUI();
      return;
    }

    try {
      const { detP } = getOrStartInflight(inputText, currentAbort.signal);
      const rate = await detP;

      if (!rate || !Object.prototype.hasOwnProperty.call(rate, "prob_ad")) {
        throw new Error("Missing prob_ad in detect-ad response");
      }
      const directPredictShape = !Object.prototype.hasOwnProperty.call(rate, "is_ad");
      const score = normalizePredictionScore(rate.prob_ad, directPredictShape);
      if (score === null) {
        throw new Error(`Invalid prob_ad in detect-ad response: ${rate.prob_ad}`);
      }
      cachePut(inputText, { score, detectCached: !!rate?.cached });
      showScore(score);
      clearLoadingUI();
      if (rate?.cached) console.log("Detect-ad: cache hit");
    } catch (err) {
      if (err?.name === "AbortError") return;
      console.error("API Error:", err);
      clearLoadingUI();
      if (numberEl) numberEl.textContent = currentStrings().scoreError;
      if (messageEl) messageEl.textContent = currentStrings().serverError;
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
