export const ENV = "local"; // "local" | "server"

const URLS = {
  local: "http://localhost:8000",
  server: "http://34.174.35.119:8000",
};

export const BASE_URL = URLS[ENV];

// (선택) API endpoint 중앙관리 — 강력 추천
export const API = {
  GEMINI: `${BASE_URL}/gemini`,
  DETECT_AD: `${BASE_URL}/detect-ad`,
  FEEDBACK: `${BASE_URL}/feedback`,
};
