"use client";

import { Fragment, useEffect, useRef, useState } from "react";

interface Source {
  source: string;
  page: number | string;
  preview: string;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  error?: boolean;
}

// Trust boundary: validate the backend payload before rendering it.
// ponytail: shallow shape check (container + array-ness); trust our own
// backend for element fields. Deepen if this ever consumes untrusted APIs.
// Backend also returns `citations` — rendered inline as chips, so not stored here.
function parseQueryResponse(raw: unknown): QueryResponse {
  if (
    typeof raw !== "object" ||
    raw === null ||
    typeof (raw as Record<string, unknown>).answer !== "string" ||
    !Array.isArray((raw as Record<string, unknown>).sources)
  ) {
    throw new Error("Malformed response from backend.");
  }
  return raw as QueryResponse;
}

const CORPUS = [
  { tag: "OWASP TOP 10", key: "owasp" },
  { tag: "MITRE ATT&CK", key: "mitre" },
  { tag: "TH-WSS · 2025", key: "thwss" },
];

const SAMPLES = [
  { meta: "OWASP", q: "What is Broken Access Control according to OWASP?" },
  { meta: "MITRE ATT&CK", q: "What is the difference between a Tactic and a Technique in MITRE ATT&CK?" },
  { meta: "ภาษาไทย · TH-WSS", q: "มาตรฐานความปลอดภัยเว็บไซต์ของไทยมีอะไรบ้าง" },
];

// owasp-top-10.pdf -> OWASP, etc. Short code for the citation chips.
function sourceCode(raw: string): string {
  const s = raw.toLowerCase();
  if (s.includes("owasp")) return "OWASP";
  if (s.includes("mitre")) return "MITRE";
  if (s.includes("thailand") || s.includes("th-wss")) return "TH-WSS";
  return raw.replace(/\.pdf$/i, "").replace(/^dataset\//, "");
}

// Split answer text on EN + TH citation markers, rendering each as an amber chip.
const CITE_RE =
  /\[Source:\s*([^,\]]+),\s*Page\s*([\d, ]+|unknown)\]|\[แหล่งที่มา:\s*([^,\]]+),\s*หน้า\s*([\d, ]+|unknown)\]/g;

function renderAnswer(text: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  CITE_RE.lastIndex = 0;
  while ((m = CITE_RE.exec(text)) !== null) {
    if (m.index > last) out.push(<Fragment key={`t${i}`}>{text.slice(last, m.index)}</Fragment>);
    const src = (m[1] ?? m[3] ?? "").trim();
    const pg = (m[2] ?? m[4] ?? "").trim();
    out.push(
      <span className="cite" key={`c${i}`} title={`${src} · page ${pg}`}>
        {sourceCode(src)}
        <span className="cite__pg">·p{pg}</span>
      </span>
    );
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) out.push(<Fragment key={`t${i}`}>{text.slice(last)}</Fragment>);
  return out;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [k, setK] = useState(8);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text?: string) {
    const question = (text ?? input).trim();
    if (!question || loading) return;

    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    // Cap the wait so a slow/hung backend fails with a clear message instead of
    // a raw socket hang-up. 60s covers a cold model call; adjust if k grows large.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 60_000);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
      }
      const data = parseQueryResponse(await res.json());
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (e) {
      const msg =
        e instanceof DOMException && e.name === "AbortError"
          ? "Request timed out after 60s — the backend is slow or unreachable on :8000."
          : e instanceof Error
            ? `Request failed — ${e.message}. Is the backend running on :8000?`
            : "Request failed.";
      setMessages((m) => [
        ...m,
        { role: "assistant", content: msg, error: true },
      ]);
    } finally {
      clearTimeout(timer);
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__mark">
          <span className="masthead__eyebrow">Evidence Console</span>
          <h1 className="masthead__title">Thai Cybersecurity RAG</h1>
        </div>
        <nav className="corpus" aria-label="Source corpus">
          {CORPUS.map((c) => (
            <span className="corpus__tag" key={c.key}>{c.tag}</span>
          ))}
        </nav>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <section className="intro">
            <p className="intro__lead">
              Ask in English or Thai. Every claim is traced back to a source PDF and page —
              nothing is asserted without a citation.
            </p>
            <div className="samples">
              <span className="samples__label">Start with</span>
              {SAMPLES.map((s) => (
                <button className="sample" key={s.q} onClick={() => send(s.q)}>
                  <span className="sample__meta">{s.meta}</span>
                  {s.q}
                </button>
              ))}
            </div>
          </section>
        )}

        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div className="turn--user" key={i}>
              <div className="turn__q">{msg.content}</div>
            </div>
          ) : (
            <article
              className={"report" + (msg.error ? " report--error" : "")}
              key={i}
              role={msg.error ? "alert" : undefined}
            >
              <div className="report__head">{msg.error ? "Error" : "Answer"}</div>
              <div className="report__body">
                {msg.error ? msg.content : renderAnswer(msg.content)}
              </div>

              {msg.sources && msg.sources.length > 0 && (
                <section className="evidence">
                  <div className="evidence__label">Retrieved evidence ({msg.sources.length})</div>
                  <div className="evidence__list">
                    {msg.sources.map((s, j) => (
                      <div className="card" key={j}>
                        <span className="card__idx">[{String(j + 1).padStart(2, "0")}]</span>
                        <div>
                          <div className="card__ref">
                            {sourceCode(s.source)} <span className="card__pg">· p{s.page}</span>
                          </div>
                          <div className="card__preview">{s.preview}…</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </article>
          )
        )}

        {loading && (
          <div className="thinking" role="status">
            <span className="thinking__dot" />
            Retrieving &amp; grounding
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <div className="composer__depth">
          <label htmlFor="k">Retrieval depth · k=<b>{k}</b></label>
          <input
            id="k"
            className="composer__slider"
            type="range"
            min={3}
            max={15}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
          />
        </div>
        <div className="composer__row">
          <textarea
            rows={2}
            className="composer__input"
            aria-label="Ask a security question"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask a security question…  (Enter to send · Shift+Enter for newline)"
          />
          <button
            className="composer__send"
            onClick={() => send()}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </footer>
    </div>
  );
}
