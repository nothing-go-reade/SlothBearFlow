import React from "react";
import {
  Activity,
  AlertCircle,
  ArrowUp,
  BookOpenText,
  Bot,
  CheckCircle2,
  Database,
  FilePlus2,
  Gauge,
  Loader2,
  MessageSquareText,
  PauseCircle,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Square,
  TerminalSquare,
  Trash2,
  UserRound,
  Waves,
} from "lucide-react";
import "./index.css";

type Health = {
  ok: boolean;
  redis?: { ok: boolean; error?: string | null };
  session_store?: { backend: string; loaded_messages: number };
  milvus?: { enabled: boolean; reason?: string; collection?: string };
  postgres_persistence?: { enabled: boolean; ready?: boolean; reason?: string };
  llm?: { provider: string; model: string };
  embedding?: { provider: string; model: string };
  ollama_base_url?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "done" | "error";
  meta?: string;
  citations?: Array<{ source: string; excerpt: string }>;
  toolsUsed?: string[];
};

type TraceEvent = {
  id: string;
  time: string;
  tone: "info" | "ok" | "warn" | "error";
  text: string;
};

const promptStarters = [
  "总结当前 Agent 服务架构。",
  "检查 Redis、Milvus、Postgres 的职责和状态。",
  "给出一个新工具接入计划。",
];

const API_BASE = "http://127.0.0.1:8000";

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function nowLabel() {
  return new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function statusTone(ok?: boolean) {
  if (ok === true) return "ok";
  if (ok === false) return "warn";
  return "info";
}

export default function HomePage() {
  const [health, setHealth] = React.useState<Health | null>(null);
  const [healthLoading, setHealthLoading] = React.useState(false);
  const [messages, setMessages] = React.useState<ChatMessage[]>([
    {
      id: uid(),
      role: "assistant",
      content: "已连接本地 Agent 服务。可以开始一次运行。",
      status: "done",
      meta: "system",
    },
  ]);
  const [input, setInput] = React.useState("");
  const [sessionId, setSessionId] = React.useState(() => `web-${Date.now()}`);
  const [isSending, setIsSending] = React.useState(false);
  const [events, setEvents] = React.useState<TraceEvent[]>([]);
  const [sourceName, setSourceName] = React.useState("manual-note.md");
  const [knowledgeText, setKnowledgeText] = React.useState("");
  const [ingesting, setIngesting] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);

  const pushEvent = React.useCallback((tone: TraceEvent["tone"], text: string) => {
    setEvents((current) =>
      [{ id: uid(), time: nowLabel(), tone, text }, ...current].slice(0, 18),
    );
  }, []);

  const refreshHealth = React.useCallback(async () => {
    setHealthLoading(true);
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as Health;
      setHealth(data);
      pushEvent(data.ok ? "ok" : "warn", `health: ${data.ok ? "ready" : "degraded"}`);
    } catch (error) {
      setHealth(null);
      pushEvent("error", `health failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setHealthLoading(false);
    }
  }, [pushEvent]);

  React.useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 30000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function readStream(
    response: Response,
    assistantId: string,
  ) {
    const contentType = response.headers.get("content-type") || "";
    if (!response.body) {
      const text = await response.text();
      updateAssistant(assistantId, text, "done");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });

      if (contentType.includes("text/event-stream")) {
        buffer += chunk;
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";
        for (const frame of frames) {
          const line = frame
            .split("\n")
            .find((item) => item.startsWith("data:"));
          if (!line) continue;
          const payload = JSON.parse(line.replace(/^data:\s*/, ""));
          if (payload.type === "chunk") {
            fullText += payload.content || "";
            updateAssistant(assistantId, fullText, "streaming");
          }
          if (payload.type === "done") {
            fullText = payload.answer || fullText;
            updateAssistant(
              assistantId,
              fullText,
              "done",
              payload.source || "agent",
              Array.isArray(payload.citations) ? payload.citations : [],
              Array.isArray(payload.tools_used) ? payload.tools_used : [],
            );
          }
        }
      } else {
        fullText += chunk;
        updateAssistant(assistantId, fullText, "streaming");
      }
    }

    updateAssistant(assistantId, fullText, "done");
  }

  function updateAssistant(
    assistantId: string,
    content: string,
    status: ChatMessage["status"],
    meta?: string,
    citations?: ChatMessage["citations"],
    toolsUsed?: string[],
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              content,
              status,
              meta: meta ?? message.meta,
              citations: citations ?? message.citations,
              toolsUsed: toolsUsed ?? message.toolsUsed,
            }
          : message,
      ),
    );
  }

  async function sendMessage(messageText = input) {
    const trimmed = messageText.trim();
    if (!trimmed || isSending) return;

    const userMessage: ChatMessage = {
      id: uid(),
      role: "user",
      content: trimmed,
      status: "done",
      meta: sessionId,
    };
    const assistantId = uid();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
      meta: "waiting for backend",
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setIsSending(true);
    pushEvent("info", `chat request: ${trimmed.slice(0, 48)}`);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: trimmed }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = await response.json();
        updateAssistant(
          assistantId,
          data.answer || data.raw_output || "",
          "done",
          data.source || "agent",
          Array.isArray(data.citations) ? data.citations : [],
          Array.isArray(data.tools_used) ? data.tools_used : [],
        );
      } else {
        await readStream(response, assistantId);
      }
      pushEvent("ok", "chat response completed");
    } catch (error) {
      const message =
        error instanceof DOMException && error.name === "AbortError"
          ? "请求已停止。"
          : error instanceof Error
            ? error.message
            : String(error);
      updateAssistant(assistantId, message, "error", "request failed");
      pushEvent(error instanceof DOMException && error.name === "AbortError" ? "warn" : "error", message);
    } finally {
      setIsSending(false);
      abortRef.current = null;
    }
  }

  async function ingestKnowledge() {
    const text = knowledgeText.trim();
    if (!text || ingesting) return;
    setIngesting(true);
    pushEvent("info", `ingest queued: ${sourceName}`);
    try {
      const response = await fetch(`${API_BASE}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: sourceName.trim() || "upload", text }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }
      setKnowledgeText("");
      pushEvent("ok", `ingest accepted: ${data.job_id}`);
    } catch (error) {
      pushEvent("warn", error instanceof Error ? error.message : String(error));
    } finally {
      setIngesting(false);
    }
  }

  const healthCards = [
    {
      label: "LLM",
      value: health?.llm ? `${health.llm.provider} · ${health.llm.model}` : "checking",
      tone: health ? "ok" : "info",
      icon: Bot,
    },
    {
      label: "Redis",
      value: health?.redis?.ok
        ? `online · ${health.session_store?.backend || "redis"}`
        : health?.session_store?.backend
          ? `fallback · ${health.session_store.backend}`
          : "checking",
      tone: statusTone(health?.redis?.ok),
      icon: Database,
    },
    {
      label: "Milvus",
      value: health?.milvus?.enabled
        ? health.milvus.collection || "enabled"
        : health?.milvus?.reason || "checking",
      tone: health?.milvus?.enabled ? "ok" : "warn",
      icon: BookOpenText,
    },
    {
      label: "Postgres",
      value: health?.postgres_persistence?.enabled
        ? health.postgres_persistence.ready
          ? "ready"
          : health.postgres_persistence.reason || "enabled"
        : health?.postgres_persistence?.reason || "checking",
      tone: health?.postgres_persistence?.enabled ? statusTone(health.postgres_persistence.ready) : "info",
      icon: Server,
    },
  ] as const;

  const serviceState = health?.ok ? "Ready" : "Checking";
  const modelName = health?.llm ? `${health.llm.provider}/${health.llm.model}` : "model pending";
  const memoryState = health?.redis?.ok
    ? health.session_store?.backend || "redis"
    : health?.session_store?.backend || "memory";
  const ragState = health?.milvus?.enabled ? "enabled" : "disabled";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Waves size={20} />
          </div>
          <div>
            <strong>SlothBearFlow</strong>
            <span>Local Agent Workspace</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="top-status">
            <span className={`status-dot ${health?.ok ? "ok" : "warn"}`} />
            <strong>{serviceState}</strong>
          </div>
          <div className="top-meta">{modelName}</div>
          <button className="icon-button" onClick={() => void refreshHealth()} aria-label="刷新健康状态">
            <RefreshCw size={17} className={healthLoading ? "spin" : ""} />
          </button>
          <a className="docs-link" href="/api/docs" target="_blank" rel="noreferrer">
            API Docs
          </a>
        </div>
      </header>

      <main className="workspace">
        <aside className="rail status-rail">
          <section className="panel status-panel">
            <div className="panel-title">
              <Activity size={18} />
              <span>Service health</span>
            </div>
            <div className="pulse-row">
              <span className={`pulse ${health?.ok ? "ok" : "warn"}`} />
              <div>
                <strong>{health?.ok ? "Backend online" : "Backend pending"}</strong>
                <span>{health?.ollama_base_url || "health check"}</span>
              </div>
            </div>
            <div className="health-stack">
              {healthCards.map((item) => {
                const Icon = item.icon;
                return (
                  <div className={`health-card ${item.tone}`} key={item.label}>
                    <Icon size={17} />
                    <div>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel session-panel">
            <div className="panel-title">
              <Gauge size={18} />
              <span>Session state</span>
            </div>
            <label className="field-label" htmlFor="session-id">
              Session ID
            </label>
            <div className="session-input">
              <input
                id="session-id"
                value={sessionId}
                onChange={(event) => setSessionId(event.target.value)}
              />
              <button
                className="icon-button"
                onClick={() => setSessionId(`web-${Date.now()}`)}
                aria-label="新建会话"
              >
                <Sparkles size={16} />
              </button>
            </div>
            <div className="metric-grid">
              <div>
                <span>Messages</span>
                <strong>{messages.length}</strong>
              </div>
              <div>
                <span>Response</span>
                <strong>JSON + citations</strong>
              </div>
            </div>
          </section>
        </aside>

        <section className="chat-stage">
          <div className="stage-header">
            <div className="stage-copy">
              <span className="eyebrow">Playground</span>
              <h1>Agent Playground</h1>
              <div className="stage-metrics">
                <span>LLM · {modelName}</span>
                <span>Memory · {memoryState}</span>
                <span>RAG · {ragState}</span>
              </div>
            </div>
            <div className="stage-controls">
              <button
                className="control-button"
                disabled={!isSending}
                onClick={() => abortRef.current?.abort()}
              >
                <Square size={15} />
                Stop
              </button>
              <button className="control-button" onClick={() => setMessages([])}>
                <Trash2 size={15} />
                Clear
              </button>
            </div>
          </div>

          <div className="message-stream" ref={scrollRef}>
            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="avatar">
                  {message.role === "user" ? <UserRound size={17} /> : <Bot size={17} />}
                </div>
                <div className="bubble">
                  <div className="message-meta">
                    <strong>{message.role === "user" ? "You" : "SlothBearFlow"}</strong>
                    <span>{message.meta}</span>
                    {message.status === "streaming" && <Loader2 size={14} className="spin" />}
                    {message.status === "error" && <AlertCircle size={14} />}
                  </div>
                  <p>{message.content || "..."}</p>
                  {message.role === "assistant" && message.citations?.length ? (
                    <div className="citation-list">
                      <div className="citation-title">
                        <BookOpenText size={13} />
                        Retrieved context
                      </div>
                      {message.citations.slice(0, 3).map((citation, index) => (
                        <div className="citation-item" key={`${citation.source}-${index}`}>
                          <strong>{citation.source}</strong>
                          <span>{citation.excerpt}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              </article>
            ))}
          </div>

          <div className="prompt-strip">
            {promptStarters.map((starter) => (
              <button key={starter} onClick={() => setInput(starter)}>
                {starter}
              </button>
            ))}
          </div>

          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
          >
            <MessageSquareText size={19} />
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask SlothBearFlow..."
              rows={1}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            <button className="send-button" disabled={isSending || !input.trim()} type="submit">
              {isSending ? <PauseCircle size={18} /> : <ArrowUp size={18} />}
            </button>
          </form>
        </section>

        <aside className="rail ops-rail">
          <section className="panel ingest-panel">
            <div className="panel-title">
              <FilePlus2 size={18} />
              <span>Knowledge source</span>
            </div>
            <label className="field-label" htmlFor="source-name">
              Document
            </label>
            <input
              id="source-name"
              value={sourceName}
              onChange={(event) => setSourceName(event.target.value)}
            />
            <textarea
              className="knowledge-box"
              value={knowledgeText}
              onChange={(event) => setKnowledgeText(event.target.value)}
              placeholder="Paste source text..."
            />
            <button className="wide-button" onClick={() => void ingestKnowledge()} disabled={ingesting || !knowledgeText.trim()}>
              {ingesting ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
              Ingest
            </button>
          </section>

          <section className="panel trace-panel">
            <div className="panel-title">
              <TerminalSquare size={18} />
              <span>Run events</span>
            </div>
            <div className="trace-list">
              {events.length === 0 ? (
                <div className="empty-trace">Waiting for activity</div>
              ) : (
                events.map((event) => (
                  <div className={`trace-item ${event.tone}`} key={event.id}>
                    {event.tone === "ok" ? <CheckCircle2 size={14} /> : <span className="trace-dot" />}
                    <div>
                      <span>{event.time}</span>
                      <strong>{event.text}</strong>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
