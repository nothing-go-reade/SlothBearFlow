import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import {
  Activity,
  AlertCircle,
  ArrowUp,
  BookOpenText,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Database,
  FilePlus2,
  Gauge,
  History,
  Layers3,
  Loader2,
  LockKeyhole,
  MessageSquareText,
  Plus,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Square,
  Trash2,
  UserRound,
  Wrench,
  XCircle,
  Zap,
} from "lucide-react";
import "./index.css";

type CapabilityState = {
  agent?: {
    executor: "basic" | "tool_calling" | "explicit_react";
    tool_calling: boolean;
    streaming: boolean;
    stream_format: string;
    structured_output: boolean;
  };
  security?: {
    tool_guard_mode: "off" | "log" | "enforce";
    output_scrubbing: boolean;
    max_tool_calls_per_turn: number;
    approval_mode: string;
  };
  rag?: {
    enabled: boolean;
    available: boolean;
    hybrid_retrieval: boolean;
  };
  memory?: {
    session_backend: string;
    window_pairs: number;
    summary_enabled: boolean;
    postgres_restore: boolean;
  };
  learning?: {
    background_review: boolean;
    prompt_injection: boolean;
  };
};

type Health = {
  ok: boolean;
  status?: "ready" | "degraded";
  redis?: { ok: boolean; error?: string | null };
  session_store?: { backend: string; loaded_messages: number };
  milvus?: { enabled: boolean; reason?: string; collection?: string };
  postgres_persistence?: {
    enabled: boolean;
    ready?: boolean;
    reason?: string;
  };
  llm?: { provider: string; model: string };
  embedding?: { provider: string; model: string };
  capabilities?: CapabilityState;
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

type ActivityEvent = {
  id: string;
  time: string;
  tone: "info" | "ok" | "warn" | "error";
  text: string;
};

type IngestResult = {
  tone: "pending" | "ok" | "error";
  source: string;
  detail: string;
  time: string;
};

type InspectorTab = "run" | "knowledge" | "security";

const API_BASE = "http://127.0.0.1:8000";

const promptStarters = [
  "梳理当前 Agent 的执行链路",
  "检查 RAG 检索与引用状态",
  "评估工具调用的安全边界",
];

const executorLabels: Record<string, string> = {
  basic: "基础对话",
  tool_calling: "Tool Calling",
  explicit_react: "Explicit ReAct",
};

const markdownComponents: Components = {
  a: ({ children }) => (
    <span className="inline-reference">{children}</span>
  ),
  img: ({ alt }) => (
    <span className="inline-reference">{alt || "image"}</span>
  ),
};

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

function welcomeMessage(): ChatMessage {
  return {
    id: uid(),
    role: "assistant",
    content: "SlothBearFlow 已连接到本地 Agent 运行时。",
    status: "done",
    meta: "system",
  };
}

function InspectorRow({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral",
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  detail?: string;
  tone?: "ok" | "warn" | "error" | "neutral";
}) {
  return (
    <div className="inspector-row">
      <span className={"row-icon " + tone}>
        <Icon size={16} />
      </span>
      <div className="row-copy">
        <span>{label}</span>
        {detail ? <small>{detail}</small> : null}
      </div>
      <strong className={"row-value " + tone}>{value}</strong>
    </div>
  );
}

export default function HomePage() {
  const [health, setHealth] = React.useState<Health | null>(null);
  const [healthLoading, setHealthLoading] = React.useState(false);
  const [lastChecked, setLastChecked] = React.useState("");
  const [messages, setMessages] = React.useState<ChatMessage[]>([
    welcomeMessage(),
  ]);
  const [input, setInput] = React.useState("");
  const [sessionId, setSessionId] = React.useState(
    () => "web-" + Date.now(),
  );
  const [isSending, setIsSending] = React.useState(false);
  const [events, setEvents] = React.useState<ActivityEvent[]>([]);
  const [activeTab, setActiveTab] = React.useState<InspectorTab>("run");
  const [sourceName, setSourceName] = React.useState("manual-note.md");
  const [knowledgeText, setKnowledgeText] = React.useState("");
  const [ingesting, setIngesting] = React.useState(false);
  const [ingestResult, setIngestResult] =
    React.useState<IngestResult | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const inspectorRef = React.useRef<HTMLElement | null>(null);
  const ingestPollTokenRef = React.useRef(0);
  const previousHealthRef = React.useRef("");

  const pushEvent = React.useCallback(
    (tone: ActivityEvent["tone"], text: string) => {
      setEvents((current) =>
        [{ id: uid(), time: nowLabel(), tone, text }, ...current].slice(0, 24),
      );
    },
    [],
  );

  const refreshHealth = React.useCallback(async () => {
    setHealthLoading(true);
    try {
      const response = await fetch(API_BASE + "/health");
      if (!response.ok) {
        throw new Error("HTTP " + response.status);
      }
      const data = (await response.json()) as Health;
      const nextState = data.status || (data.ok ? "ready" : "degraded");
      setHealth(data);
      setLastChecked(nowLabel());
      if (previousHealthRef.current !== nextState) {
        pushEvent(
          nextState === "ready" ? "ok" : "warn",
          nextState === "ready"
            ? "运行依赖已就绪"
            : "服务已响应，部分依赖正在降级",
        );
        previousHealthRef.current = nextState;
      }
    } catch (error) {
      setHealth(null);
      setLastChecked(nowLabel());
      if (previousHealthRef.current !== "offline") {
        pushEvent(
          "error",
          "健康检查失败：" +
            (error instanceof Error ? error.message : String(error)),
        );
        previousHealthRef.current = "offline";
      }
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

  async function readStream(response: Response, assistantId: string) {
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
    let completed = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });

      if (!contentType.includes("text/event-stream")) {
        fullText += chunk;
        updateAssistant(assistantId, fullText, "streaming");
        continue;
      }

      buffer += chunk;
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const data = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.replace(/^data:\s*/, ""))
          .join("\n");
        if (!data) continue;
        try {
          const payload = JSON.parse(data);
          if (payload.type === "chunk") {
            fullText += payload.content || "";
            updateAssistant(assistantId, fullText, "streaming");
          }
          if (payload.type === "done") {
            completed = true;
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
        } catch {
          pushEvent("warn", "收到无法解析的流事件");
        }
      }
    }

    if (!completed) {
      updateAssistant(assistantId, fullText, "done");
    }
  }

  async function sendMessage(messageText = input) {
    const trimmed = messageText.trim();
    if (!trimmed || !sessionId.trim() || isSending) return;

    const assistantId = uid();
    setMessages((current) => [
      ...current,
      {
        id: uid(),
        role: "user",
        content: trimmed,
        status: "done",
        meta: sessionId,
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
        meta: "agent",
      },
    ]);
    setInput("");
    setIsSending(true);
    pushEvent("info", "已提交对话请求");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId.trim(),
          message: trimmed,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "HTTP " + response.status);
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
      pushEvent("ok", "Agent 响应完成");
    } catch (error) {
      const aborted =
        error instanceof DOMException && error.name === "AbortError";
      const message = aborted
        ? "请求已停止。"
        : error instanceof Error
          ? error.message
          : String(error);
      updateAssistant(assistantId, message, "error", "request failed");
      pushEvent(aborted ? "warn" : "error", message);
    } finally {
      setIsSending(false);
      abortRef.current = null;
    }
  }

  async function ingestKnowledge() {
    const text = knowledgeText.trim();
    const source = sourceName.trim() || "upload";
    if (!text || ingesting) return;

    setIngesting(true);
    setIngestResult(null);
    const pollToken = ++ingestPollTokenRef.current;
    pushEvent("info", "知识写入任务已提交");
    try {
      const response = await fetch(API_BASE + "/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, text }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "HTTP " + response.status);
      }
      setKnowledgeText("");
      setIngestResult({
        tone: "pending",
        source,
        detail: "排队中 · " + data.job_id,
        time: nowLabel(),
      });
      pushEvent("ok", "知识写入任务已进入队列");
      void pollIngestJob(data.job_id, source, pollToken);
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : String(error);
      setIngestResult({
        tone: "error",
        source,
        detail,
        time: nowLabel(),
      });
      pushEvent("error", "知识写入失败");
    } finally {
      setIngesting(false);
    }
  }

  async function pollIngestJob(
    jobId: string,
    source: string,
    pollToken: number,
  ) {
    const terminalStates = new Set(["completed", "failed", "skipped"]);
    for (let attempt = 0; attempt < 30; attempt += 1) {
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 1000);
      });
      if (ingestPollTokenRef.current !== pollToken) return;

      try {
        const response = await fetch(API_BASE + "/ingest/" + jobId);
        if (response.status === 503) {
          setIngestResult({
            tone: "pending",
            source,
            detail: "已接受，任务状态查询未启用 · " + jobId,
            time: nowLabel(),
          });
          return;
        }
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "HTTP " + response.status);
        }

        const status = String(data.status || "queued");
        if (status === "completed") {
          setIngestResult({
            tone: "ok",
            source,
            detail: "写入完成 · " + jobId,
            time: nowLabel(),
          });
          pushEvent("ok", "知识写入任务已完成");
          return;
        }
        if (terminalStates.has(status)) {
          setIngestResult({
            tone: "error",
            source,
            detail:
              (data.error_detail || "任务未完成") + " · " + jobId,
            time: nowLabel(),
          });
          pushEvent("error", "知识写入任务状态：" + status);
          return;
        }
        setIngestResult({
          tone: "pending",
          source,
          detail:
            (status === "processing" ? "正在向量化" : "排队中") +
            " · " +
            jobId,
          time: nowLabel(),
        });
      } catch (error) {
        if (attempt === 29) {
          setIngestResult({
            tone: "pending",
            source,
            detail: "状态查询超时 · " + jobId,
            time: nowLabel(),
          });
          pushEvent(
            "warn",
            error instanceof Error ? error.message : String(error),
          );
        }
      }
    }
  }

  function startNewSession() {
    const nextSession = "web-" + Date.now();
    setSessionId(nextSession);
    setMessages([welcomeMessage()]);
    setInput("");
    pushEvent("info", "已创建新会话");
  }

  function showInspector(nextTab: InspectorTab) {
    setActiveTab(nextTab);
    if (window.matchMedia("(max-width: 920px)").matches) {
      window.requestAnimationFrame(() => {
        inspectorRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    }
  }

  const capabilities = health?.capabilities;
  const backendState = health
    ? health.status || (health.ok ? "ready" : "degraded")
    : healthLoading
      ? "checking"
      : "offline";
  const stateTone =
    backendState === "ready"
      ? "ok"
      : backendState === "offline"
        ? "error"
        : "warn";
  const stateLabel =
    backendState === "ready"
      ? "Ready"
      : backendState === "degraded"
        ? "Degraded"
        : backendState === "offline"
          ? "Offline"
          : "Checking";
  const modelName = health?.llm
    ? health.llm.provider + " / " + health.llm.model
    : "等待后端";
  const executor = capabilities?.agent?.executor || "basic";
  const executorLabel = executorLabels[executor] || executor;
  const sessionBackend =
    capabilities?.memory?.session_backend ||
    health?.session_store?.backend ||
    "checking";
  const ragConfigured =
    capabilities?.rag?.enabled ??
    Boolean(health?.milvus?.enabled);
  const ragAvailable =
    capabilities?.rag?.available ??
    Boolean(health?.milvus?.enabled);
  const streamLabel = capabilities?.agent?.streaming
    ? (capabilities.agent.stream_format || "stream").toUpperCase()
    : "JSON";
  const toolGuardMode =
    capabilities?.security?.tool_guard_mode || "unknown";
  const approvalMode = capabilities?.security?.approval_mode;
  const approvalLabel =
    approvalMode === "headless_auto_deny"
      ? "auto deny"
      : approvalMode || "unknown";

  const serviceRows = [
    {
      icon: Bot,
      label: "模型运行时",
      value: health?.llm?.provider || "checking",
      detail: health?.llm?.model || "等待健康检查",
      tone: health?.llm ? "ok" : "neutral",
    },
    {
      icon: Database,
      label: "会话记忆",
      value: sessionBackend,
      detail: health?.redis?.ok
        ? "Redis 可用"
        : health?.redis?.error || "状态未知",
      tone: health?.redis?.ok ? "ok" : health ? "warn" : "neutral",
    },
    {
      icon: Search,
      label: "RAG 检索",
      value: ragAvailable ? "available" : ragConfigured ? "degraded" : "off",
      detail: ragAvailable
        ? health?.milvus?.collection || "混合检索"
        : health?.milvus?.reason || "等待健康检查",
      tone: ragAvailable ? "ok" : ragConfigured ? "warn" : "neutral",
    },
    {
      icon: Server,
      label: "持久化",
      value: health?.postgres_persistence?.enabled
        ? health.postgres_persistence.ready
          ? "ready"
          : "degraded"
        : "off",
      detail:
        health?.postgres_persistence?.reason ||
        (health?.postgres_persistence?.ready ? "Postgres 可用" : "按配置关闭"),
      tone: health?.postgres_persistence?.ready
        ? "ok"
        : health?.postgres_persistence?.enabled
          ? "warn"
          : "neutral",
    },
  ] as const;

  return (
    <div className="console-shell">
      <aside className="side-rail">
        <div className="brand-block">
          <span className="brand-mark">
            <Layers3 size={19} />
          </span>
          <div className="brand-copy">
            <strong>SlothBearFlow</strong>
            <span>Agent Console</span>
          </div>
        </div>

        <nav className="rail-nav" aria-label="工作台导航">
          <button
            className={activeTab === "run" ? "active" : ""}
            onClick={() => showInspector("run")}
            title="运行状态"
          >
            <Activity size={18} />
            <span>运行状态</span>
          </button>
          <button
            className={activeTab === "knowledge" ? "active" : ""}
            onClick={() => showInspector("knowledge")}
            title="知识库"
          >
            <BookOpenText size={18} />
            <span>知识库</span>
          </button>
          <button
            className={activeTab === "security" ? "active" : ""}
            onClick={() => showInspector("security")}
            title="安全护栏"
          >
            <ShieldCheck size={18} />
            <span>安全护栏</span>
          </button>
        </nav>

        <div className="rail-spacer" />

        <section className="session-block">
          <div className="section-kicker">
            <History size={14} />
            <span>当前会话</span>
          </div>
          <label htmlFor="session-id">Session ID</label>
          <div className="session-field">
            <input
              id="session-id"
              value={sessionId}
              maxLength={128}
              onChange={(event) => setSessionId(event.target.value)}
            />
            <button
              type="button"
              onClick={startNewSession}
              aria-label="新建会话"
              title="新建会话"
            >
              <Plus size={16} />
            </button>
          </div>
          <div className="session-stats">
            <span>
              <strong>{Math.max(0, messages.length - 1)}</strong>
              消息
            </span>
            <span>
              <strong>
                {capabilities?.memory?.window_pairs ?? "-"}
              </strong>
              记忆窗口
            </span>
          </div>
        </section>

        <div className="rail-status">
          <span className={"status-dot " + stateTone} />
          <div>
            <strong>{stateLabel}</strong>
            <span>{lastChecked || "等待检查"}</span>
          </div>
        </div>
      </aside>

      <main className="conversation">
        <header className="conversation-header">
          <div className="conversation-title">
            <span className="eyebrow">Agent Workspace</span>
            <h1>实时对话</h1>
            <div className="runtime-line">
              <span>{executorLabel}</span>
              <i />
              <span>{streamLabel}</span>
              <i />
              <span>Memory · {sessionBackend}</span>
            </div>
          </div>
          <div className="header-actions">
            <a
              className="text-link"
              href={API_BASE + "/docs"}
              target="_blank"
              rel="noreferrer"
            >
              API
            </a>
            <button
              className="icon-action"
              onClick={() => void refreshHealth()}
              aria-label="刷新运行状态"
              title="刷新运行状态"
            >
              <RefreshCw
                size={17}
                className={healthLoading ? "spin" : ""}
              />
            </button>
            <button
              className="icon-action"
              onClick={() => setMessages([welcomeMessage()])}
              aria-label="清空对话"
              title="清空对话"
            >
              <Trash2 size={17} />
            </button>
          </div>
        </header>

        <div className="message-stream" ref={scrollRef} aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-conversation">
              <MessageSquareText size={22} />
              <span>当前会话暂无消息</span>
            </div>
          ) : (
            messages.map((message) => (
              <article
                className={"message " + message.role}
                key={message.id}
              >
                <div className="avatar" aria-hidden="true">
                  {message.role === "user" ? (
                    <UserRound size={17} />
                  ) : (
                    <Bot size={17} />
                  )}
                </div>
                <div className="message-content">
                  <div className="message-meta">
                    <strong>
                      {message.role === "user" ? "你" : "SlothBearFlow"}
                    </strong>
                    <span>{message.meta}</span>
                    {message.status === "streaming" ? (
                      <Loader2 size={14} className="spin" />
                    ) : null}
                    {message.status === "error" ? (
                      <AlertCircle size={14} />
                    ) : null}
                  </div>
                  <div
                    className={
                      "message-text" +
                      (message.role === "assistant" ? " markdown-body" : "")
                    }
                  >
                    {message.role === "assistant" && message.content ? (
                      <ReactMarkdown components={markdownComponents}>
                        {message.content}
                      </ReactMarkdown>
                    ) : (
                      message.content || "正在生成响应…"
                    )}
                  </div>

                  {message.toolsUsed?.length ? (
                    <div className="tool-evidence">
                      <span className="evidence-label">
                        <Wrench size={13} />
                        工具调用
                      </span>
                      <div className="tool-list">
                        {message.toolsUsed.map((tool) => (
                          <span key={tool}>{tool}</span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {message.citations?.length ? (
                    <div className="citation-block">
                      <span className="evidence-label">
                        <BookOpenText size={13} />
                        引用来源
                      </span>
                      <div className="citation-list">
                        {message.citations.slice(0, 4).map((citation, index) => (
                          <div
                            className="citation-item"
                            key={citation.source + "-" + index}
                          >
                            <strong>{citation.source}</strong>
                            <span>{citation.excerpt}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </article>
            ))
          )}
        </div>

        <div className="composer-zone">
          <div className="prompt-strip">
            {promptStarters.map((starter) => (
              <button
                type="button"
                key={starter}
                onClick={() => setInput(starter)}
              >
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
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="向 SlothBearFlow 发送消息"
              rows={2}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            {isSending ? (
              <button
                className="stop-button"
                type="button"
                onClick={() => abortRef.current?.abort()}
                aria-label="停止生成"
                title="停止生成"
              >
                <Square size={16} />
              </button>
            ) : (
              <button
                className="send-button"
                disabled={!input.trim() || !sessionId.trim()}
                type="submit"
                aria-label="发送消息"
                title="发送消息"
              >
                <ArrowUp size={18} />
              </button>
            )}
          </form>
        </div>
      </main>

      <aside className="inspector" ref={inspectorRef}>
        <header className="inspector-header">
          <div>
            <span className="eyebrow">Operations</span>
            <h2>运行控制台</h2>
          </div>
          <span className={"compact-state " + stateTone}>
            <i />
            {stateLabel}
          </span>
        </header>

        <div className="inspector-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={activeTab === "run"}
            className={activeTab === "run" ? "active" : ""}
            onClick={() => setActiveTab("run")}
          >
            运行
          </button>
          <button
            role="tab"
            aria-selected={activeTab === "knowledge"}
            className={activeTab === "knowledge" ? "active" : ""}
            onClick={() => setActiveTab("knowledge")}
          >
            知识
          </button>
          <button
            role="tab"
            aria-selected={activeTab === "security"}
            className={activeTab === "security" ? "active" : ""}
            onClick={() => setActiveTab("security")}
          >
            护栏
          </button>
        </div>

        <div className="inspector-content">
          {activeTab === "run" ? (
            <>
              <section className="state-summary">
                <div className={"summary-signal " + stateTone}>
                  <Zap size={18} />
                </div>
                <div>
                  <span>Agent runtime</span>
                  <strong>{modelName}</strong>
                </div>
              </section>

              <section className="inspector-section">
                <div className="section-heading">
                  <Gauge size={15} />
                  <h3>依赖状态</h3>
                </div>
                <div className="inspector-list">
                  {serviceRows.map((item) => (
                    <InspectorRow key={item.label} {...item} />
                  ))}
                </div>
              </section>

              <section className="inspector-section event-section">
                <div className="section-heading">
                  <Activity size={15} />
                  <h3>当前会话事件</h3>
                  <span>{events.length}</span>
                </div>
                <div className="event-list">
                  {events.length ? (
                    events.map((event) => (
                      <div
                        className={"event-item " + event.tone}
                        key={event.id}
                      >
                        <span className="event-marker">
                          {event.tone === "ok" ? (
                            <CheckCircle2 size={13} />
                          ) : event.tone === "error" ? (
                            <XCircle size={13} />
                          ) : (
                            <i />
                          )}
                        </span>
                        <div>
                          <span>{event.time}</span>
                          <strong>{event.text}</strong>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="empty-state">暂无运行事件</div>
                  )}
                </div>
              </section>
            </>
          ) : null}

          {activeTab === "knowledge" ? (
            <>
              <section className="state-summary">
                <div className={"summary-signal " + (ragAvailable ? "ok" : "warn")}>
                  <Search size={18} />
                </div>
                <div>
                  <span>RAG pipeline</span>
                  <strong>
                    {ragAvailable
                      ? "Milvus · Hybrid retrieval"
                      : ragConfigured
                        ? "Configured · unavailable"
                        : "Disabled"}
                  </strong>
                </div>
              </section>

              <section className="inspector-section ingest-section">
                <div className="section-heading">
                  <FilePlus2 size={15} />
                  <h3>写入知识库</h3>
                </div>
                <label htmlFor="source-name">来源名称</label>
                <input
                  id="source-name"
                  value={sourceName}
                  maxLength={256}
                  onChange={(event) => setSourceName(event.target.value)}
                />
                <label htmlFor="knowledge-text">文档内容</label>
                <textarea
                  id="knowledge-text"
                  value={knowledgeText}
                  onChange={(event) => setKnowledgeText(event.target.value)}
                  placeholder="粘贴待索引的文档内容"
                  rows={9}
                />
                <button
                  className="primary-action"
                  onClick={() => void ingestKnowledge()}
                  disabled={
                    ingesting || !knowledgeText.trim() || !ragAvailable
                  }
                >
                  {ingesting ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <FilePlus2 size={16} />
                  )}
                  写入知识库
                </button>
              </section>

              {ingestResult ? (
                <section className={"job-result " + ingestResult.tone}>
                  <div>
                    {ingestResult.tone === "ok" ? (
                      <CheckCircle2 size={15} />
                    ) : ingestResult.tone === "pending" ? (
                      <Loader2 size={15} className="spin" />
                    ) : (
                      <AlertCircle size={15} />
                    )}
                    <strong>{ingestResult.source}</strong>
                    <span>{ingestResult.time}</span>
                  </div>
                  <p>{ingestResult.detail}</p>
                </section>
              ) : null}
            </>
          ) : null}

          {activeTab === "security" ? (
            <>
              <section className="state-summary">
                <div
                  className={
                    "summary-signal " +
                    (toolGuardMode === "enforce"
                      ? "ok"
                      : toolGuardMode === "off"
                        ? "error"
                        : "warn")
                  }
                >
                  <LockKeyhole size={18} />
                </div>
                <div>
                  <span>Execution policy</span>
                  <strong>Tool Guard · {toolGuardMode}</strong>
                </div>
              </section>

              <section className="inspector-section">
                <div className="section-heading">
                  <ShieldCheck size={15} />
                  <h3>执行护栏</h3>
                </div>
                <div className="inspector-list">
                  <InspectorRow
                    icon={LockKeyhole}
                    label="工具白名单"
                    value={toolGuardMode}
                    detail={
                      toolGuardMode === "enforce"
                        ? "未知工具默认拒绝"
                        : "策略未处于强制模式"
                    }
                    tone={
                      toolGuardMode === "enforce"
                        ? "ok"
                        : toolGuardMode === "off"
                          ? "error"
                          : "warn"
                    }
                  />
                  <InspectorRow
                    icon={Wrench}
                    label="调用预算"
                    value={
                      (capabilities?.security?.max_tool_calls_per_turn ?? "-") +
                      " / turn"
                    }
                    detail="全局工具调用上限"
                    tone="neutral"
                  />
                  <InspectorRow
                    icon={ShieldCheck}
                    label="输出脱敏"
                    value={
                      capabilities?.security?.output_scrubbing
                        ? "active"
                        : "off"
                    }
                    detail="工具结果敏感信息清理"
                    tone={
                      capabilities?.security?.output_scrubbing ? "ok" : "warn"
                    }
                  />
                  <InspectorRow
                    icon={AlertCircle}
                    label="人工审批"
                    value={approvalLabel}
                    detail={
                      approvalMode === "headless_auto_deny"
                        ? "无人值守下拒绝需审批工具"
                        : "等待后端能力状态"
                    }
                    tone={approvalMode ? "ok" : "neutral"}
                  />
                </div>
              </section>

              <section className="inspector-section">
                <div className="section-heading">
                  <BrainCircuit size={15} />
                  <h3>记忆与复盘</h3>
                </div>
                <div className="inspector-list">
                  <InspectorRow
                    icon={BrainCircuit}
                    label="后台 Reflection"
                    value={
                      capabilities?.learning?.background_review ? "on" : "off"
                    }
                    detail="会话后复盘任务"
                    tone={
                      capabilities?.learning?.background_review
                        ? "ok"
                        : "neutral"
                    }
                  />
                  <InspectorRow
                    icon={Zap}
                    label="学习结果注入"
                    value={
                      capabilities?.learning?.prompt_injection ? "on" : "off"
                    }
                    detail="受预算约束的提示词注入"
                    tone={
                      capabilities?.learning?.prompt_injection
                        ? "ok"
                        : "neutral"
                    }
                  />
                  <InspectorRow
                    icon={History}
                    label="异步摘要"
                    value={
                      capabilities?.memory?.summary_enabled ? "on" : "off"
                    }
                    detail="滚动会话摘要"
                    tone={
                      capabilities?.memory?.summary_enabled ? "ok" : "neutral"
                    }
                  />
                </div>
              </section>
            </>
          ) : null}
        </div>
      </aside>
    </div>
  );
}
