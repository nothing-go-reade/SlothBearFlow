import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import {
  Activity,
  AlertCircle,
  ArrowUp,
  BookOpenText,
  BrainCircuit,
  Check,
  CheckCircle2,
  Clock3,
  Database,
  FilePlus2,
  Gauge,
  History,
  Loader2,
  LogIn,
  LogOut,
  MessageSquareText,
  Network,
  Plus,
  RefreshCw,
  Route,
  Search,
  Server,
  ShieldAlert,
  ShieldCheck,
  Square,
  TimerReset,
  Trash2,
  UserRound,
  Wrench,
  XCircle,
  Zap,
} from "lucide-react";
import "./index.css";

const API_BASE = process.env.UMI_APP_API_BASE || "/api";
type Tab = "run" | "memory" | "knowledge" | "security" | "traces";
type Tone = "info" | "ok" | "warn" | "error";
type PanelPhase = "idle" | "loading" | "ready" | "forbidden" | "error";
type PanelState = { phase: PanelPhase; message?: string };
type User = {
  username: string;
  roles?: string[];
  scopes?: string[];
  tenant_id?: string;
};
type ToolTrace = {
  call_id?: string;
  name?: string;
  ok?: boolean;
  status?: string;
  duration_ms?: number;
  policy_decision?: string;
  observation?: string;
};
type TurnMeta = {
  trace_id?: string;
  turn_id?: string;
  stop_reason?: string;
  steps?: number;
  latency_ms?: number;
  executor?: string;
  prompt_version?: string;
  tool_trace?: ToolTrace[];
};
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "done" | "error";
  meta?: string;
  citations?: Array<{
    source: string;
    excerpt: string;
    supported?: boolean;
    support_score?: number;
  }>;
  toolsUsed?: string[];
  turn?: TurnMeta;
};
type Health = {
  ok: boolean;
  status?: "ready" | "degraded";
  redis?: { ok: boolean; error?: string };
  session_store?: { backend: string };
  milvus?: { enabled: boolean; reason?: string; collection?: string };
  llm?: { provider: string; model: string };
  capabilities?: {
    agent?: {
      executor?: string;
      streaming?: boolean;
      stream_format?: string;
      prompt_version?: string;
    };
    security?: {
      tool_guard_mode?: string;
      approval_mode?: string;
      auth_required?: boolean;
      audit_enabled?: boolean;
    };
    memory?: { session_backend?: string; window_pairs?: number };
    mcp?: { enabled?: boolean; servers?: unknown[] };
    observability?: {
      enabled?: boolean;
      local_trace_store?: boolean;
      prometheus?: boolean;
      langfuse?: boolean;
      langfuse_api?: string;
      langfuse_configured?: boolean;
    };
  };
};
type Event = { id: string; time: string; tone: Tone; text: string };
type IngestJob = {
  job_id: string;
  source: string;
  status: string;
  error_detail?: string;
};
type SessionSummary = {
  session_id: string;
  created_at: string;
  updated_at: string;
  last_user_message: string;
  last_assistant_message: string;
  turn_count: number;
};

const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const initialPanelState: Record<Tab, PanelState> = {
  run: { phase: "loading" },
  memory: { phase: "idle" },
  knowledge: { phase: "idle" },
  security: { phase: "idle" },
  traces: { phase: "idle" },
};

const starters = [
  "梳理当前 Agent 的执行链路",
  "检查 RAG 检索与引用状态",
  "评估工具调用的安全边界",
];
const markdownComponents: Components = {
  a: ({ children }) => <span className="inline-reference">{children}</span>,
  img: ({ alt }) => <span className="inline-reference">{alt || "image"}</span>,
};
const uid = () => Math.random().toString(36).slice(2, 10);
const time = () =>
  new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
const welcome = (): Message => ({
  id: uid(),
  role: "assistant",
  content:
    "SlothBearFlow 已连接。会话记忆、知识检索、安全审批与执行追踪均可在当前工作台查看。",
  status: "done",
  meta: "system",
});

const sleep = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function normalizeSessionId(value: string): string {
  return value.trim();
}

function validateSessionId(value: string): string {
  if (!value) return "Session ID 不能为空";
  if (!SESSION_ID_PATTERN.test(value)) {
    return "仅支持字母、数字、点、下划线、冒号和连字符，且须以字母或数字开头";
  }
  return "";
}

async function responseError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const payload = JSON.parse(text) as { detail?: string; message?: string };
    return (
      payload.detail || payload.message || text || `HTTP ${response.status}`
    );
  } catch {
    return text || `HTTP ${response.status}`;
  }
}

function formatTimestamp(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value || "");
  const milliseconds = numeric < 1_000_000_000_000 ? numeric * 1000 : numeric;
  return new Date(milliseconds).toLocaleString("zh-CN", { hour12: false });
}

function formatSessionOption(item: SessionSummary): string {
  const date = new Date(item.updated_at);
  const updated = Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      });
  const preview = item.last_user_message.trim() || item.session_id;
  const compactPreview = preview.length > 22 ? preview.slice(0, 22) + "…" : preview;
  return [updated, compactPreview].filter(Boolean).join(" · ");
}

function traceRequestLabel(item: Record<string, unknown>): string {
  const metadata = (item.metadata || {}) as Record<string, unknown>;
  const method = String(metadata.method || item.method || "").toUpperCase();
  const path = String(metadata.path || item.path || "");
  return (
    [method, path].filter(Boolean).join(" ") ||
    String(item.operation || item.name || "agent turn")
  );
}

function trapDialogFocus(event: React.KeyboardEvent<HTMLFormElement>) {
  if (event.key !== "Tab") return;
  const controls = Array.from(
    event.currentTarget.querySelectorAll<HTMLElement>(
      "input:not([disabled]), button:not([disabled])",
    ),
  ).filter((element) => element.offsetParent !== null);
  if (!controls.length) return;
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
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
  const [user, setUser] = React.useState<User | null>(null);
  const [loginOpen, setLoginOpen] = React.useState(false);
  const [credentials, setCredentials] = React.useState({
    username: "",
    password: "",
  });
  const [loginError, setLoginError] = React.useState("");
  const [loadingLogin, setLoadingLogin] = React.useState(false);
  const [signingOut, setSigningOut] = React.useState(false);
  const [messages, setMessages] = React.useState<Message[]>([welcome()]);
  const [input, setInput] = React.useState("");
  const [sessionId, setSessionId] = React.useState(() => "web-" + Date.now());
  const [sessionDraft, setSessionDraft] = React.useState(sessionId);
  const [sessionError, setSessionError] = React.useState("");
  const [sessions, setSessions] = React.useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = React.useState(false);
  const [isSending, setIsSending] = React.useState(false);
  const [tab, setTab] = React.useState<Tab>("run");
  const [events, setEvents] = React.useState<Event[]>([]);
  const [panelData, setPanelData] = React.useState<Record<string, unknown[]>>(
    {},
  );
  const [panels, setPanels] =
    React.useState<Record<Tab, PanelState>>(initialPanelState);
  const [approvalBusy, setApprovalBusy] = React.useState<string[]>([]);
  const [sourceName, setSourceName] = React.useState("manual-note.md");
  const [knowledgeText, setKnowledgeText] = React.useState("");
  const [ingesting, setIngesting] = React.useState(false);
  const [ingestJob, setIngestJob] = React.useState<IngestJob | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const streamRef = React.useRef<HTMLDivElement | null>(null);
  const inspectorRef = React.useRef<HTMLElement | null>(null);
  const shellRef = React.useRef<HTMLDivElement | null>(null);
  const previousFocusRef = React.useRef<HTMLElement | null>(null);
  const dialogWasOpenRef = React.useRef(false);
  const signingOutRef = React.useRef(false);
  const authEpochRef = React.useRef(0);
  const panelRequestRef = React.useRef<Record<Tab, number>>({
    run: 0,
    memory: 0,
    knowledge: 0,
    security: 0,
    traces: 0,
  });
  const tabRefs = React.useRef<Partial<Record<Tab, HTMLButtonElement | null>>>(
    {},
  );

  const caps = health?.capabilities;
  const authRequired = caps?.security?.auth_required ?? true;
  const accessGranted = Boolean(health) && (!authRequired || Boolean(user));
  const canDeleteMemory = Boolean(user?.scopes?.includes("memory:delete"));
  const canWriteKnowledge = Boolean(user?.scopes?.includes("knowledge:write"));
  const canApproveTools = Boolean(user?.scopes?.includes("security:approve"));
  const dialogOpen = loginOpen && authRequired;
  const conversationMessageCount = messages.filter(
    (message) => message.meta !== "system",
  ).length;

  const push = React.useCallback(
    (tone: Tone, text: string) =>
      setEvents((items) =>
        [{ id: uid(), time: time(), tone, text }, ...items].slice(0, 30),
      ),
    [],
  );
  const openLogin = React.useCallback(() => {
    if (!loginOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement | null;
    }
    setLoginOpen(true);
  }, [loginOpen]);
  const resetUserWorkspace = React.useCallback(() => {
    const nextSession = "web-" + Date.now();
    setMessages([welcome()]);
    setInput("");
    setSessionId(nextSession);
    setSessionDraft(nextSession);
    setSessionError("");
    setSessions([]);
    setSessionsLoading(false);
    setIsSending(false);
    setEvents([]);
    setPanelData({});
    setPanels(initialPanelState);
    setApprovalBusy([]);
    setSourceName("manual-note.md");
    setKnowledgeText("");
    setIngestJob(null);
    setIngesting(false);
  }, []);
  const signOut = React.useCallback(async () => {
    if (signingOutRef.current) return false;
    signingOutRef.current = true;
    authEpochRef.current += 1;
    abortRef.current?.abort();
    for (const panel of Object.keys(panelRequestRef.current) as Tab[]) {
      panelRequestRef.current[panel] += 1;
    }
    setSigningOut(true);
    try {
      const response = await fetch(API_BASE + "/auth/logout", {
        method: "POST",
        credentials: "include",
      });
      if (!response.ok) throw new Error(await responseError(response));
      setUser(null);
      resetUserWorkspace();
      if (authRequired) openLogin();
      else setLoginOpen(false);
      push("warn", authRequired ? "登录状态已清除" : "已切换为本地模式");
    } catch (error) {
      setUser(null);
      resetUserWorkspace();
      if (authRequired) openLogin();
      push(
        "error",
        "退出请求失败，本地工作区已清除：" +
          (error instanceof Error ? error.message : String(error)),
      );
      return false;
    } finally {
      signingOutRef.current = false;
      setSigningOut(false);
    }
    return true;
  }, [authRequired, openLogin, push, resetUserWorkspace]);
  const request = React.useCallback(
    async (path: string, init: RequestInit = {}) => {
      const headers = new Headers(init.headers);
      const response = await fetch(API_BASE + path, {
        ...init,
        headers,
        credentials: "include",
      });
      if (response.status === 401) {
        await signOut().catch(() => undefined);
        throw new Error(
          authRequired
            ? "登录已失效，请重新登录"
            : "凭据无效，已切换为本地模式",
        );
      }
      return response;
    },
    [authRequired, signOut],
  );

  const refreshSessions = React.useCallback(async (): Promise<SessionSummary[]> => {
    setSessionsLoading(true);
    try {
      const response = await request("/sessions?limit=100");
      if (!response.ok) throw new Error(await responseError(response));
      const data = (await response.json()) as { items?: SessionSummary[] };
      const next = [...(data.items || [])].sort(
        (left, right) =>
          new Date(right.updated_at).getTime() -
          new Date(left.updated_at).getTime(),
      );
      setSessions(next);
      return next;
    } finally {
      setSessionsLoading(false);
    }
  }, [request]);

  const loadSessionHistory = React.useCallback(
    async (
      nextSessionId: string,
      options: { announce?: boolean; authEpoch?: number } = {},
    ): Promise<boolean> => {
      const expectedEpoch = options.authEpoch ?? authEpochRef.current;
      const response = await request(
        "/memory/" + encodeURIComponent(nextSessionId),
      );
      if (!response.ok) throw new Error(await responseError(response));
      const data = (await response.json()) as {
        memory?: { messages?: Array<Record<string, unknown>> };
      };
      if (authEpochRef.current !== expectedEpoch) return false;
      const rows = (data.memory?.messages || []).filter(
        (row) => row.role === "user" || row.role === "assistant",
      );
      const restored: Message[] = rows.map((row) => ({
        id: uid(),
        role: row.role as Message["role"],
        content: String(row.content || ""),
        status: "done",
        meta: row.role === "user" ? nextSessionId : "history",
      }));
      abortRef.current?.abort();
      setSessionId(nextSessionId);
      setSessionDraft(nextSessionId);
      setSessionError("");
      setMessages(restored.length ? restored : [welcome()]);
      setPanelData((current) => ({ ...current, memory: rows }));
      setPanels((current) => ({
        ...current,
        memory: { phase: "ready" },
      }));
      if (options.announce) push("info", `已恢复会话：${nextSessionId}`);
      return true;
    },
    [push, request],
  );

  const restoreLatestSession = React.useCallback(
    async (authEpoch: number): Promise<void> => {
      try {
        const items = await refreshSessions();
        if (!items.length || authEpochRef.current !== authEpoch) return;
        await loadSessionHistory(items[0].session_id, { authEpoch });
      } catch (error) {
        if (authEpochRef.current !== authEpoch) return;
        push(
          "warn",
          "历史会话恢复失败：" +
            (error instanceof Error ? error.message : String(error)),
        );
      }
    },
    [loadSessionHistory, push, refreshSessions],
  );

  const refreshHealth = React.useCallback(async () => {
    const requestId = ++panelRequestRef.current.run;
    setPanels((current) => ({
      ...current,
      run: { phase: "loading" },
    }));
    try {
      const statusPath = user ? "/runtime/status" : "/health";
      const response = await fetch(API_BASE + statusPath, {
        credentials: "include",
      });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const nextHealth = (await response.json()) as Health;
      if (panelRequestRef.current.run !== requestId) return;
      setHealth(nextHealth);
      const requiresLogin =
        nextHealth.capabilities?.security?.auth_required ?? true;
      if (!requiresLogin) setLoginOpen(false);
      setPanels((current) => ({
        ...current,
        run: { phase: "ready" },
      }));
    } catch (error) {
      if (panelRequestRef.current.run !== requestId) return;
      setHealth(null);
      const detail = error instanceof Error ? error.message : String(error);
      setPanels((current) => ({
        ...current,
        run: { phase: "error", message: `健康检查失败：${detail}` },
      }));
      push("error", "健康检查失败：" + detail);
    }
  }, [push, user]);
  const loadMe = React.useCallback(async () => {
    const authEpoch = authEpochRef.current;
    try {
      const response = await request("/auth/me");
      if (!response.ok) throw new Error("HTTP " + response.status);
      const nextUser = await response.json();
      if (authEpochRef.current !== authEpoch) return;
      setUser(nextUser);
      await restoreLatestSession(authEpoch);
    } catch {
      /* request handles expired credentials */
    }
  }, [request, restoreLatestSession]);
  React.useEffect(() => {
    void refreshHealth();
    const id = window.setInterval(() => void refreshHealth(), 30000);
    return () => clearInterval(id);
  }, [refreshHealth]);
  React.useEffect(() => {
    void loadMe();
  }, [loadMe]);
  React.useEffect(() => {
    const shell = shellRef.current as
      (HTMLDivElement & { inert: boolean }) | null;
    if (!shell) return;
    shell.inert = dialogOpen;
    if (dialogOpen) {
      shell.setAttribute("inert", "");
      shell.setAttribute("aria-hidden", "true");
      dialogWasOpenRef.current = true;
      return;
    }
    shell.removeAttribute("inert");
    shell.removeAttribute("aria-hidden");
    if (dialogWasOpenRef.current) {
      dialogWasOpenRef.current = false;
      window.requestAnimationFrame(() => previousFocusRef.current?.focus());
    }
  }, [dialogOpen]);
  React.useEffect(() => {
    streamRef.current?.scrollTo({
      top: streamRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  function commitSession(rawValue = sessionDraft): string | null {
    const normalized = normalizeSessionId(rawValue);
    const error = validateSessionId(normalized);
    setSessionDraft(normalized);
    setSessionError(error);
    if (error) return null;
    if (normalized === sessionId) return normalized;

    abortRef.current?.abort();
    panelRequestRef.current.memory += 1;
    setSessionId(normalized);
    setMessages([welcome()]);
    setPanelData((current) => ({ ...current, memory: [] }));
    setPanels((current) => ({
      ...current,
      memory: { phase: "idle" },
    }));
    push("info", `已切换会话：${normalized}`);
    return normalized;
  }

  const updateAssistant = (
    id: string,
    content: string,
    status: Message["status"],
    turn?: TurnMeta,
    citations?: Message["citations"],
    toolsUsed?: string[],
  ) =>
    setMessages((items) =>
      items.map((item) =>
        item.id === id
          ? {
              ...item,
              content,
              status,
              turn: turn || item.turn,
              citations: citations || item.citations,
              toolsUsed: toolsUsed || item.toolsUsed,
              meta: turn?.trace_id || item.meta,
            }
          : item,
      ),
    );
  const consumePayload = (
    id: string,
    payload: Record<string, unknown>,
    text: string,
  ) => {
    const turn: TurnMeta = {
      trace_id: String(payload.trace_id || ""),
      turn_id: String(payload.turn_id || ""),
      stop_reason: String(payload.stop_reason || ""),
      steps: Number(payload.steps || 0),
      latency_ms: Number(payload.latency_ms || 0),
      executor: String(payload.executor || ""),
      prompt_version: String(payload.prompt_version || ""),
      tool_trace: Array.isArray(payload.tool_trace)
        ? (payload.tool_trace as ToolTrace[])
        : [],
    };
    updateAssistant(
      id,
      String(payload.answer || text || ""),
      "done",
      turn,
      Array.isArray(payload.citations)
        ? (payload.citations as Message["citations"])
        : [],
      Array.isArray(payload.tools_used) ? (payload.tools_used as string[]) : [],
    );
    if (turn.trace_id) void loadPanel("traces");
  };
  async function readStream(response: Response, id: string) {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("响应流不可用");
    const decoder = new TextDecoder();
    let answer = "";

    if (
      !(response.headers.get("content-type") || "").includes(
        "text/event-stream",
      )
    ) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        answer += decoder.decode(value, { stream: true });
        updateAssistant(id, answer, "streaming");
      }
      answer += decoder.decode();
      updateAssistant(id, answer, "done");
      return;
    }

    let buffer = "";
    let completed = false;
    const consumeFrame = (frame: string) => {
      const raw = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.replace(/^data:\s*/, ""))
        .join("\n");
      if (!raw) return;
      const payload = JSON.parse(raw) as Record<string, unknown>;
      if (payload.type === "chunk") {
        answer += String(payload.content || "");
        updateAssistant(id, answer, "streaming");
      } else if (payload.type === "done") {
        completed = true;
        consumePayload(id, payload, answer);
      } else if (payload.type === "error") {
        throw new Error(
          String(payload.message || payload.detail || "流式请求失败"),
        );
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      frames.forEach(consumeFrame);
    }
    buffer += decoder.decode();
    if (buffer.trim()) consumeFrame(buffer);
    if (!completed) throw new Error("响应流在完成事件前中断");
  }
  async function sendMessage(text = input) {
    const message = text.trim();
    if (!message || isSending) return;
    const currentSessionId = commitSession();
    if (!currentSessionId) return;
    if (!accessGranted) {
      if (authRequired) openLogin();
      else push("warn", "后端状态检查中，请稍后重试");
      return;
    }
    const assistantId = uid();
    setMessages((items) => [
      ...items,
      { id: uid(), role: "user", content: message, meta: currentSessionId },
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
    const authEpoch = authEpochRef.current;
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await request("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId, message }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await responseError(response));
      if (authEpochRef.current !== authEpoch) return;
      if (
        (response.headers.get("content-type") || "").includes(
          "application/json",
        )
      )
        consumePayload(assistantId, await response.json(), "");
      else await readStream(response, assistantId);
      if (authEpochRef.current !== authEpoch) return;
      push("ok", "Agent 响应完成");
      void refreshSessions().catch(() => undefined);
    } catch (error) {
      if (authEpochRef.current !== authEpoch) return;
      const detail =
        error instanceof DOMException && error.name === "AbortError"
          ? "请求已停止。"
          : error instanceof Error
            ? error.message
            : String(error);
      updateAssistant(assistantId, detail, "error");
      push("error", detail);
    } finally {
      if (authEpochRef.current === authEpoch) setIsSending(false);
      if (abortRef.current === controller) abortRef.current = null;
    }
  }
  async function loadPanel(next: Tab = tab) {
    if (!accessGranted) {
      setPanels((current) => ({
        ...current,
        [next]: {
          phase: "forbidden",
          message: authRequired ? "登录后才能查看此面板" : "后端尚未就绪",
        },
      }));
      if (authRequired) openLogin();
      return;
    }
    const currentSessionId = next === "memory" ? commitSession() : sessionId;
    if (next === "memory" && !currentSessionId) {
      setPanelData((current) => ({ ...current, memory: [] }));
      setPanels((current) => ({
        ...current,
        memory: {
          phase: "error",
          message: "请先输入有效的 Session ID",
        },
      }));
      return;
    }
    const path: Partial<Record<Tab, string>> = {
      memory: "/memory/" + encodeURIComponent(currentSessionId || ""),
      knowledge: "/knowledge/documents?limit=50",
      security: "/security/approvals?limit=20",
      traces: "/observability/traces?limit=20",
    };
    if (!path[next]) return;
    const requestId = ++panelRequestRef.current[next];
    const authEpoch = authEpochRef.current;
    let failurePhase: PanelPhase = "error";
    setPanels((current) => ({
      ...current,
      [next]: { phase: "loading" },
    }));
    try {
      const response = await request(path[next]!);
      if (response.status === 403) failurePhase = "forbidden";
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      const nextItems = Array.isArray(data.items) ? data.items : [data];
      let auditItems: unknown[] | null = null;
      if (next === "security") {
        const audit = await request("/security/audit?limit=30");
        if (audit.status === 403) failurePhase = "forbidden";
        if (!audit.ok) throw new Error(await responseError(audit));
        const auditData = await audit.json();
        const chain = auditData.chain as
          { valid?: boolean; checked?: number; reason?: string } | undefined;
        auditItems = [
          ...(chain
            ? [
                {
                  event_type: chain.valid
                    ? "audit.chain_verified"
                    : "audit.chain_broken",
                  actor: "system",
                  time: Date.now() / 1000,
                  outcome: chain.valid ? "success" : "failed",
                  checked: chain.checked,
                  reason: chain.reason,
                },
              ]
            : []),
          ...(Array.isArray(auditData.items) ? auditData.items : []),
        ];
      }
      if (
        panelRequestRef.current[next] !== requestId ||
        authEpochRef.current !== authEpoch
      )
        return;
      setPanelData((current) => ({
        ...current,
        [next]: nextItems,
        ...(auditItems ? { audit: auditItems } : {}),
      }));
      setPanels((current) => ({
        ...current,
        [next]: { phase: "ready" },
      }));
    } catch (error) {
      if (
        panelRequestRef.current[next] !== requestId ||
        authEpochRef.current !== authEpoch
      )
        return;
      const detail = error instanceof Error ? error.message : String(error);
      setPanelData((current) => ({
        ...current,
        [next]: [],
        ...(next === "security" ? { audit: [] } : {}),
      }));
      if (next === "security") setApprovalBusy([]);
      setPanels((current) => ({
        ...current,
        [next]: { phase: failurePhase, message: detail },
      }));
      push("warn", detail);
    }
  }
  function selectTab(next: Tab) {
    setTab(next);
    if (next !== "run") void loadPanel(next);
    if (window.matchMedia("(max-width: 920px)").matches)
      window.setTimeout(
        () => inspectorRef.current?.scrollIntoView({ behavior: "smooth" }),
        0,
      );
  }
  function handleTabKeyDown(
    event: React.KeyboardEvent<HTMLButtonElement>,
    current: Tab,
  ) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const ids = nav.map(([id]) => id);
    const currentIndex = ids.indexOf(current);
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? ids.length - 1
          : (currentIndex +
              (event.key === "ArrowRight" ? 1 : -1) +
              ids.length) %
            ids.length;
    const next = ids[nextIndex];
    selectTab(next);
    window.requestAnimationFrame(() => tabRefs.current[next]?.focus());
  }
  async function login(event: React.FormEvent) {
    event.preventDefault();
    const authEpoch = authEpochRef.current;
    setLoadingLogin(true);
    setLoginError("");
    try {
      const response = await fetch(API_BASE + "/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(credentials),
        credentials: "include",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "登录失败");
      if (authEpochRef.current !== authEpoch) return;
      authEpochRef.current += 1;
      for (const panel of Object.keys(panelRequestRef.current) as Tab[]) {
        panelRequestRef.current[panel] += 1;
      }
      resetUserWorkspace();
      setUser(data.user);
      setCredentials({ username: "", password: "" });
      setLoginOpen(false);
      push("ok", "已登录为 " + data.user.username);
      await restoreLatestSession(authEpochRef.current);
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingLogin(false);
    }
  }
  async function decideApproval(id: string, approve: boolean) {
    if (!id || approvalBusy.includes(id)) return;
    const authEpoch = authEpochRef.current;
    let responseStatus = 0;
    setApprovalBusy((current) => [...current, id]);
    try {
      const response = await request(
        "/security/approvals/" + encodeURIComponent(id),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ approve }),
        },
      );
      responseStatus = response.status;
      if (!response.ok) throw new Error(await responseError(response));
      const result = (await response.json()) as Record<string, unknown>;
      if (authEpochRef.current !== authEpoch) return;
      const status = String(result.status || "unknown");
      setPanelData((current) => ({
        ...current,
        security: (current.security || []).map((item) => {
          const row = item as Record<string, unknown>;
          return String(row.approval_id || row.id) === id
            ? { ...row, ...result }
            : item;
        }),
      }));
      push(
        status === (approve ? "approved" : "rejected") ? "ok" : "warn",
        `审批当前状态：${status}`,
      );
      void loadPanel("security");
    } catch (error) {
      if (authEpochRef.current !== authEpoch) return;
      const detail = error instanceof Error ? error.message : String(error);
      if (responseStatus === 403) {
        panelRequestRef.current.security += 1;
        setPanelData((current) => ({ ...current, security: [], audit: [] }));
        setPanels((current) => ({
          ...current,
          security: { phase: "forbidden", message: detail },
        }));
      } else if (responseStatus === 404 || responseStatus === 409) {
        setPanelData((current) => ({
          ...current,
          security: (current.security || []).filter((item) => {
            const row = item as Record<string, unknown>;
            return String(row.approval_id || row.id) !== id;
          }),
        }));
      }
      push("error", detail);
    } finally {
      if (authEpochRef.current === authEpoch) {
        setApprovalBusy((current) => current.filter((item) => item !== id));
      }
    }
  }

  function startNewSession() {
    commitSession("web-" + Date.now());
  }

  async function selectHistoricalSession(nextSessionId: string) {
    if (!nextSessionId || nextSessionId === sessionId || sessionsLoading) return;
    setSessionsLoading(true);
    try {
      await loadSessionHistory(nextSessionId, { announce: true });
    } catch (error) {
      push(
        "error",
        "会话加载失败：" +
          (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setSessionsLoading(false);
    }
  }

  async function deleteMemory() {
    const currentSessionId = commitSession();
    if (!currentSessionId) return;
    if (!window.confirm(`确定删除会话 ${currentSessionId} 的全部记忆吗？`)) {
      return;
    }
    const authEpoch = authEpochRef.current;
    try {
      const response = await request(
        "/memory/" + encodeURIComponent(currentSessionId),
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error(await responseError(response));
      const result = (await response.json()) as { deleted?: boolean };
      if (authEpochRef.current !== authEpoch) return;
      setPanelData((current) => ({ ...current, memory: [] }));
      commitSession("web-" + Date.now());
      void refreshSessions().catch(() => undefined);
      push(
        result.deleted ? "ok" : "warn",
        result.deleted ? "会话记忆已删除" : "该会话没有可删除的记忆",
      );
    } catch (error) {
      if (authEpochRef.current !== authEpoch) return;
      push("error", error instanceof Error ? error.message : String(error));
    }
  }

  async function pollIngestJob(
    initial: IngestJob,
    authEpoch: number,
  ): Promise<IngestJob> {
    let current = initial;
    for (let attempt = 0; attempt < 120; attempt += 1) {
      if (authEpochRef.current !== authEpoch) {
        throw new DOMException("账号已切换", "AbortError");
      }
      let response: Response;
      try {
        response = await request(
          "/ingest/" + encodeURIComponent(initial.job_id),
        );
      } catch (error) {
        if (authEpochRef.current !== authEpoch) {
          throw new DOMException("账号已切换", "AbortError");
        }
        if (!(error instanceof TypeError)) throw error;
        current = {
          ...current,
          status: "waiting",
          error_detail: "暂时无法连接任务状态服务，正在重试",
        };
        setIngestJob(current);
        await sleep(1000);
        continue;
      }
      if (authEpochRef.current !== authEpoch) {
        throw new DOMException("账号已切换", "AbortError");
      }
      if (response.status === 404) {
        const detail = await responseError(response);
        current = {
          ...current,
          status: "not_found",
          error_detail: detail || "任务不存在或已过期",
        };
        setIngestJob(current);
        throw new Error(current.error_detail);
      }
      if ([429, 502, 503, 504].includes(response.status)) {
        const detail = await responseError(response);
        current = {
          ...current,
          status: "waiting",
          error_detail: detail || "任务状态服务暂时不可用，正在重试",
        };
        setIngestJob(current);
        await sleep(Math.min(3000, 750 + attempt * 50));
        continue;
      }
      if (!response.ok) throw new Error(await responseError(response));
      current = (await response.json()) as IngestJob;
      setIngestJob(current);
      if (current.status === "completed") return current;
      if (["failed", "skipped"].includes(current.status)) {
        throw new Error(current.error_detail || `入库任务 ${current.status}`);
      }
      await sleep(1000);
    }
    return { ...current, status: "processing" };
  }

  async function ingest() {
    if (!knowledgeText.trim() || !accessGranted || !canWriteKnowledge) return;
    const authEpoch = authEpochRef.current;
    setIngesting(true);
    setIngestJob(null);
    try {
      const response = await request("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: sourceName || "upload",
          text: knowledgeText,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const accepted = (await response.json()) as {
        job_id: string;
        accepted: boolean;
      };
      if (authEpochRef.current !== authEpoch) return;
      const initial: IngestJob = {
        job_id: accepted.job_id,
        source: sourceName || "upload",
        status: "queued",
      };
      setIngestJob(initial);
      push("info", `知识写入任务已提交：${accepted.job_id}`);
      const completed = await pollIngestJob(initial, authEpoch);
      if (authEpochRef.current !== authEpoch) return;
      setKnowledgeText("");
      if (completed.status === "completed") {
        push("ok", "知识索引构建完成");
        void loadPanel("knowledge");
      } else if (completed.status === "submitted") {
        push("warn", "任务已提交，但当前未启用入库状态持久化");
      } else {
        push("warn", "任务仍在后台处理，可稍后刷新知识库");
      }
    } catch (error) {
      if (authEpochRef.current !== authEpoch) return;
      setIngestJob((current) =>
        current && current.status !== "not_found"
          ? {
              ...current,
              status: "failed",
              error_detail:
                error instanceof Error ? error.message : String(error),
            }
          : current,
      );
      push("error", error instanceof Error ? error.message : String(error));
    } finally {
      if (authEpochRef.current === authEpoch) setIngesting(false);
    }
  }

  React.useEffect(() => {
    if (
      accessGranted &&
      tab !== "run" &&
      ["forbidden", "idle"].includes(panels[tab].phase)
    ) {
      void loadPanel(tab);
    }
    // Reload only when access transitions; tab changes already call loadPanel.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessGranted]);

  const state = health ? health.status || "ready" : "offline";
  const tone =
    state === "ready" ? "ok" : state === "offline" ? "error" : "warn";
  const sessionBackend =
    caps?.memory?.session_backend ||
    health?.session_store?.backend ||
    "checking";
  const identityLabel =
    user?.username || (health && !authRequired ? "本地模式" : "未登录");
  const observabilityDetail = caps?.observability?.langfuse
    ? `Langfuse ${caps.observability.langfuse_api || "v3"}`
    : caps?.observability?.prometheus
      ? "Local traces + Prometheus"
      : caps?.observability?.local_trace_store
        ? "Local trace store"
        : "";
  const tabTitle: Record<Tab, string> = {
    run: "运行控制台",
    memory: "会话记忆",
    knowledge: "知识库",
    security: "安全护栏",
    traces: "执行观测",
  };
  const nav = [
    ["run", Activity, "运行状态"],
    ["memory", BrainCircuit, "记忆"],
    ["knowledge", BookOpenText, "知识库"],
    ["security", ShieldCheck, "安全"],
    ["traces", Route, "观测"],
  ] as const;
  const list = (panelData[tab] || []) as Array<Record<string, unknown>>;
  const activePanel = panels[tab];
  const liveStatus = isSending
    ? "Agent 正在生成响应"
    : events[0]?.text || "工作台已就绪";
  return (
    <>
      <div className="console-shell" ref={shellRef}>
        <div
          className="sr-only"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {liveStatus}
        </div>
        <aside className="side-rail">
          <div className="brand-block">
            <span className="brand-mark" aria-hidden="true">
              <img src="/assets/sloth-mascot-128.png" alt="" />
            </span>
            <div className="brand-copy">
              <strong>SlothBearFlow</strong>
              <span>Agent Console</span>
            </div>
          </div>
          <nav className="rail-nav" aria-label="工作台导航">
            {nav.map(([id, Icon, label]) => (
              <button
                key={id}
                type="button"
                className={tab === id ? "active" : ""}
                onClick={() => selectTab(id)}
                title={label}
                aria-current={tab === id ? "page" : undefined}
              >
                <Icon size={18} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
          <div className="rail-spacer" />
          <section className="session-block">
            <div className="section-kicker">
              <History size={14} />
              <span>当前会话</span>
            </div>
            <label htmlFor="session-id">历史会话</label>
            <div className="session-field">
              <select
                id="session-id"
                value={sessionId}
                disabled={sessionsLoading || isSending}
                onFocus={() => void refreshSessions().catch(() => undefined)}
                onChange={(event) =>
                  void selectHistoricalSession(event.target.value)
                }
              >
                {!sessions.some((item) => item.session_id === sessionId) ? (
                  <option value={sessionId}>新会话 · {sessionId}</option>
                ) : null}
                {sessions.map((item) => (
                  <option key={item.session_id} value={item.session_id}>
                    {formatSessionOption(item)}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => {
                  startNewSession();
                }}
                title="新建会话"
                aria-label="新建会话"
              >
                <Plus size={16} />
              </button>
            </div>
            {sessionError ? (
              <p className="session-error" id="session-id-error" role="alert">
                {sessionError}
              </p>
            ) : null}
            <div className="session-stats">
              <span>
                <strong>{conversationMessageCount}</strong>消息
              </span>
              <span>
                <strong>{caps?.memory?.window_pairs ?? "-"}</strong>记忆窗口
              </span>
            </div>
          </section>
          <div className="rail-status">
            <span className={"status-dot " + tone} />
            <div>
              <strong>{state}</strong>
              <span>{identityLabel}</span>
            </div>
          </div>
        </aside>
        <main className="conversation">
          <header className="conversation-header">
            <div className="conversation-title">
              <span className="eyebrow">Agent Workspace</span>
              <h1>实时对话</h1>
              <div className="runtime-line">
                <span>{caps?.agent?.executor || "basic"}</span>
                <i />
                <span>{caps?.agent?.stream_format || "JSON"}</span>
                <i />
                <span>Memory · {sessionBackend}</span>
              </div>
            </div>
            <div className="header-actions">
              <button
                className="icon-action"
                onClick={startNewSession}
                title="新建会话"
                aria-label="新建会话"
              >
                <Plus size={17} />
              </button>
              <button
                className="icon-action"
                onClick={() => void refreshHealth()}
                title="刷新运行状态"
                aria-label="刷新运行状态"
              >
                <RefreshCw size={17} />
              </button>
              {user ? (
                <button
                  className="icon-action"
                  onClick={() => void signOut()}
                  title="退出登录"
                  aria-label="退出登录"
                  disabled={signingOut}
                >
                  {signingOut ? (
                    <Loader2 size={17} className="spin" />
                  ) : (
                    <LogOut size={17} />
                  )}
                </button>
              ) : authRequired ? (
                <button
                  className="icon-action"
                  onClick={openLogin}
                  title="登录"
                  aria-label="登录"
                >
                  <LogIn size={17} />
                </button>
              ) : null}
              <button
                className="icon-action"
                onClick={() => setMessages([welcome()])}
                title="清空对话"
                aria-label="清空对话"
              >
                <Trash2 size={17} />
              </button>
            </div>
            <div className="mobile-session">
              <label htmlFor="mobile-session-id">历史会话</label>
              <div className="mobile-session-field">
                <select
                  id="mobile-session-id"
                  value={sessionId}
                  disabled={sessionsLoading || isSending}
                  onFocus={() => void refreshSessions().catch(() => undefined)}
                  onChange={(event) =>
                    void selectHistoricalSession(event.target.value)
                  }
                >
                  {!sessions.some((item) => item.session_id === sessionId) ? (
                    <option value={sessionId}>新会话 · {sessionId}</option>
                  ) : null}
                  {sessions.map((item) => (
                    <option key={item.session_id} value={item.session_id}>
                      {formatSessionOption(item)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={startNewSession}
                  title="新建会话"
                  aria-label="新建会话"
                >
                  <Plus size={16} />
                </button>
              </div>
              {sessionError ? (
                <p
                  className="session-error"
                  id="mobile-session-id-error"
                  role="alert"
                >
                  {sessionError}
                </p>
              ) : null}
            </div>
          </header>
          <div
            className="message-stream"
            ref={streamRef}
            aria-live="polite"
            aria-busy={isSending}
          >
            {messages.map((message) => (
              <article className={"message " + message.role} key={message.id}>
                <div className="avatar">
                  {message.role === "user" ? (
                    <UserRound size={17} />
                  ) : (
                    <img src="/assets/sloth-mascot-128.png" alt="" />
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
                    ) : message.status === "error" ? (
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
                      message.content || "正在生成响应..."
                    )}
                  </div>
                  {message.turn ? (
                    <div className="turn-meta">
                      <span>
                        <Route size={12} />
                        {message.turn.trace_id || "trace pending"}
                      </span>
                      <span>
                        <TimerReset size={12} />
                        {message.turn.latency_ms?.toFixed(0) || "-"} ms
                      </span>
                      <span>{message.turn.executor || "agent"}</span>
                      <span>{message.turn.stop_reason || "running"}</span>
                    </div>
                  ) : null}
                  {message.turn?.tool_trace?.length ? (
                    <div className="tool-trace">
                      {message.turn.tool_trace.map((tool, index) => (
                        <details key={tool.call_id || index}>
                          <summary>
                            <Wrench size={12} />
                            {tool.name || "tool"}
                            <span>{tool.duration_ms?.toFixed(0) || 0} ms</span>
                            <b className={tool.ok ? "ok" : "error"}>
                              {tool.status || (tool.ok ? "ok" : "failed")}
                            </b>
                          </summary>
                          <p>
                            {tool.policy_decision || "allow"}
                            {tool.observation ? " · " + tool.observation : ""}
                          </p>
                        </details>
                      ))}
                    </div>
                  ) : null}
                  {message.citations?.some(
                    (citation) => citation.supported !== false,
                  ) ? (
                    <div className="citation-block">
                      <span className="evidence-label">
                        <BookOpenText size={13} />
                        引用来源
                      </span>
                      <div className="citation-list">
                        {message.citations
                          .filter((citation) => citation.supported !== false)
                          .slice(0, 4)
                          .map((citation, index) => (
                            <div
                              className="citation-item"
                              key={citation.source + index}
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
            ))}
          </div>
          <div className="composer-zone">
            <div className="prompt-strip">
              {starters.map((starter) => (
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
                aria-label="Agent 消息"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  accessGranted
                    ? "向 SlothBearFlow 发送消息"
                    : authRequired
                      ? "请先登录以发送消息"
                      : "正在连接后端"
                }
                disabled={!accessGranted}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendMessage();
                  }
                }}
              />
              {isSending ? (
                <button
                  className="stop-button"
                  type="button"
                  onClick={() => abortRef.current?.abort()}
                  title="停止生成"
                  aria-label="停止生成"
                >
                  <Square size={16} />
                </button>
              ) : (
                <button
                  className="send-button"
                  disabled={!input.trim() || !accessGranted}
                  type="submit"
                  title="发送消息"
                  aria-label="发送消息"
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
              <h2>{tabTitle[tab]}</h2>
            </div>
            <button
              className="icon-action"
              onClick={() =>
                tab === "run" ? void refreshHealth() : void loadPanel()
              }
              title="刷新面板"
              aria-label="刷新面板"
              disabled={activePanel.phase === "loading"}
            >
              <RefreshCw
                size={16}
                className={activePanel.phase === "loading" ? "spin" : ""}
              />
            </button>
          </header>
          <div
            className="inspector-tabs"
            role="tablist"
            aria-label="运维面板"
            aria-orientation="horizontal"
          >
            {nav.map(([id, Icon, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                id={`tab-${id}`}
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                tabIndex={tab === id ? 0 : -1}
                className={tab === id ? "active" : ""}
                onClick={() => selectTab(id)}
                onKeyDown={(event) => handleTabKeyDown(event, id)}
                title={label}
                aria-label={label}
                ref={(element) => {
                  tabRefs.current[id] = element;
                }}
              >
                <Icon size={15} />
              </button>
            ))}
          </div>
          <div
            className="inspector-content"
            id={`panel-${tab}`}
            role="tabpanel"
            aria-labelledby={`tab-${tab}`}
            aria-busy={activePanel.phase === "loading"}
            tabIndex={0}
          >
            {activePanel.phase !== "ready" ? (
              <PanelFeedback state={activePanel} />
            ) : (
              <>
                {tab === "run" ? (
                  <>
                    <section className="state-summary">
                      <div className={"summary-signal " + tone}>
                        <Zap size={18} />
                      </div>
                      <div>
                        <span>Agent runtime</span>
                        <strong>
                          {health?.llm
                            ? health.llm.provider + " / " + health.llm.model
                            : "等待后端"}
                        </strong>
                      </div>
                    </section>
                    <section className="inspector-section">
                      <div className="section-heading">
                        <Gauge size={15} />
                        <h3>能力与依赖</h3>
                      </div>
                      <div className="inspector-list">
                        <InspectorRow
                          icon={Database}
                          label="会话记忆"
                          value={sessionBackend}
                          detail={
                            health?.redis?.ok
                              ? "Redis 可用"
                              : health?.redis?.error
                          }
                          tone={health?.redis?.ok ? "ok" : "warn"}
                        />
                        <InspectorRow
                          icon={ShieldCheck}
                          label="安全护栏"
                          value={caps?.security?.tool_guard_mode || "unknown"}
                          detail={caps?.security?.approval_mode}
                          tone="ok"
                        />
                        <InspectorRow
                          icon={Network}
                          label="MCP"
                          value={caps?.mcp?.enabled ? "enabled" : "off"}
                          detail={
                            Array.isArray(caps?.mcp?.servers)
                              ? caps?.mcp?.servers.length + " servers"
                              : ""
                          }
                        />
                        <InspectorRow
                          icon={Activity}
                          label="可观测性"
                          value={
                            caps?.observability?.enabled ? "enabled" : "off"
                          }
                          detail={observabilityDetail}
                        />
                      </div>
                    </section>
                    <section className="inspector-section event-section">
                      <div className="section-heading">
                        <Clock3 size={15} />
                        <h3>工作台事件</h3>
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
                {tab === "memory" ? (
                  <PanelList
                    icon={BrainCircuit}
                    title="当前会话快照"
                    items={list}
                    empty="当前会话还没有短期或摘要记忆"
                    render={(item) => {
                      const memory = (item.memory || item) as Record<
                        string,
                        unknown
                      >;
                      const messages = Array.isArray(memory.messages)
                        ? memory.messages
                        : [];
                      return (
                        <>
                          <strong>
                            {String(item.session_id || sessionId)}
                          </strong>
                          <p>{String(memory.summary || "尚未生成会话摘要")}</p>
                          <span className="panel-meta">
                            {messages.length} 条消息 · 版本{" "}
                            {String(memory.version || 0)}
                          </span>
                          {canDeleteMemory ? (
                            <div className="panel-actions">
                              <button
                                className="danger"
                                title="删除会话记忆"
                                aria-label="删除会话记忆"
                                onClick={() => void deleteMemory()}
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          ) : null}
                        </>
                      );
                    }}
                  />
                ) : null}
                {tab === "knowledge" ? (
                  <>
                    <PanelList
                      icon={BookOpenText}
                      title="已登记文档"
                      items={list}
                      empty="尚未发现已登记文档"
                      render={(item) => (
                        <>
                          <strong>
                            {String(item.source || item.name || "document")}
                          </strong>
                          <p>
                            {String(item.status || item.visibility || "已登记")}
                          </p>
                        </>
                      )}
                    />
                    <section className="inspector-section ingest-section">
                      <div className="section-heading">
                        <FilePlus2 size={15} />
                        <h3>写入知识库</h3>
                      </div>
                      <label htmlFor="source-name">来源名称</label>
                      <input
                        id="source-name"
                        value={sourceName}
                        onChange={(e) => setSourceName(e.target.value)}
                      />
                      <label htmlFor="knowledge-text">文档内容</label>
                      <textarea
                        id="knowledge-text"
                        value={knowledgeText}
                        onChange={(e) => setKnowledgeText(e.target.value)}
                        placeholder="粘贴待索引的文档内容"
                      />
                      <button
                        className="primary-action"
                        disabled={
                          ingesting ||
                          !knowledgeText.trim() ||
                          !accessGranted ||
                          !canWriteKnowledge
                        }
                        onClick={() => void ingest()}
                      >
                        {ingesting ? (
                          <Loader2 size={16} className="spin" />
                        ) : (
                          <FilePlus2 size={16} />
                        )}
                        {ingesting ? "正在构建索引" : "写入知识库"}
                      </button>
                      {ingestJob ? (
                        <div
                          className={
                            "job-result " +
                            (ingestJob.status === "completed"
                              ? ""
                              : ["failed", "skipped", "not_found"].includes(
                                    ingestJob.status,
                                  )
                                ? "error"
                                : "pending")
                          }
                        >
                          <div>
                            {ingestJob.status === "completed" ? (
                              <CheckCircle2 size={15} />
                            ) : ["failed", "skipped", "not_found"].includes(
                                ingestJob.status,
                              ) ? (
                              <XCircle size={15} />
                            ) : (
                              <Loader2 size={15} className="spin" />
                            )}
                            <strong>{ingestJob.source}</strong>
                            <span>{ingestJob.status}</span>
                          </div>
                          <p>
                            {ingestJob.job_id}
                            {ingestJob.error_detail
                              ? ` · ${ingestJob.error_detail}`
                              : ""}
                          </p>
                        </div>
                      ) : null}
                    </section>
                  </>
                ) : null}
                {tab === "security" ? (
                  <>
                    <PanelList
                      icon={ShieldAlert}
                      title="最近审批"
                      items={list}
                      empty="暂无审批记录"
                      render={(item) => {
                        const approvalId = String(
                          item.approval_id || item.id || "",
                        );
                        const status = String(item.status || "unknown");
                        const busy = approvalBusy.includes(approvalId);
                        return (
                          <>
                            <strong>
                              {String(
                                item.tool_name ||
                                  item.action ||
                                  item.id ||
                                  "approval",
                              )}
                            </strong>
                            <p>
                              {item.args_summary
                                ? typeof item.args_summary === "string"
                                  ? item.args_summary
                                  : JSON.stringify(item.args_summary)
                                : String(item.reason || "未提供参数摘要")}
                            </p>
                            <span className={`panel-meta status-${status}`}>
                              状态：{status}
                            </span>
                            {canApproveTools && status === "pending" ? (
                              <div className="approval-actions">
                                <button
                                  title="同意"
                                  aria-label="同意审批"
                                  disabled={busy}
                                  onClick={() =>
                                    void decideApproval(approvalId, true)
                                  }
                                >
                                  {busy ? (
                                    <Loader2 size={14} className="spin" />
                                  ) : (
                                    <Check size={14} />
                                  )}
                                </button>
                                <button
                                  title="拒绝"
                                  aria-label="拒绝审批"
                                  disabled={busy}
                                  onClick={() =>
                                    void decideApproval(approvalId, false)
                                  }
                                >
                                  <XCircle size={14} />
                                </button>
                              </div>
                            ) : null}
                          </>
                        );
                      }}
                    />
                    <PanelList
                      icon={History}
                      title="安全审计"
                      items={
                        (panelData.audit || []) as Array<
                          Record<string, unknown>
                        >
                      }
                      empty="暂无审计记录"
                      render={(item) => (
                        <>
                          <strong>
                            {String(
                              item.event_type ||
                                item.event ||
                                item.action ||
                                "audit",
                            )}
                          </strong>
                          <p>
                            {String(item.actor || "system")} ·{" "}
                            {formatTimestamp(item.timestamp || item.time)}
                          </p>
                        </>
                      )}
                    />
                  </>
                ) : null}
                {tab === "traces" ? (
                  <PanelList
                    icon={Route}
                    title="最近 Trace"
                    items={list}
                    empty="发送对话后将显示可观测性记录"
                    render={(item) => {
                      const spans = Array.isArray(item.spans) ? item.spans : [];
                      return (
                        <>
                          <strong>
                            {String(item.trace_id || item.id || "trace")}
                          </strong>
                          <p className="trace-route">
                            {traceRequestLabel(item)}
                          </p>
                          <span className="panel-meta">
                            {String(item.status || "unknown")} · {spans.length}{" "}
                            spans ·{" "}
                            {String(item.duration_ms ?? item.latency_ms ?? "-")}{" "}
                            ms
                          </span>
                        </>
                      );
                    }}
                  />
                ) : null}
              </>
            )}
          </div>
        </aside>
      </div>
      {dialogOpen ? (
        <div className="login-backdrop">
          <form
            className="login-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="login-title"
            aria-describedby="login-description"
            aria-busy={loadingLogin}
            onSubmit={login}
            onKeyDown={trapDialogFocus}
          >
            <span className="sr-only" id="login-description">
              使用工作台账户登录
            </span>
            <span className="brand-mark" aria-hidden="true">
              <img src="/assets/sloth-mascot-128.png" alt="" />
            </span>
            <h2 id="login-title">登录工作台</h2>
            <label>
              用户名
              <input
                autoFocus
                autoComplete="username"
                value={credentials.username}
                onChange={(e) =>
                  setCredentials((value) => ({
                    ...value,
                    username: e.target.value,
                  }))
                }
              />
            </label>
            <label>
              密码
              <input
                type="password"
                autoComplete="current-password"
                value={credentials.password}
                onChange={(e) =>
                  setCredentials((value) => ({
                    ...value,
                    password: e.target.value,
                  }))
                }
              />
            </label>
            {loginError ? (
              <p className="login-error" role="alert">
                {loginError}
              </p>
            ) : null}
            <button
              className="primary-action"
              disabled={
                loadingLogin || !credentials.username || !credentials.password
              }
            >
              {loadingLogin ? (
                <Loader2 size={16} className="spin" />
              ) : (
                <LogIn size={16} />
              )}
              登录
            </button>
          </form>
        </div>
      ) : null}
    </>
  );
}

function PanelFeedback({ state }: { state: PanelState }) {
  const loading = state.phase === "loading" || state.phase === "idle";
  const forbidden = state.phase === "forbidden";
  return (
    <div
      className={`panel-feedback ${state.phase}`}
      role={loading ? "status" : "alert"}
    >
      <span className="feedback-icon">
        {loading ? (
          <Loader2 size={18} className="spin" />
        ) : forbidden ? (
          <ShieldAlert size={18} />
        ) : (
          <AlertCircle size={18} />
        )}
      </span>
      <div>
        <strong>
          {loading
            ? state.phase === "idle"
              ? "等待加载"
              : "正在加载"
            : forbidden
              ? "无权访问"
              : "加载失败"}
        </strong>
        <p>
          {state.message || (loading ? "正在获取最新面板数据" : "请稍后重试")}
        </p>
      </div>
    </div>
  );
}

function PanelList({
  icon: Icon,
  title,
  items,
  empty,
  render,
}: {
  icon: React.ElementType;
  title: string;
  items: Array<Record<string, unknown>>;
  empty: string;
  render: (item: Record<string, unknown>) => React.ReactNode;
}) {
  return (
    <section className="inspector-section">
      <div className="section-heading">
        <Icon size={15} />
        <h3>{title}</h3>
        <span>{items.length}</span>
      </div>
      <div className="panel-list">
        {items.length ? (
          items.map((item, index) => (
            <article
              className="panel-item"
              key={String(
                item.id || item.trace_id || item.approval_id || index,
              )}
            >
              {render(item)}
            </article>
          ))
        ) : (
          <div className="empty-state">{empty}</div>
        )}
      </div>
    </section>
  );
}
