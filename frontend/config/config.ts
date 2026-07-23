import { defineConfig } from "umi";

const apiProxyTarget =
  process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  title: "SlothBearFlow Console",
  favicons: [
    "/assets/favicon-32.png",
    "/assets/favicon-64.png",
  ],
  npmClient: "pnpm",
  mfsu: false,
  define: {
    "process.env.UMI_APP_API_BASE":
      process.env.UMI_APP_API_BASE || process.env.API_BASE || "/api",
  },
  routes: [{ path: "/", component: "index" }],
  proxy: {
    "/api": {
      target: apiProxyTarget,
      changeOrigin: true,
      pathRewrite: { "^/api": "" },
    },
    "/openapi.json": {
      target: apiProxyTarget,
      changeOrigin: true,
    },
  },
});
