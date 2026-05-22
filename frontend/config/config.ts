import { defineConfig } from "umi";

export default defineConfig({
  title: "SlothBearFlow Console",
  npmClient: "pnpm",
  routes: [{ path: "/", component: "index" }],
  proxy: {
    "/api": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
      pathRewrite: { "^/api": "" },
    },
    "/openapi.json": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    },
  },
});
