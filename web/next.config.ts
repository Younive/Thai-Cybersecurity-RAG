import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A stray lockfile in the home dir makes Next guess the wrong workspace root.
  // Pin it to this app's dir.
  turbopack: { root: __dirname },

  // Proxy API calls to the FastAPI backend (src/api.py on :8000), so the
  // browser only ever talks same-origin — no CORS in the normal path.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

export default nextConfig;
