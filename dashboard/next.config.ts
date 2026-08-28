import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Pin the workspace root. Without it Turbopack walks up looking for a
  // lockfile and can land on one outside the project.
  turbopack: { root: path.join(__dirname) },
};

export default nextConfig;
