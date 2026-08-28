import "server-only";

import fs from "node:fs";
import path from "node:path";
import type { Lab } from "./lab";

/**
 * Read the results the Python pipeline wrote. Read at request time rather
 * than imported as a module so a rerun of `run_demo.py` shows up on the next
 * refresh without a rebuild.
 */
export function getLab(): Lab {
  const p = path.join(process.cwd(), "public", "data", "lab.json");
  return JSON.parse(fs.readFileSync(p, "utf-8")) as Lab;
}
