import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Instrument_Serif } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { getLab } from "@/lib/server";
import { int } from "@/lib/lab";

const display = Instrument_Serif({
  weight: "400", subsets: ["latin"], variable: "--font-instrument",
});
const sans = IBM_Plex_Sans({
  weight: ["400", "500", "600"], subsets: ["latin"], variable: "--font-plex-sans",
});
const mono = IBM_Plex_Mono({
  weight: ["400", "500"], subsets: ["latin"], variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "SERP Factor Lab",
  description:
    "A ranking-factor model that reports how wrong it is. Calibrated against " +
    "a synthetic search engine whose weights are known.",
};

const NAV = [
  { href: "/", label: "Verdict", n: "00" },
  { href: "/calibration", label: "Calibration", n: "01" },
  { href: "/factors", label: "Factor bench", n: "02" },
  { href: "/clusters", label: "Intent clusters", n: "03" },
  { href: "/simulator", label: "Simulator", n: "04" },
  { href: "/experiments", label: "Experiments", n: "05" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const { meta } = getLab();
  return (
    <html lang="en">
      <body className={`${display.variable} ${sans.variable} ${mono.variable}`}>
        <div className="min-h-screen flex flex-col">
          <header className="border-b" style={{ borderColor: "var(--rule)" }}>
            <div className="mx-auto max-w-[1180px] px-6 py-5 flex flex-wrap
                            items-baseline gap-x-8 gap-y-3">
              <Link href="/" className="flex items-baseline gap-3 shrink-0">
                <span
                  className="display text-[22px]"
                  style={{ color: "var(--paper)" }}
                >
                  SERP Factor Lab
                </span>
                <span
                  className="mono text-[10px] tracking-[0.18em] uppercase"
                  style={{ color: "var(--signal)" }}
                >
                  calibrated
                </span>
              </Link>
              <nav className="flex flex-wrap gap-x-6 gap-y-2 ml-auto">
                {NAV.map((it) => (
                  <Link
                    key={it.href}
                    href={it.href}
                    className="group flex items-baseline gap-1.5 text-[13px]
                               transition-colors"
                    style={{ color: "var(--paper-dim)" }}
                  >
                    <span
                      className="mono text-[9.5px]"
                      style={{ color: "var(--paper-faint)" }}
                    >
                      {it.n}
                    </span>
                    <span className="group-hover:text-[var(--signal)]
                                     transition-colors">
                      {it.label}
                    </span>
                  </Link>
                ))}
              </nav>
            </div>
          </header>

          <main className="flex-1 mx-auto w-full max-w-[1180px] px-6 py-12">
            {children}
          </main>

          <footer
            className="border-t mt-16"
            style={{ borderColor: "var(--rule)" }}
          >
            <div className="mx-auto max-w-[1180px] px-6 py-6 flex flex-wrap
                            gap-x-7 gap-y-1.5 mono text-[10.5px]"
                 style={{ color: "var(--paper-faint)" }}>
              <span>seed {meta.seed}</span>
              <span>pilot {int(meta.pilot_queries)} queries</span>
              <span>
                scale-up{" "}
                {meta.scale_queries ? `${int(meta.scale_queries)} queries` : "skipped"}
              </span>
              <span>{meta.n_factors} factors tracked</span>
              <span>{meta.n_simulated} carried into the harness</span>
              <span>numpy {meta.numpy}</span>
              <span>pandas {meta.pandas}</span>
              <span>
                lightgbm {meta.lightgbm ? "yes" : "no, sklearn fallback"}
              </span>
              <span>run {meta.runtime_s}s</span>
              <span className="ml-auto">{meta.generated}</span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
