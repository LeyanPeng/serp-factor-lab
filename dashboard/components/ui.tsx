import { fmt, VERDICT_TONE, type Verdict } from "@/lib/lab";

/* --------------------------------------------------------------------------
   The error bar is the whole design language here. It is drawn at the same
   scale everywhere on a page so two bars can be compared by eye, and the
   ground-truth marker is a separate vertical rule so you can see at a glance
   whether the interval caught it.
   -------------------------------------------------------------------------- */

export function ErrorBar({
  lo, hi, point, truth, max = 1.15, tone = "var(--signal)", height = 22,
}: {
  lo: number; hi: number; point: number; truth?: number;
  max?: number; tone?: string; height?: number;
}) {
  const pc = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  return (
    <div className="relative w-full" style={{ height }}>
      {/* baseline */}
      <div
        className="absolute left-0 right-0 top-1/2 -translate-y-1/2"
        style={{ height: 1, background: "var(--rule-soft)" }}
      />
      {/* interval */}
      <div
        className="absolute top-1/2 -translate-y-1/2"
        style={{
          left: pc(lo), width: `calc(${pc(hi)} - ${pc(lo)})`,
          height: 5, background: tone, opacity: 0.26,
        }}
      />
      {/* interval caps */}
      {[lo, hi].map((v, i) => (
        <div
          key={i}
          className="absolute top-1/2 -translate-y-1/2"
          style={{ left: pc(v), width: 1, height: 11, background: tone,
                   opacity: 0.65 }}
        />
      ))}
      {/* point estimate */}
      <div
        className="absolute top-1/2 -translate-y-1/2"
        style={{ left: pc(point), width: 2.5, height: height - 4,
                 background: tone, marginLeft: -1 }}
      />
      {/* ground truth */}
      {truth !== undefined && (
        <div
          className="absolute top-0"
          style={{
            left: pc(truth), width: 1, height,
            background: "var(--truth)",
            boxShadow: "0 0 0 0.5px var(--ink)",
          }}
        />
      )}
    </div>
  );
}

export function VerdictBadge({ v }: { v: Verdict }) {
  const tone = VERDICT_TONE[v] ?? "var(--muted)";
  return (
    <span
      className="mono text-[9.5px] tracking-[0.11em] uppercase whitespace-nowrap
                 px-1.5 py-[3px] border"
      style={{ color: tone, borderColor: tone, opacity: 0.92 }}
    >
      {v}
    </span>
  );
}

export function TierBadge({ tier }: { tier: "A" | "B" | "C" }) {
  const tone =
    tier === "A" ? "var(--verify)" : tier === "B" ? "var(--signal)" : "var(--muted)";
  return (
    <span
      className="mono text-[9.5px] w-[15px] h-[15px] inline-flex items-center
                 justify-center border shrink-0"
      style={{ color: tone, borderColor: tone }}
      title={
        tier === "A" ? "Confirmed by Google"
        : tier === "B" ? "DOJ trial record / 2024 API leak"
        : "Industry folklore -- not tracked"
      }
    >
      {tier}
    </span>
  );
}

export function Stat({
  label, value, sub, tone = "var(--paper)", big = false,
}: {
  label: string; value: string; sub?: string; tone?: string; big?: boolean;
}) {
  return (
    <div>
      <div className="eyebrow mb-2">{label}</div>
      <div
        className={`num ${big ? "text-[40px]" : "text-[26px]"} leading-none`}
        style={{ color: tone }}
      >
        {value}
      </div>
      {sub && (
        <div className="text-[12.5px] mt-2" style={{ color: "var(--paper-dim)" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function SectionTitle({
  eyebrow, title, lede,
}: { eyebrow: string; title: string; lede?: string }) {
  return (
    <div className="mb-8 max-w-[62ch]">
      <div className="eyebrow mb-3">{eyebrow}</div>
      <h2 className="display text-[34px] mb-3">{title}</h2>
      {lede && (
        <p className="text-[15px]" style={{ color: "var(--paper-dim)" }}>
          {lede}
        </p>
      )}
    </div>
  );
}

export function Note({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="panel px-5 py-4 text-[13.5px] leading-relaxed border-l-2"
      style={{ borderLeftColor: "var(--signal)", color: "var(--paper-dim)" }}
    >
      {children}
    </div>
  );
}

export function Legend() {
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2 mono text-[10px]"
         style={{ color: "var(--paper-faint)" }}>
      <span className="flex items-center gap-2">
        <span style={{ width: 3, height: 12, background: "var(--signal)" }} />
        point estimate
      </span>
      <span className="flex items-center gap-2">
        <span style={{ width: 18, height: 5, background: "var(--signal)",
                       opacity: 0.26 }} />
        95% interval
      </span>
      <span className="flex items-center gap-2">
        <span style={{ width: 1.5, height: 14, background: "var(--truth)" }} />
        ground truth
      </span>
    </div>
  );
}

export { fmt };
