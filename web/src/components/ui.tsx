import type { ReactNode } from "react";
import { useInView } from "../lib/hooks";
import { TONE, type Tone } from "../lib/format";
import Backdrop, { type BackdropVariant } from "./Backdrop";

/**
 * Section headings.
 *
 * No index numbers. A numbered heading tells the reader they are working through a
 * document, which is exactly the wrong posture for a page whose job is to make you want to
 * look at the thing. The ornament is a short coloured rule and a kicker — enough to mark a
 * new movement, not enough to feel like a table of contents.
 */
export function SectionHead({
  kicker,
  title,
  lede,
  tone = "gold",
  align = "left",
}: {
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  tone?: Tone;
  align?: "left" | "center";
}) {
  const centered = align === "center";
  return (
    <header className={`mb-12 md:mb-16 ${centered ? "text-center" : ""}`}>
      <div className={`flex items-center gap-3 ${centered ? "justify-center" : ""}`}>
        <span
          className="h-[2px] w-7 rounded-full"
          style={{ background: TONE[tone] }}
          aria-hidden
        />
        <span className="mono text-[10.5px] uppercase tracking-[0.26em] text-paper-3">
          {kicker}
        </span>
      </div>
      <h2
        className={`display mt-5 text-[clamp(1.95rem,4.4vw,3.35rem)] font-extrabold text-paper text-balance ${
          centered ? "mx-auto max-w-[20ch]" : ""
        }`}
      >
        {title}
      </h2>
      {lede ? (
        <p
          className={`mt-6 text-[17px] leading-[1.72] text-paper-2 md:text-[19.5px] ${
            centered ? "mx-auto max-w-[62ch]" : "max-w-[64ch]"
          }`}
        >
          {lede}
        </p>
      ) : null}
    </header>
  );
}

export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const { ref, inView } = useInView<HTMLDivElement>();
  return (
    <div
      ref={ref}
      data-shown={inView}
      className={`reveal ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

export function Section({
  id,
  children,
  className = "",
  backdrop,
  tone = "gold",
}: {
  id: string;
  children: ReactNode;
  className?: string;
  backdrop?: BackdropVariant;
  tone?: Tone;
}) {
  return (
    <section id={id} className={`seam relative z-10 py-28 md:py-40 ${className}`}>
      {backdrop ? <Backdrop variant={backdrop} tone={tone} /> : null}
      <div className="shell relative z-10">{children}</div>
    </section>
  );
}

/**
 * The panel everything sits in.
 *
 * One surface treatment used everywhere — a soft top-lit gradient, a hairline border and a
 * hover lift — so a stat, a chart and a case all feel like the same family of object
 * rather than three different designers' work.
 */
export function Panel({
  children,
  className = "",
  hover = false,
  glow,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: Tone;
}) {
  return (
    <div
      className={`panel ${hover ? "panel-hover" : ""} ${className}`}
      style={
        glow
          ? ({ "--panel-glow": TONE[glow] } as React.CSSProperties)
          : undefined
      }
    >
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "gold",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
}) {
  return (
    <Panel hover glow={tone} className="px-6 py-6">
      <div className="mono text-[10px] uppercase tracking-[0.2em] text-paper-3">{label}</div>
      <div
        className="mono mt-3.5 text-[clamp(1.5rem,2.8vw,2.15rem)] font-semibold leading-none"
        style={{ color: TONE[tone] }}
      >
        {value}
      </div>
      {sub ? <div className="mt-3 text-[13.5px] leading-[1.62] text-paper-3">{sub}</div> : null}
    </Panel>
  );
}

export function Badge({
  children,
  tone = "muted",
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
}) {
  return (
    <span
      title={title}
      className="mono inline-flex items-center gap-1.5 rounded-full px-2.5 py-[3px] text-[10px] uppercase tracking-[0.12em]"
      style={{
        color: TONE[tone],
        background: `color-mix(in srgb, ${TONE[tone]} 14%, transparent)`,
        border: `1px solid color-mix(in srgb, ${TONE[tone]} 26%, transparent)`,
      }}
    >
      {children}
    </span>
  );
}

/** A horizontal bar. Used everywhere a share needs to be read against its whole. */
export function Bar({
  value,
  max,
  tone = "gold",
  height = 8,
}: {
  value: number;
  max: number;
  tone?: Tone;
  height?: number;
}) {
  const w = max > 0 ? Math.max(0.6, (value / max) * 100) : 0;
  return (
    <div
      className="w-full overflow-hidden rounded-full"
      style={{ height, background: "rgba(233,230,222,0.06)" }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-700 ease-out"
        style={{
          width: `${w}%`,
          background: `linear-gradient(90deg, color-mix(in srgb, ${TONE[tone]} 55%, transparent), ${TONE[tone]})`,
        }}
      />
    </div>
  );
}

/** An aside. Short by design — if it needs three sentences it belongs in the body. */
export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-6 max-w-[68ch] border-l border-rule pl-5 text-[14.5px] leading-[1.78] text-paper-2">
      {children}
    </p>
  );
}

export function KeyValue({ k, v, tone }: { k: string; v: ReactNode; tone?: Tone }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-rule py-2 last:border-0">
      <span className="text-[12.5px] text-paper-3">{k}</span>
      <span className="mono text-[13px]" style={tone ? { color: TONE[tone] } : undefined}>
        {v}
      </span>
    </div>
  );
}
