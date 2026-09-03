import { useId } from "react";
import { TONE, type Tone } from "../lib/format";

/**
 * Per-section background art.
 *
 * The hero and the funnel already carry WebGL. Giving every other section its own canvas
 * would mean seven live GL contexts on one page, which browsers cap and this laptop cannot
 * afford — so these are SVG and CSS only: no JavaScript loop, no context, nothing to pause
 * when they scroll away.
 *
 * Each variant is drawn from the same two nouns as everything else. `ledger` is ruled paper
 * seen in perspective. `orbit` is the concentric rings of the sieve, flattened. `stream` is
 * the hero's money crossing the frame. `grain` is the paper itself. They are deliberately
 * faint — this is the room the content stands in, not a second thing to look at.
 */
export type BackdropVariant = "ledger" | "orbit" | "stream" | "grain";

export default function Backdrop({
  variant,
  tone = "gold",
  /** 0–1. Everything here sits under body copy, so the ceiling is low on purpose. */
  intensity = 1,
}: {
  variant: BackdropVariant;
  tone?: Tone;
  intensity?: number;
}) {
  const uid = useId().replace(/:/g, "");
  const colour = TONE[tone];

  return (
    <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden" aria-hidden>
      {/* A single wash of colour per section, so the page moves through a palette as you
          scroll instead of sitting on one flat black. */}
      <div
        className="absolute inset-0"
        style={{
          background: `radial-gradient(70% 55% at 78% 12%, color-mix(in srgb, ${colour} ${
            7 * intensity
          }%, transparent) 0%, transparent 68%)`,
        }}
      />

      {variant === "ledger" && (
        <svg
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="none"
          viewBox="0 0 1000 700"
        >
          <defs>
            <linearGradient id={`${uid}-f`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colour} stopOpacity="0" />
              <stop offset="45%" stopColor={colour} stopOpacity={0.2 * intensity} />
              <stop offset="100%" stopColor={colour} stopOpacity="0" />
            </linearGradient>
          </defs>
          {/* Ruled lines converging to a vanishing point: the ledger, in perspective. */}
          {Array.from({ length: 26 }, (_, i) => {
            const t = i / 25;
            const y = 700 - Math.pow(1 - t, 2.1) * 700;
            return (
              <line
                key={i}
                x1="-100"
                x2="1100"
                y1={y}
                y2={y}
                stroke={`url(#${uid}-f)`}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          {Array.from({ length: 19 }, (_, i) => {
            const x = (i / 18) * 1000;
            return (
              <line
                key={`v${i}`}
                x1={500 + (x - 500) * 0.12}
                x2={x}
                y1="0"
                y2="700"
                stroke={`url(#${uid}-f)`}
                strokeWidth="1"
                opacity="0.5"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      )}

      {variant === "orbit" && (
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1000 700">
          <g className="drift-slow" style={{ transformOrigin: "760px 220px" }}>
            {Array.from({ length: 9 }, (_, i) => (
              <ellipse
                key={i}
                cx="760"
                cy="220"
                rx={70 + i * 62}
                ry={(70 + i * 62) * 0.34}
                fill="none"
                stroke={colour}
                strokeOpacity={(0.13 - i * 0.012) * intensity}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
          <g className="drift-slow-reverse" style={{ transformOrigin: "180px 560px" }}>
            {Array.from({ length: 6 }, (_, i) => (
              <ellipse
                key={i}
                cx="180"
                cy="560"
                rx={50 + i * 52}
                ry={(50 + i * 52) * 0.3}
                fill="none"
                stroke={colour}
                strokeOpacity={(0.1 - i * 0.013) * intensity}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
        </svg>
      )}

      {variant === "stream" && (
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1000 700">
          <defs>
            <linearGradient id={`${uid}-s`} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={colour} stopOpacity="0" />
              <stop offset="50%" stopColor={colour} stopOpacity={0.28 * intensity} />
              <stop offset="100%" stopColor={colour} stopOpacity="0" />
            </linearGradient>
          </defs>
          {Array.from({ length: 14 }, (_, i) => {
            const y = 40 + i * 48;
            const amp = 12 + (i % 4) * 9;
            return (
              <path
                key={i}
                className="streak"
                style={{ animationDelay: `${-i * 1.7}s`, animationDuration: `${19 + (i % 5) * 4}s` }}
                d={`M -300 ${y} C 100 ${y - amp}, 400 ${y + amp}, 700 ${y - amp * 0.5} S 1100 ${y}, 1400 ${y}`}
                fill="none"
                stroke={`url(#${uid}-s)`}
                strokeWidth={i % 3 === 0 ? 1.6 : 1}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      )}

      {variant === "grain" && (
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1000 700">
          {/* A deterministic scatter — the same field on every load, so a screenshot taken
              for the pitch matches the page a judge opens. */}
          {Array.from({ length: 110 }, (_, i) => {
            const a = Math.sin(i * 12.9898) * 43758.5453;
            const b = Math.sin(i * 78.233) * 12345.6789;
            const x = (a - Math.floor(a)) * 1000;
            const y = (b - Math.floor(b)) * 700;
            const r = 0.7 + ((i * 7) % 5) * 0.32;
            return (
              <circle
                key={i}
                className="twinkle"
                style={{ animationDelay: `${-(i % 17) * 0.7}s` }}
                cx={x}
                cy={y}
                r={r}
                fill={colour}
                opacity={(0.1 + ((i * 13) % 7) * 0.025) * intensity}
              />
            );
          })}
        </svg>
      )}

      {/* Every backdrop fades at the top and bottom so it never fights the section seam. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(to bottom, var(--color-ink) 0%, transparent 22%, transparent 78%, var(--color-ink) 100%)",
        }}
      />
    </div>
  );
}
