import { useId, useMemo, useState } from "react";
import { inrCompact, TONE, type Tone } from "../lib/format";
import { useInView, useReducedMotion } from "../lib/hooks";
import { TooltipCard, useTooltip } from "./Tooltip";

export interface Series {
  name: string;
  tone: Tone;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

/**
 * A hand-rolled SVG line chart rather than a charting library.
 *
 * A library brings its own grid colours, tooltip chrome and default font, and the argument
 * of this page is that one theme drives every layer. It is also a hundred lines; importing
 * 200 KB to draw a polyline is not a trade worth making.
 *
 * The line draws itself on first view, and every day on it is hoverable: move across the
 * plot and you get that day's figure for both series, with a marker on each line. Reading
 * a shape is not the same as reading a number.
 */
export default function Curve({
  series,
  height = 260,
  yFormat = inrCompact,
  xLabel = "day",
  yTicks = 4,
}: {
  series: Series[];
  height?: number;
  yFormat?: (v: number) => string;
  xLabel?: string;
  yTicks?: number;
}) {
  const uid = useId().replace(/:/g, "");
  const reduced = useReducedMotion();
  const { ref: viewRef, inView } = useInView<HTMLDivElement>({ threshold: 0.25 });
  const { ref: hostRef, tip, show, hide } = useTooltip<number>();
  const [hoverX, setHoverX] = useState<number | null>(null);

  const W = 900;
  const H = height;
  const P = { t: 16, r: 14, b: 28, l: 62 };

  const all = series.flatMap((s) => s.points);
  const xMax = Math.max(...all.map((p) => p.x), 1);
  const yMax = Math.max(...all.map((p) => p.y), 1);

  const sx = (x: number) => P.l + (x / xMax) * (W - P.l - P.r);
  const sy = (y: number) => H - P.b - (y / yMax) * (H - P.t - P.b);

  const paths = useMemo(
    () =>
      series.map((s) => ({
        ...s,
        d: s.points
          .map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
          .join(" "),
        len: s.points.reduce((acc, p, i) => {
          if (!i) return 0;
          const a = s.points[i - 1];
          return acc + Math.hypot(sx(p.x) - sx(a.x), sy(p.y) - sy(a.y));
        }, 0),
        last: s.points[s.points.length - 1],
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, W, H, xMax, yMax],
  );

  if (!all.length) return null;
  const animate = inView && !reduced;
  const ticks = Array.from({ length: yTicks + 1 }, (_, i) => (yMax / yTicks) * i);

  /** Pointer position → the nearest day, in data space. */
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    const day = Math.round(Math.max(0, Math.min(1, frac)) * xMax);
    setHoverX(day);
    show(e, day);
  };
  const onLeave = () => {
    setHoverX(null);
    hide();
  };

  const at = (s: (typeof paths)[number], day: number) =>
    s.points.reduce((best, p) =>
      Math.abs(p.x - day) < Math.abs(best.x - day) ? p : best,
    );

  return (
    <div className="relative w-full" ref={hostRef}>
      <div ref={viewRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full overflow-visible"
          role="img"
          aria-label={paths
            .map((s) => `${s.name}, ending at ${yFormat(s.last?.y ?? 0)}`)
            .join("; ")}
        >
          <defs>
            {paths.map((s) => (
              <linearGradient
                key={s.name}
                id={`${uid}-${s.name.replace(/\W/g, "")}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={TONE[s.tone]} stopOpacity="0.26" />
                <stop offset="100%" stopColor={TONE[s.tone]} stopOpacity="0" />
              </linearGradient>
            ))}
          </defs>

          {ticks.map((t, i) => (
            <g key={t}>
              <line
                x1={P.l}
                x2={W - P.r}
                y1={sy(t)}
                y2={sy(t)}
                stroke="rgba(233,230,222,0.075)"
                strokeDasharray={i === 0 ? undefined : "3 5"}
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={P.l - 10}
                y={sy(t) + 3.5}
                textAnchor="end"
                className="mono"
                fontSize="10.5"
                fill="var(--color-paper-3)"
              >
                {yFormat(t)}
              </text>
            </g>
          ))}

          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text
              key={f}
              x={sx(xMax * f)}
              y={H - 9}
              textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}
              className="mono"
              fontSize="10.5"
              fill="var(--color-paper-3)"
            >
              {f === 1 ? `${xLabel} ${Math.round(xMax)}` : Math.round(xMax * f)}
            </text>
          ))}

          {paths.map((s, si) => (
            <g key={s.name}>
              {!s.dashed && (
                <path
                  d={`${s.d} L${sx(s.last.x)},${H - P.b} L${sx(s.points[0].x)},${H - P.b} Z`}
                  fill={`url(#${uid}-${s.name.replace(/\W/g, "")})`}
                  className={animate ? "fade-up" : undefined}
                  style={animate ? { animationDelay: `${700 + si * 150}ms` } : undefined}
                />
              )}
              <path
                key={`${s.name}-${animate}`}
                d={s.d}
                fill="none"
                stroke={TONE[s.tone]}
                strokeWidth="2.25"
                strokeDasharray={s.dashed ? "5 5" : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
                className={animate && !s.dashed ? "draw-line" : undefined}
                style={
                  animate && !s.dashed
                    ? ({ "--len": s.len, animationDelay: `${si * 180}ms` } as React.CSSProperties)
                    : undefined
                }
              />
              <circle
                cx={sx(s.last.x)}
                cy={sy(s.last.y)}
                r="4"
                fill={TONE[s.tone]}
                className={animate ? "pop-in" : undefined}
                style={animate ? { animationDelay: `${1500 + si * 180}ms` } : undefined}
              />
            </g>
          ))}

          {/* Hover guide and per-series markers. */}
          {hoverX != null && (
            <g pointerEvents="none">
              <line
                x1={sx(hoverX)}
                x2={sx(hoverX)}
                y1={P.t}
                y2={H - P.b}
                stroke="var(--color-paper-3)"
                strokeOpacity="0.45"
                strokeDasharray="3 4"
                vectorEffect="non-scaling-stroke"
              />
              {paths.map((s) => {
                const p = at(s, hoverX);
                return (
                  <g key={s.name}>
                    <circle cx={sx(p.x)} cy={sy(p.y)} r="7" fill={TONE[s.tone]} opacity="0.22" />
                    <circle
                      cx={sx(p.x)}
                      cy={sy(p.y)}
                      r="4"
                      fill="var(--color-ink)"
                      stroke={TONE[s.tone]}
                      strokeWidth="2.5"
                      vectorEffect="non-scaling-stroke"
                    />
                  </g>
                );
              })}
            </g>
          )}

          {/* One transparent hit area over the plot: cheaper and steadier than a hit target
              per point, and it means the readout follows the pointer everywhere rather than
              only when it lands on a 4px dot. */}
          <rect
            x={P.l}
            y={P.t}
            width={W - P.l - P.r}
            height={H - P.t - P.b}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={onLeave}
            style={{ cursor: "crosshair" }}
          />
        </svg>
      </div>

      {tip && (
        <TooltipCard x={tip.x} y={tip.y} width={224}>
          <div className="mono text-[9.5px] uppercase tracking-[0.18em] text-paper-3">
            {xLabel} {tip.data}
          </div>
          <dl className="mt-2.5 space-y-1.5">
            {paths.map((s) => {
              const p = at(s, tip.data);
              return (
                <div key={s.name} className="flex items-baseline justify-between gap-3">
                  <dt className="flex items-center gap-2 text-[11.5px] text-paper-2">
                    <span
                      className="inline-block h-[3px] w-3.5 rounded-full"
                      style={{ background: TONE[s.tone], opacity: s.dashed ? 0.6 : 1 }}
                    />
                    {s.name}
                  </dt>
                  <dd className="mono text-[12px]" style={{ color: TONE[s.tone] }}>
                    {yFormat(p.y)}
                  </dd>
                </div>
              );
            })}
          </dl>
          <p className="mt-2.5 border-t border-rule pt-2 text-[11px] leading-[1.5] text-paper-3">
            Cash banked by the end of {xLabel} {tip.data}, running total.
          </p>
        </TooltipCard>
      )}

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2">
        {series.map((s) => (
          <span key={s.name} className="mono flex items-center gap-2.5 text-[12px] text-paper-2">
            <span
              className="inline-block h-[3px] w-6 rounded-full"
              style={{ background: TONE[s.tone], opacity: s.dashed ? 0.6 : 1 }}
            />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
