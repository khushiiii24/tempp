import { useRef, useState, type ReactNode } from "react";

/**
 * Hover readouts for charts.
 *
 * Every chart on this page plots a real, specific figure, and until now you could see the
 * shape but not the number — a bar was 62% of the width of another bar and that was all you
 * got. This puts the value, the label and what it means under the pointer.
 *
 * Positioned inside the chart's own container rather than the document, so it scrolls with
 * the chart and needs no portal. It flips near the right and bottom edges instead of being
 * clipped.
 */
export interface TipState<T> {
  x: number;
  y: number;
  data: T;
}

export function useTooltip<T, E extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<E | null>(null);
  const [tip, setTip] = useState<TipState<T> | null>(null);

  const show = (e: { clientX: number; clientY: number }, data: T) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, data });
  };

  const hide = () => setTip(null);

  return { ref, tip, show, hide };
}

export function TooltipCard({
  x,
  y,
  width = 232,
  children,
}: {
  x: number;
  y: number;
  width?: number;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const host = ref.current?.parentElement;
  const hostW = host?.clientWidth ?? 0;
  const hostH = host?.clientHeight ?? 0;

  // Flip rather than clip. On the first render the parent size is unknown, which puts the
  // card in its default place for one frame — invisible at 60fps and preferable to
  // measuring in a layout effect for a hover readout.
  const flipX = hostW > 0 && x + width + 26 > hostW;
  const flipY = hostH > 0 && y + 150 > hostH;

  return (
    <div
      ref={ref}
      className="pointer-events-none absolute z-30"
      style={{
        left: flipX ? undefined : x + 16,
        right: flipX ? Math.max(8, hostW - x + 16) : undefined,
        top: flipY ? undefined : y + 14,
        bottom: flipY ? Math.max(8, hostH - y + 14) : undefined,
        width,
      }}
    >
      <div className="rounded-xl border border-rule-2 bg-ink-2/95 px-3.5 py-3 shadow-2xl backdrop-blur-md">
        {children}
      </div>
    </div>
  );
}

/** The standard shape of a readout: a title, a big number, and a line saying what it is. */
export function TipBody({
  label,
  value,
  tone,
  rows,
  note,
}: {
  label: string;
  value?: ReactNode;
  tone?: string;
  rows?: { k: string; v: ReactNode }[];
  note?: ReactNode;
}) {
  return (
    <>
      <div className="mono text-[9.5px] uppercase tracking-[0.18em] text-paper-3">{label}</div>
      {value != null && (
        <div className="mono mt-1.5 text-[17px] font-semibold leading-none" style={{ color: tone }}>
          {value}
        </div>
      )}
      {rows?.length ? (
        <dl className="mt-2.5 space-y-1">
          {rows.map((r) => (
            <div key={r.k} className="flex items-baseline justify-between gap-3">
              <dt className="text-[11.5px] text-paper-3">{r.k}</dt>
              <dd className="mono text-[11.5px] text-paper-2">{r.v}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {note ? (
        <p className="mt-2.5 border-t border-rule pt-2 text-[11.5px] leading-[1.5] text-paper-3">
          {note}
        </p>
      ) : null}
    </>
  );
}
