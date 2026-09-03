import { leak } from "../lib/data";
import { inrCompact, TONE } from "../lib/format";

/**
 * The seam between the hero and the first section.
 *
 * Section boundaries on this page are designed objects rather than a background swap, and
 * this one is a reel of the actual reason codes with their actual share of the leak —
 * gold if the code is chaseable, slate if it never was. It does the work of a taxonomy
 * table without asking anyone to read a taxonomy table.
 */
export default function CodeTicker() {
  const codes = leak.by_code.slice(0, 14);

  return (
    <div className="relative z-10 overflow-hidden border-y border-rule bg-ink-2/50 py-3">
      <div
        className="pointer-events-none absolute inset-y-0 left-0 z-10 w-24"
        style={{ background: "linear-gradient(to right, var(--color-ink), transparent)" }}
      />
      <div
        className="pointer-events-none absolute inset-y-0 right-0 z-10 w-24"
        style={{ background: "linear-gradient(to left, var(--color-ink), transparent)" }}
      />
      <div className="ticker-track flex w-max gap-10">
        {[0, 1].map((copy) => (
          <div key={copy} className="flex gap-10" aria-hidden={copy === 1}>
            {codes.map((c) => (
              <span
                key={c.code}
                className="mono flex items-center gap-2.5 whitespace-nowrap text-[11px]"
              >
                <span
                  className="inline-block h-1.5 w-1.5 rounded-full"
                  style={{ background: c.chaseable ? TONE.gold : TONE.slate }}
                />
                <span className="text-paper-2">{c.label}</span>
                <span style={{ color: c.chaseable ? TONE.gold : TONE.slate }}>
                  {inrCompact(c.paise)}
                </span>
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
