/**
 * Money formatting. Everything on the wire is integer paise, exactly as it is in the
 * Python — the front end never sees a float rupee value and never does arithmetic on one.
 *
 * The Indian grouping (2,2,3) is not what `Intl` gives you by default for `en-US`, and a
 * receivables page that renders Rs 1,220,331 instead of Rs 12,20,331 reads as foreign to
 * every person it is built for. `en-IN` handles it.
 */

const INR = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const INR2 = new Intl.NumberFormat("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export const rupees = (paise: number) => paise / 100;

/** Rs 12,20,331 — full precision, for tables and anywhere a figure is quoted. */
export function inr(paise: number | null | undefined): string {
  if (paise == null) return "—";
  return "₹" + INR.format(Math.round(paise / 100));
}

/** Rs 12,20,331.56 — where the paise themselves are the point. */
export function inrExact(paise: number | null | undefined): string {
  if (paise == null) return "—";
  return "₹" + INR2.format(paise / 100);
}

/**
 * Rs 1.22 Cr — for headlines only.
 *
 * Compact forms are lossy, so they are used where the shape of the number matters and the
 * exact figure is one scroll away, never as the only place a number appears.
 */
export function inrCompact(paise: number | null | undefined): string {
  if (paise == null) return "—";
  const r = Math.abs(paise / 100);
  const sign = paise < 0 ? "-" : "";
  if (r >= 1e7) return `${sign}₹${(r / 1e7).toFixed(2)} Cr`;
  if (r >= 1e5) return `${sign}₹${(r / 1e5).toFixed(2)} L`;
  if (r >= 1e3) return `${sign}₹${(r / 1e3).toFixed(1)}k`;
  return `${sign}₹${INR.format(Math.round(r))}`;
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function bp(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value / 100).toFixed(2)}%`;
}

export const num = (value: number | null | undefined) =>
  value == null ? "—" : INR.format(value);

/** `resolved_closed_valid` -> `resolved closed valid`. Machine keys are not display copy. */
export const humanise = (key: string) => key.replace(/_/g, " ");

export const titleCase = (key: string) =>
  humanise(key).replace(/\b\w/g, (c) => c.toUpperCase());

/**
 * The semantic palette, in one place.
 *
 * Colour carries meaning on this site rather than decorating it, so every chart, badge and
 * particle resolves its colour through here. A jade bar and a jade particle are jade for the
 * same reason.
 */
export const TONE = {
  gold: "var(--color-gold)",
  leak: "var(--color-leak)",
  jade: "var(--color-jade)",
  slate: "var(--color-slate)",
  violet: "var(--color-violet)",
  muted: "var(--color-paper-3)",
} as const;

export type Tone = keyof typeof TONE;

export const FAMILY_TONE: Record<string, Tone> = {
  tds: "slate",
  gst: "slate",
  credit_note: "gold",
  scheme: "gold",
  contract: "violet",
  goods: "leak",
  duplicate: "leak",
  unexplained: "leak",
  deminimis: "muted",
  freight: "violet",
};

export const VERDICT_TONE: Record<string, Tone> = {
  valid: "slate",
  provisional_valid: "slate",
  invalid: "leak",
  partially_valid: "gold",
  unknown: "muted",
};

export const STATE_TONE: Record<string, Tone> = {
  resolved_recovered: "jade",
  resolved_closed_valid: "slate",
  resolved_written_off: "muted",
  human_queue: "violet",
  stopped: "gold",
  disputed: "leak",
  awaiting_settlement: "muted",
  new: "muted",
};
