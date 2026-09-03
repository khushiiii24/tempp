import { BRAND } from "../lib/brand";

/**
 * The lockup: a mark, then the name in plain type.
 *
 * The previous wordmark drew every letter as faux-Devanagari, and it was unreadable —
 * under a continuous top rule, letters built from vertical stems collapse into each other,
 * and the word came out as KATAUM. Distinctiveness bought at the cost of the reader being
 * able to read the name is a bad trade for a logo.
 *
 * So the unusual part is the mark and the name is just set, in the same display face as
 * every heading on the page. The mark is a return arrow closing most of a circle around a
 * single coin: money going out and coming back, which is the entire product. It reads at
 * 16px and it reads as a favicon.
 */
export default function Wordmark({
  className = "",
  /** Hide the name and show only the mark — for tight spaces and the favicon. */
  markOnly = false,
  /** Outline treatment for the oversized footer lockup. */
  outline = false,
}: {
  className?: string;
  markOnly?: boolean;
  outline?: boolean;
}) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <Mark className="h-[1.05em] w-[1.05em] shrink-0" outline={outline} />
      {!markOnly && (
        <span
          className="display font-extrabold tracking-[-0.035em]"
          style={
            outline
              ? // `-webkit-text-fill-color`, not `color`. Setting `color: transparent` also
                // makes `currentColor` transparent, so the stroke painted itself in
                // nothing and the name vanished — leaving the mark alone in the footer
                // with no word beside it.
                {
                  WebkitTextFillColor: "transparent",
                  WebkitTextStroke: "1.5px currentColor",
                }
              : undefined
          }
        >
          {BRAND.name}
        </span>
      )}
    </span>
  );
}

export function Mark({
  className = "",
  outline = false,
}: {
  className?: string;
  outline?: boolean;
}) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden focusable="false">
      {/* Three-quarters of a circle, open at the top — the money's round trip, with the gap
          where it went missing. The arc starts at -70° and sweeps 270° clockwise. */}
      <path
        d="M14.74 4.48 A8 8 0 1 1 4.48 9.26"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {/* The arrowhead, sitting on the arc's start and pointing along its tangent. The
          first version was an axis-aligned triangle dropped near the endpoint, and at logo
          size it read as detached — an arrowhead has to ride the curve it belongs to. */}
      <path d="M18.69 5.91 L13.72 7.30 L15.76 1.66 Z" fill="currentColor" />
      {/* The coin at the centre. Hollow in the outline treatment so the footer lockup
          stays a line drawing all the way through. */}
      <circle
        cx="12"
        cy="12"
        r="3"
        fill={outline ? "none" : "currentColor"}
        stroke={outline ? "currentColor" : "none"}
        strokeWidth="1.8"
      />
    </svg>
  );
}
