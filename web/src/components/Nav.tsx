import { useEffect, useState } from "react";
import { BRAND } from "../lib/brand";
import Wordmark from "./Wordmark";
import { useActiveSection } from "../lib/hooks";

const LINKS = [
  { id: "leak", label: "the leak" },
  { id: "pipeline", label: "how it works" },
  { id: "cases", label: "cases" },
  { id: "scoreboard", label: "results" },
  { id: "guardrails", label: "guardrails" },
];

/**
 * Lowercase anchors floating over the hero rather than sitting in an opaque bar — a small
 * informality against otherwise severe typography, and it keeps the hero composition whole
 * instead of slicing a strip off the top of it.
 */
export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const active = useActiveSection(LINKS.map((l) => l.id));

  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 40);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      <a
        href="#leak"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[120] focus:rounded-lg focus:bg-ink-3 focus:px-4 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <header
        className="fixed inset-x-0 top-0 z-[70] transition-all duration-500"
        style={{
          background: scrolled ? "rgba(5,6,10,0.78)" : "transparent",
          backdropFilter: scrolled ? "blur(16px) saturate(150%)" : "none",
          borderBottom: `1px solid ${scrolled ? "var(--color-rule)" : "transparent"}`,
        }}
      >
        <nav className="shell flex h-16 items-center justify-between md:h-[78px]">
          <a href="#top" className="flex items-center text-paper" aria-label={BRAND.name}>
            <Wordmark className="text-[21px]" />
          </a>

          <ul className="hidden items-center gap-1 md:flex">
            {LINKS.map((l) => {
              const on = active === l.id;
              return (
                <li key={l.id}>
                  <a
                    href={`#${l.id}`}
                    className="mono relative rounded-full px-3.5 py-2 text-[11.5px] tracking-[0.04em] transition-colors"
                    style={{
                      color: on ? "var(--color-gold)" : "var(--color-paper-2)",
                      background: on ? "color-mix(in srgb, var(--color-gold) 10%, transparent)" : "transparent",
                    }}
                  >
                    {l.label}
                  </a>
                </li>
              );
            })}
          </ul>

          <button
            className="mono rounded-full border border-rule-2 px-4 py-2 text-[10.5px] uppercase tracking-[0.18em] text-paper-2 md:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "close" : "menu"}
          </button>
        </nav>
      </header>

      {/* A separate drawer, not a squashed bar — six anchors do not survive being scaled
          down to 360px. */}
      <div
        id="mobile-nav"
        className="fixed inset-0 z-[69] flex flex-col justify-center bg-ink px-8 transition-all duration-300 md:hidden"
        style={{
          opacity: open ? 1 : 0,
          pointerEvents: open ? "auto" : "none",
          transform: open ? "none" : "translateY(-8px)",
        }}
      >
        <div className="rules" />
        <ul className="relative space-y-1">
          {LINKS.map((l) => (
            <li key={l.id}>
              <a
                href={`#${l.id}`}
                onClick={() => setOpen(false)}
                className="flex items-baseline gap-4 border-b border-rule py-4"
              >
                <span className="display text-3xl font-bold text-paper">{l.label}</span>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}
