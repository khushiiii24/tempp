import { lazy, Suspense } from "react";
import { agent, leak } from "../lib/data";
import { BRAND } from "../lib/brand";
import { inr, inrCompact, pct, TONE } from "../lib/format";

const LeakScene = lazy(() => import("./LeakScene"));

/**
 * Maximum density, once. Everything below this is quieter, and the section immediately
 * after is the quietest thing on the page — that contrast does more work than any single
 * element.
 *
 * The three shares are computed here and handed to the 3D scene, so the legend and the
 * particles read the same numbers by construction and cannot drift apart.
 */
export default function Hero() {
  const valid = leak.valid_paise;
  const addressed = Number(agent.money.addressed_paise ?? 0);
  const rest = Math.max(0, leak.short_paid_paise - valid - addressed);
  const total = leak.short_paid_paise || 1;

  const forks = [
    {
      tone: TONE.slate,
      label: "theirs by right",
      paise: valid,
      note: "Tax they were required to withhold, rebates we agreed to, credit notes we already issued.",
    },
    {
      tone: TONE.jade,
      label: "brought back",
      paise: addressed,
      note: "Either paid back, or sitting on an analyst's desk with the evidence attached.",
    },
    {
      tone: TONE.leak,
      label: "still out there",
      paise: rest,
      note: "Too small to chase, stuck in dispute, or we ran out of clock at day 45.",
    },
  ];

  return (
    <section id="top" className="relative min-h-[100svh] overflow-hidden">
      <Suspense fallback={null}>
        <LeakScene
          shares={{ valid: valid / total, addressed: addressed / total, lost: rest / total }}
        />
      </Suspense>

      {/* Scrims. The hero never ends with an edge — it darkens on the left so the type
          holds, and dissolves at the base into whatever is next. */}
      <div
        className="pointer-events-none absolute inset-0 z-[2] hidden md:block"
        style={{
          background:
            "linear-gradient(100deg, rgba(5,6,10,0.95) 0%, rgba(5,6,10,0.84) 24%, rgba(5,6,10,0.24) 42%, transparent 54%)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-0 z-[2] md:hidden"
        style={{
          background:
            "linear-gradient(180deg, rgba(5,6,10,0.96) 0%, rgba(5,6,10,0.92) 44%, rgba(5,6,10,0.5) 66%, rgba(5,6,10,0.18) 84%, transparent 100%)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "radial-gradient(46% 52% at 66% 46%, rgba(232,178,76,0.15) 0%, transparent 70%)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 z-[2] h-40"
        style={{ background: "linear-gradient(to bottom, transparent, var(--color-ink) 96%)" }}
      />

      <div className="shell relative z-[3] flex min-h-[100svh] flex-col justify-between pb-10 pt-32 md:pt-40">
        <div className="max-w-[47rem]">
          <h1 className="display text-[clamp(2.35rem,5.6vw,4.35rem)] font-extrabold text-paper text-balance">
            Every invoice was paid.
            <br />
            <span style={{ color: TONE.gold }}>Not every rupee arrived.</span>
          </h1>

          <p className="mt-7 max-w-[54ch] text-[17.5px] leading-[1.7] text-paper-2 md:text-[20px]">
            We sent {leak.invoices} invoices worth {inrCompact(leak.invoice_value_paise)}.{" "}
            <span className="mono text-paper">{inr(leak.short_paid_paise)}</span> of that never came
            back. It left as {leak.deductions} separate deductions, most of them explained in one
            line of Hinglish attached to a bank credit.
          </p>
          <p className="mt-4 max-w-[54ch] text-[17.5px] leading-[1.7] text-paper-2 md:text-[20px]">
            About half of it the buyer was entitled to keep.{" "}
            <strong className="font-medium text-paper">{BRAND.name}</strong> works out which half,
            then writes only to the rest.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <a
              href="#leak"
              className="mono group inline-flex items-center gap-3 rounded-full px-7 py-4 text-[12px] uppercase tracking-[0.18em] text-ink transition-transform duration-300 hover:-translate-y-0.5"
              style={{ background: TONE.gold }}
            >
              See the leak
              <span className="transition-transform duration-300 group-hover:translate-x-1">→</span>
            </a>
            <a
              href="#scoreboard"
              className="mono inline-flex items-center gap-2 rounded-full border border-rule-2 px-7 py-4 text-[12px] uppercase tracking-[0.18em] text-paper-2 transition-colors duration-300 hover:border-paper-3 hover:text-paper"
            >
              Straight to the numbers
            </a>
          </div>
        </div>

        <div className="mt-14">
          <div className="mono mb-5 text-[10px] uppercase tracking-[0.24em] text-paper-3">
            where {inr(leak.short_paid_paise)} ended up
          </div>
          <div className="grid gap-x-8 gap-y-6 border-t border-rule pt-6 sm:grid-cols-3">
            {forks.map((f) => (
              <div key={f.label} className="group">
                <div className="flex items-baseline gap-3">
                  <span
                    className="mono text-[clamp(1.5rem,2.4vw,2rem)] font-semibold leading-none"
                    style={{ color: f.tone }}
                  >
                    {pct(f.paise / total)}
                  </span>
                  <span className="mono text-[12.5px] text-paper-2">{inrCompact(f.paise)}</span>
                </div>
                <div className="mt-2.5 flex items-center gap-2.5">
                  <span
                    className="h-[2px] w-5 rounded-full"
                    style={{ background: f.tone }}
                    aria-hidden
                  />
                  <span className="mono text-[10.5px] uppercase tracking-[0.16em] text-paper">
                    {f.label}
                  </span>
                </div>
                <p className="mt-2 max-w-[38ch] text-[13.5px] leading-[1.62] text-paper-3">
                  {f.note}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
