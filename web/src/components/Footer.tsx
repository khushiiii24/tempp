import { useState } from "react";
import { BRAND } from "../lib/brand";
import Wordmark from "./Wordmark";
import { agent, leak, meta } from "../lib/data";
import { inr, num, TONE } from "../lib/format";

const QUICKSTART = [
  "python -m venv .venv",
  ".venv/Scripts/pip install -e .",
  "python -m deduction_desk generate --seed 42 --n 400",
  "python -m deduction_desk match",
  "python -m deduction_desk classify --offline",
  "python -m deduction_desk report --compare agent,b0,b1,b2,b3",
  "python -m deduction_desk export-web",
];

/**
 * The fade. Closes on the same ruled paper the page opened on.
 *
 * The one thing here that matters is the middle column — the exact commands that rebuild
 * every number above, offline, with no key and no GPU. A results page that cannot be
 * re-run is a screenshot.
 */
export default function Footer() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(QUICKSTART.join("\n"));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <footer className="seam relative z-10 overflow-hidden pt-28">
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(90% 120% at 50% 120%, rgba(232,178,76,0.12) 0%, transparent 62%)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[420px] opacity-60"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, transparent 0 23px, rgba(233,230,222,0.05) 23px 24px)",
          maskImage: "linear-gradient(to bottom, transparent, #000 70%)",
        }}
      />

      <div className="shell relative">
        <div className="grid gap-12 border-t border-rule pt-14 md:grid-cols-3 md:gap-10">
          <div>
            <Wordmark className="text-[32px] text-paper" />
            <p className="mt-3 max-w-[36ch] text-[13.5px] leading-[1.75] text-paper-2">
              Indian buyers pay invoices net of deductions.{" "}
              <span className="text-paper">{BRAND.name}</span> works out how much of that you are
              actually owed back, and goes after only that.
            </p>
            <dl className="mono mt-6 space-y-1.5 text-[11px]">
              {[
                ["batch", meta.db_content_hash.slice(0, 16)],
                ["seed", String(meta.seed)],
                ["clock", `${meta.days} days`],
                ["cases", num(meta.cases)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-4 border-b border-rule/60 py-1">
                  <dt className="text-paper-3">{k}</dt>
                  <dd className="text-paper-2">{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div>
            <div className="mb-4 flex items-center justify-between gap-4">
              <h3 className="mono text-[10px] uppercase tracking-[0.24em] text-paper-3">
                rebuild every number above
              </h3>
              {/* Above the block, not floating on top of it. Overlaying the button meant
                  reserving 80px of right padding, which pushed the longest command into a
                  horizontal scrollbar. */}
              <button
                onClick={copy}
                className="mono shrink-0 rounded-full border border-rule px-2.5 py-1 text-[9.5px] uppercase tracking-[0.14em] text-paper-3 transition-colors hover:border-rule-2 hover:text-paper"
              >
                {copied ? "copied" : "copy"}
              </button>
            </div>
            <p className="mb-4 max-w-[38ch] text-[13px] leading-[1.7] text-paper-3">
              The model runs on your own machine and its cache is in the repo, so{" "}
              <span className="mono text-paper-2">--offline</span> rebuilds all of this with no API
              key, no GPU and no network. On a cache miss it stops instead of quietly calling
              out.
            </p>
            <pre className="mono overflow-x-auto rounded-xl border border-rule bg-ink-2 px-4 py-3.5 text-[10px] leading-[1.95] text-paper-2">
              {QUICKSTART.join("\n")}
            </pre>
          </div>

          <div>
            <h3 className="mono mb-4 text-[10px] uppercase tracking-[0.24em] text-paper-3">
              in one line
            </h3>
            <div className="space-y-5">
              {[
                { k: "went missing", v: inr(leak.short_paid_paise), t: TONE.gold },
                { k: "handled", v: inr(Number(agent.money.addressed_paise)), t: TONE.jade },
                {
                  k: "letters to customers who owed nothing",
                  v: num(Number(agent.harm.false_chase_contacts)),
                  t: TONE.leak,
                },
                {
                  k: "compliance violations",
                  v: num(Number(agent.compliance?.violations ?? 0)),
                  t: TONE.jade,
                },
              ].map((r) => (
                <div key={r.k}>
                  <div
                    className="mono text-[1.45rem] font-semibold leading-none"
                    style={{ color: r.t }}
                  >
                    {r.v}
                  </div>
                  <div className="mt-1.5 text-[12.5px] text-paper-3">{r.k}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* The sign-off. The wordmark used to sit here alone at 15vw with nothing under it
            and a screen of empty space around it, which read as a page that had run out
            rather than one that had finished. It closes on the headline it opened with. */}
        <div className="relative mt-24 border-t border-rule pt-16 text-center">
          <div
            className="pointer-events-none absolute inset-x-0 top-0 h-px"
            style={{
              background:
                "linear-gradient(to right, transparent, color-mix(in srgb, var(--color-gold) 55%, transparent), transparent)",
            }}
          />

          <Wordmark
            outline
            className="justify-center text-[clamp(2.6rem,8.5vw,6.5rem)] text-paper/40"
          />

          <p className="mt-7 text-[15px] leading-[1.65] text-paper-2 md:text-[17.5px]">
            Every invoice was paid.
            <br />
            <span style={{ color: TONE.gold }}>Not every rupee arrived.</span>
          </p>

          <div className="mono mt-8 flex flex-wrap items-center justify-center gap-x-3 gap-y-2 text-[10.5px] uppercase tracking-[0.18em] text-paper-3">
            <span>seed {meta.seed}</span>
            <span className="text-rule-2">/</span>
            <span>{meta.days}-day clock</span>
            <span className="text-rule-2">/</span>
            <span>{num(meta.cases)} cases</span>
            <span className="text-rule-2">/</span>
            <span style={{ color: TONE.jade }}>0 messages sent</span>
          </div>

          <a
            href="#top"
            className="mono mt-10 inline-flex items-center gap-2.5 rounded-full border border-rule-2 px-5 py-2.5 text-[10.5px] uppercase tracking-[0.18em] text-paper-2 transition-colors hover:border-paper-3 hover:text-paper"
          >
            <span aria-hidden>↑</span> back to the top
          </a>
        </div>

        <div className="mono mt-16 flex flex-wrap items-center justify-between gap-4 border-t border-rule py-6 text-[10.5px] text-paper-3">
          <span>
            Synthetic data throughout. No real buyer, invoice, GSTIN or PAN appears anywhere.
          </span>
          <span>
            {num(Number(agent.operations.messages_queued ?? 0))} messages queued · 0 sent
          </span>
        </div>
      </div>
    </footer>
  );
}
