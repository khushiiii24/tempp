import { BRAND } from "../lib/brand";
import { leak } from "../lib/data";
import { FAMILY_TONE, humanise, inr, inrCompact, pct } from "../lib/format";
import { TipBody, TooltipCard, useTooltip } from "./Tooltip";
import { Bar, Note, Panel, Reveal, Section, SectionHead, Stat } from "./ui";

/**
 * The quiet beat.
 *
 * The hero was maximum density; this is a full stop. One heading, prose, one chart, three
 * real documents. The numbers live inside the sentences rather than in a row of animated
 * counters — a wall of big numbers reads as a template, a paragraph that happens to contain
 * them reads as someone who knows the domain.
 */
/** Plain-English gloss per family, for the hover readout. */
const FAMILY_NOTE: Record<string, string> = {
  tds: "Income tax the buyer is legally required to withhold and deposit with the government.",
  gst: "GST TDS, deducted by government and PSU buyers at 2%.",
  credit_note: "The buyer netting off a credit note against the invoice.",
  scheme: "Trade or quarterly purchase scheme rebates, claimed at payment time.",
  contract: "Terms disputes: freight, early-payment discount, or the rate we billed at.",
  goods: "Damage, shortage or quality rejection claimed against the goods receipt.",
  duplicate: "The buyer says they already paid, or raised their own debit note.",
  unexplained: "Short paid with nothing said.",
  deminimis: "Rounding and bank charges. Real, and never worth a phone call.",
};

export default function TheLeak() {
  const families = [...leak.by_family].sort((a, b) => b.paise - a.paise);
  const maxFamily = families[0]?.paise ?? 1;
  const { ref: barsRef, tip, show, hide } = useTooltip<(typeof families)[number]>();

  return (
    <Section id="leak" backdrop="ledger" tone="gold">
      <SectionHead
        kicker="the problem"
        title={
          <>
            A short payment is not
            <br className="hidden md:block" /> a late payment.
          </>
        }
        lede={
          <>
            Indian buyers pay net of deductions. Some withhold tax under 194C or 194J. Some claim
            a rebate from a circular nobody can find. Some net off a credit note, or dock you for a
            short delivery. Whatever the reason, the money has already gone by the time the payment
            lands, and the explanation turns up as one sentence in an email.
          </>
        }
      />

      <div className="grid gap-14 lg:grid-cols-[1.1fr_1fr] lg:gap-16">
        <Reveal>
          <h3 className="mono mb-7 text-[10.5px] uppercase tracking-[0.24em] text-paper-3">
            where it goes
          </h3>
          <div className="relative space-y-5" ref={barsRef}>
            {families.map((f) => {
              const tone = FAMILY_TONE[f.family] ?? "muted";
              const recoverableShare = f.paise ? f.recoverable_paise / f.paise : 0;
              return (
                <div
                  key={f.family}
                  className="cursor-default rounded-md transition-colors hover:bg-paper/[0.03]"
                  onMouseMove={(e) => show(e, f)}
                  onMouseLeave={hide}
                >
                  <div className="mb-2 flex items-baseline justify-between gap-4">
                    <span className="text-[13.5px] text-paper">{humanise(f.family)}</span>
                    <span className="mono text-[12.5px] text-paper-2">
                      {inrCompact(f.paise)}
                      <span className="ml-2.5 text-paper-3">{f.count}</span>
                    </span>
                  </div>
                  <div className="relative">
                    <Bar value={f.paise} max={maxFamily} tone={tone} height={7} />
                    {/* The recoverable slice sits on the same bar, not beside it. The
                        question is always "how much of that is ours". */}
                    <div
                      className="pointer-events-none absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${(f.paise / maxFamily) * recoverableShare * 100}%`,
                        background: "var(--color-jade)",
                        opacity: 0.9,
                      }}
                    />
                  </div>
                  <div className="mono mt-1.5 text-[10.5px] text-paper-3">
                    {pct(recoverableShare, 0)} ours · {f.valid_count} of {f.count} legitimate
                  </div>
                </div>
              );
            })}

            {tip && (
              <TooltipCard x={tip.x} y={tip.y} width={244}>
                <TipBody
                  label={humanise(tip.data.family)}
                  value={inr(tip.data.paise)}
                  tone={`var(--color-${FAMILY_TONE[tip.data.family] ?? "muted"})`}
                  rows={[
                    { k: "deductions", v: tip.data.count },
                    { k: "owed back to us", v: inr(tip.data.recoverable_paise) },
                    { k: "legitimate", v: `${tip.data.valid_count} of ${tip.data.count}` },
                    { k: "share of the gap", v: pct(tip.data.paise / leak.short_paid_paise, 1) },
                  ]}
                  note={FAMILY_NOTE[tip.data.family]}
                />
              </TooltipCard>
            )}
          </div>
          <Note>
            Green is the part we can actually ask for. Notice the shape: <em>tds</em> is the
            biggest bucket and nearly all of it is legitimate, while <em>goods</em> is small and
            nearly all recoverable. Sort the work by rupee value and week one goes to customers who
            did nothing wrong.
          </Note>
        </Reveal>

        <Reveal delay={120}>
          <h3 className="mono mb-7 text-[10.5px] uppercase tracking-[0.24em] text-paper-3">
            what actually lands in the inbox
          </h3>
          <div className="space-y-4">
            {leak.sample_advices.map((a) => (
              <Panel key={a.id} hover className="overflow-hidden">
                <figcaption className="mono flex items-center justify-between border-b border-rule px-4 py-2.5 text-[10px] uppercase tracking-[0.16em] text-paper-3">
                  <span>{a.format.replace("_", " ")}</span>
                  <span>{a.id}</span>
                </figcaption>
                <pre className="mono overflow-x-auto px-4 py-3.5 text-[11px] leading-[1.7] whitespace-pre-wrap text-paper-2">
                  {a.raw_text.trim()}
                </pre>
              </Panel>
            ))}
          </div>
          <Note>
            Straight from the batch, nothing cleaned up. The PDF has lost its columns:{" "}
            <span className="mono text-paper">1973476.9444359.341929117.60</span> is gross,
            deduction and net jammed together, and the only way to split them is knowing the three
            have to add up. The reason sits at the end of the row in four words:{" "}
            <span className="text-paper">gst tds deducted</span>.
          </Note>
        </Reveal>
      </div>

      <Reveal>
        <div className="mt-20 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="the gap"
            tone="gold"
            value={inr(leak.short_paid_paise)}
            sub={`${(leak.leak_rate_bp / 100).toFixed(2)}% of everything billed, across ${leak.deductions} deductions`}
          />
          <Stat
            label="genuinely owed back"
            tone="jade"
            value={pct(leak.recoverable_paise / leak.short_paid_paise, 0)}
            sub={`${inr(leak.recoverable_paise)}. The rest was theirs to keep.`}
          />
          <Stat
            label="of that, collectable"
            tone="slate"
            value={pct(leak.reachable_paise / leak.recoverable_paise, 0)}
            sub="Some buyers owe it and will not pay however nicely you ask."
          />
          <Stat
            label="under ₹2,000"
            tone="muted"
            value={`${leak.smallest_bucket_count} of ${leak.deductions}`}
            sub="Not one of them is worth a phone call on its own."
          />
        </div>
      </Reveal>

      <Reveal>
        <p className="mt-14 max-w-[62ch] text-[17px] leading-[1.78] text-paper-2 md:text-[19.5px]">
          A good AR analyst can sort these out one remittance at a time, given an afternoon.
          Nobody gets the afternoon, so anything under a threshold quietly becomes a write-off.
          That is the job <strong className="font-medium text-paper">{BRAND.name}</strong> takes
          over.
        </p>
      </Reveal>
    </Section>
  );
}
