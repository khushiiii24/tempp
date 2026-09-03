import { lazy, Suspense } from "react";
import { BRAND } from "../lib/brand";
import { byPolicy, pipeline } from "../lib/data";
import { humanise, inr, inrCompact, num, pct, STATE_TONE, TONE } from "../lib/format";
import Curve from "./Curve";
import { TipBody, TooltipCard, useTooltip } from "./Tooltip";
import { Badge, Note, Panel, Reveal, Section, SectionHead } from "./ui";

const SieveScene = lazy(() => import("./SieveScene"));

/**
 * How the money is filtered, and the honest count at every level.
 *
 * The 3D sieve and the list beside it are the same data: particles are assigned the ring
 * they stop at from these stage counts, so the attrition you watch is the attrition that
 * was measured.
 */
/** What each ending actually means, for the hover readout. */
const STATE_NOTE: Record<string, string> = {
  resolved_closed_valid:
    "Checked, found legitimate, closed. Nobody was contacted about these.",
  human_queue:
    "Handed to a person — too large to decide alone, or the model would not commit to a code.",
  stopped:
    "A stopping rule ended it: too small to be worth chasing, disputed, or no lawful way to make contact.",
  resolved_recovered: "The buyer paid the shortfall back.",
  resolved_written_off: "Booked as a loss.",
};

export default function Pipeline() {
  const stages = pipeline.funnel;
  const maxPaise = Math.max(...stages.map((s) => s.paise));
  const m = pipeline.matching;

  const agentCurve = byPolicy.agent.curve;
  const b1Curve = byPolicy.b1?.curve ?? [];
  const totalCases = pipeline.states.reduce((s, r) => s + r.count, 0);
  const funnelTip = useTooltip<(typeof stages)[number], HTMLOListElement>();
  const stateTip = useTooltip<(typeof pipeline.states)[number]>();

  return (
    <Section id="pipeline" backdrop="orbit" tone="jade">
      <SectionHead
        kicker="how it works"
        title="Everything goes in. Very little should come out."
        lede={
          <>
            From bank credit to closed case. Going from {num(stages[0].count)} invoices to{" "}
            {num(stages[stages.length - 1].count)} recoveries looks like failure until you
            remember that most of the gap was tax the buyer had to withhold. That money was never
            ours to collect.
          </>
        }
      />

      <div className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr] lg:gap-10">
        <Reveal>
          <Panel className="relative min-h-[420px] overflow-hidden lg:min-h-[620px]">
            <Suspense fallback={null}>
              <SieveScene stages={stages.map((s) => ({ label: s.label, count: s.count }))} />
            </Suspense>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 p-5">
              <div className="mono text-[10px] uppercase tracking-[0.2em] text-paper-3">
                {num(stages[0].count)} in · {num(stages[stages.length - 1].count)} out
              </div>
            </div>
          </Panel>
        </Reveal>

        <Reveal delay={100}>
          <ol className="relative" ref={funnelTip.ref}>
            <span
              className="absolute bottom-6 left-[15px] top-3 w-px"
              style={{ background: "var(--color-rule)" }}
              aria-hidden
            />
            {stages.map((s, i) => {
              const share = s.paise / maxPaise;
              const isLast = i === stages.length - 1;
              return (
                <li
                  key={s.key}
                  className="relative cursor-default rounded-md pb-6 pl-11 transition-colors last:pb-0 hover:bg-paper/[0.03]"
                  onMouseMove={(e) => funnelTip.show(e, s)}
                  onMouseLeave={funnelTip.hide}
                >
                  <span
                    className="absolute left-[9px] top-[7px] h-[13px] w-[13px] rounded-full border-2"
                    style={{
                      borderColor: isLast ? TONE.jade : TONE.gold,
                      background: "var(--color-ink)",
                    }}
                  />
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                    <h3 className="text-[15px] font-medium text-paper">{s.label}</h3>
                    <div className="mono flex items-baseline gap-3 text-[13px]">
                      <span style={{ color: isLast ? TONE.jade : "var(--color-paper)" }}>
                        {num(s.count)}
                      </span>
                      <span className="text-paper-3">{inrCompact(s.paise)}</span>
                    </div>
                  </div>
                  <div
                    className="mt-2 h-[5px] rounded-full transition-all duration-700"
                    style={{
                      width: `${Math.max(1.5, share * 100)}%`,
                      background: isLast ? TONE.jade : TONE.gold,
                      opacity: 0.28 + share * 0.62,
                    }}
                  />
                  <p className="mt-2 max-w-[54ch] text-[13.5px] leading-[1.68] text-paper-3">
                    {s.note}
                  </p>
                </li>
              );
            })}

            {funnelTip.tip && (
              <TooltipCard x={funnelTip.tip.x} y={funnelTip.tip.y} width={252}>
                <TipBody
                  label={funnelTip.tip.data.label}
                  value={num(funnelTip.tip.data.count)}
                  tone={TONE.gold}
                  rows={[
                    { k: "value at this stage", v: inr(funnelTip.tip.data.paise) },
                    {
                      k: "of what came in",
                      v: pct(funnelTip.tip.data.count / Math.max(1, stages[0].count), 1),
                    },
                  ]}
                  note={funnelTip.tip.data.note}
                />
              </TooltipCard>
            )}
          </ol>
        </Reveal>
      </div>

      <div className="mt-8 grid gap-8 md:grid-cols-2">
        <Reveal>
          <Panel hover className="h-full p-6">
            <div className="mb-5 flex items-baseline justify-between">
              <h3 className="mono text-[10.5px] uppercase tracking-[0.22em] text-paper-3">
                matching
              </h3>
              <Badge tone="leak">{num(m.exceptions_total)} refused</Badge>
            </div>
            <div className="flex items-baseline gap-4">
              <span className="mono text-[2.4rem] font-semibold leading-none" style={{ color: TONE.gold }}>
                {pct(m.match_rate)}
              </span>
              <span className="mono text-[12px] text-paper-3">
                {num(m.matched)} of {num(m.payments)} credits
              </span>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-x-8 gap-y-2">
              {m.by_method.map((r) => (
                <div key={r.method} className="flex items-baseline justify-between gap-3">
                  <span className="mono text-[11.5px] text-paper-2">{r.method}</span>
                  <span className="mono text-[11.5px] text-paper-3">{num(r.count)}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 border-t border-rule pt-4">
              {m.exceptions.map((e) => (
                <div key={e.kind} className="flex items-baseline justify-between gap-4 py-1">
                  <span className="text-[12px] text-paper-2">{humanise(e.kind)}</span>
                  <span className="mono text-[12px] text-paper-3">{num(e.count)}</span>
                </div>
              ))}
            </div>

            <p className="mt-5 text-[13.5px] leading-[1.72] text-paper-2">
              Guessing would score better on match rate and worse on everything after it.
              Twenty-one of these are two invoices in the same batch for exactly the same amount.
              There is no way to tell them apart, and picking one turns into a wrong letter three
              steps later.
            </p>
          </Panel>
        </Reveal>

        <Reveal delay={100}>
          <Panel hover className="h-full p-6">
            <h3 className="mono mb-5 text-[10.5px] uppercase tracking-[0.22em] text-paper-3">
              where {num(totalCases)} cases ended
            </h3>
            <div className="relative space-y-4" ref={stateTip.ref}>
              {[...pipeline.states]
                .sort((a, b) => b.count - a.count)
                .map((s) => {
                  const tone = STATE_TONE[s.state] ?? "muted";
                  return (
                    <div
                      key={s.state}
                      className="cursor-default rounded-md transition-colors hover:bg-paper/[0.03]"
                      onMouseMove={(e) => stateTip.show(e, s)}
                      onMouseLeave={stateTip.hide}
                    >
                      <div className="mb-1.5 flex items-baseline justify-between gap-4">
                        <span className="text-[13px] text-paper-2">{humanise(s.state)}</span>
                        <span className="mono text-[12px] text-paper-3">
                          {num(s.count)} · {pct(s.count / totalCases, 0)}
                        </span>
                      </div>
                      <div
                        className="h-[6px] rounded-full"
                        style={{
                          width: `${(s.count / totalCases) * 100}%`,
                          background: TONE[tone],
                          opacity: 0.9,
                        }}
                      />
                    </div>
                  );
                })}

              {stateTip.tip && (
                <TooltipCard x={stateTip.tip.x} y={stateTip.tip.y} width={248}>
                  <TipBody
                    label={humanise(stateTip.tip.data.state)}
                    value={`${num(stateTip.tip.data.count)} cases`}
                    tone={`var(--color-${STATE_TONE[stateTip.tip.data.state] ?? "muted"})`}
                    rows={[
                      { k: "share of all cases", v: pct(stateTip.tip.data.count / totalCases, 1) },
                    ]}
                    note={STATE_NOTE[stateTip.tip.data.state]}
                  />
                </TooltipCard>
              )}
            </div>
            <p className="mt-6 text-[13.5px] leading-[1.72] text-paper-2">
              Half were checked, found legitimate and closed without anyone being contacted. A
              third went to a person with the checking already done. Eighteen paid up.
            </p>
          </Panel>
        </Reveal>
      </div>

      <Reveal>
        <Panel className="mt-8 p-6 md:p-8">
          <h3 className="mono mb-7 text-[10.5px] uppercase tracking-[0.22em] text-paper-3">
            cash collected over forty-five days
          </h3>
          <Curve
            height={280}
            series={[
              {
                name: BRAND.name,
                tone: "jade",
                points: agentCurve.map((p) => ({ x: p.day, y: p.recovered_paise })),
              },
              {
                name: "write to everyone",
                tone: "leak",
                dashed: true,
                points: b1Curve.map((p) => ({ x: p.day, y: p.recovered_paise })),
              },
            ]}
          />
          <Note>
            We lose this one. Write to everyone and you collect about twice the cash, because
            some of the people you write to pay up. What a cash curve leaves out is the{" "}
            <span style={{ color: TONE.leak }}>
              {num(Number(byPolicy.b1?.harm.false_chase_contacts ?? 0))} letters
            </span>{" "}
            that went to customers who owed nothing, and the{" "}
            {inr(Number(byPolicy.agent.money.queued_reachable_paise ?? 0))} sitting on an analyst's
            desk ready to collect. Both are below.
          </Note>
        </Panel>
      </Reveal>
    </Section>
  );
}
