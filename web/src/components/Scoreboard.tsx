import { useState } from "react";
import { agent, byPolicy, meta, POLICY_BLURB, scoreboard } from "../lib/data";
import { inr, inrCompact, num, pct, TONE } from "../lib/format";
import { BRAND } from "../lib/brand";
import { TipBody, TooltipCard, useTooltip } from "./Tooltip";
import { Note, Panel, Reveal, Section, SectionHead } from "./ui";

const ORDER = ["agent", "b0", "b1", "b2", "b3"];

/**
 * The scoreboard, including the row where we lose.
 *
 * The agent collects roughly half the raw cash blanket dunning collects. That is stated
 * here, in the same size type as everything else, because a submission that buries it is
 * one query away from being caught and the honest version is a stronger argument anyway:
 * blanket dunning bought that cash with 473 letters to customers who had done nothing
 * wrong, and those letters are a cost that no recovery metric charges it for.
 */
export default function Scoreboard() {
  const [tab, setTab] = useState<"money" | "harm" | "accuracy" | "operations">("money");
  const policies = ORDER.map((p) => byPolicy[p]).filter(Boolean);
  const { ref: plotRef, tip, show, hide } = useTooltip<(typeof policies)[number]>();

  const maxAddressed = Math.max(...policies.map((p) => Number(p.money.addressed_paise ?? 0)));
  const maxFalse = Math.max(...policies.map((p) => Number(p.harm.false_chase_contacts ?? 0)));

  const ROWS: Record<string, { label: string; get: (p: (typeof policies)[number]) => string; strong?: boolean }[]> = {
    money: [
      { label: "Addressed — recovered + correctly escalated", get: (p) => inr(Number(p.money.addressed_paise)), strong: true },
      { label: "Recovered", get: (p) => inr(Number(p.money.recovered_paise)) },
      { label: "Reachable money escalated to a human", get: (p) => inr(Number(p.money.queued_reachable_paise)) },
      { label: "Recovery vs reachable ceiling", get: (p) => pct(Number(p.money.recovery_rate_vs_ceiling)) },
      { label: "Money correctly not chased", get: (p) => inr(Number(p.money.correctly_closed_valid_paise)) },
      { label: "Recoverable money abandoned", get: (p) => inr(Number(p.money.wrongly_written_off_paise)) },
      { label: "Intervention cost", get: (p) => inr(Number(p.money.cost_paise)) },
      { label: "₹ recovered per ₹ spent", get: (p) => (p.money.rupees_per_rupee_spent == null ? "—" : String(p.money.rupees_per_rupee_spent)) },
    ],
    harm: [
      { label: "False chases — contacts about valid deductions", get: (p) => num(Number(p.harm.false_chase_contacts)), strong: true },
      { label: "Customers wrongly contacted", get: (p) => num(Number(p.harm.false_chase_cases)) },
      { label: "Total contacts", get: (p) => num(Number(p.harm.contacts_total)) },
      { label: "Escalations to senior roles", get: (p) => num(Number(p.harm.escalations_to_senior_roles)) },
      { label: "Credit holds proposed", get: (p) => num(Number(p.harm.credit_holds_proposed)) },
      { label: "Credit holds executed", get: (p) => num(Number(p.harm.credit_holds_executed)) },
      { label: "Compliance violations (independent audit)", get: (p) => num(Number(p.compliance?.violations ?? 0)), strong: true },
    ],
    accuracy: [
      { label: "Verdict accuracy — chase or do not chase", get: (p) => pct(p.accuracy.verdict_accuracy) },
      { label: "Reason-code accuracy (answered only)", get: (p) => pct(p.accuracy.code_accuracy_answered) },
      { label: "Abstention rate", get: (p) => pct(p.accuracy.abstention_rate) },
    ],
    operations: [
      { label: "Cases", get: (p) => num(Number(p.operations.cases)) },
      { label: "Auto-resolved, no human touch", get: (p) => pct(Number(p.operations.auto_resolved_rate)) },
      { label: "Human queue", get: (p) => num(Number(p.operations.human_queue)) },
      { label: "Messages queued", get: (p) => num(Number(p.operations.messages_queued)) },
      { label: "Messages actually sent", get: (p) => num(Number(p.operations.messages_sent_for_real)) },
      { label: "Mean days to resolution", get: (p) => (p.operations.mean_days_to_resolution == null ? "—" : String(p.operations.mean_days_to_resolution)) },
    ],
  };

  return (
    <Section id="scoreboard" backdrop="stream" tone="jade">
      <SectionHead
        kicker="the results"
        title="Five ways to run a collections desk."
        lede={
          <>
            Same invoices, same {scoreboard.days}-day clock, same cost model, same customers. Every
            policy runs the identical loop; only the decision changes. Batch{" "}
            <span className="mono">{meta.db_content_hash.slice(0, 10)}</span>.
          </>
        }
      />

      {/* The chart the whole project is an argument about: money addressed against wrong
          letters sent. Up and to the left is the only place worth being. */}
      <Reveal>
        <Panel className="p-6 md:p-8">
          <h3 className="mono mb-1 text-[10.5px] uppercase tracking-[0.24em] text-paper-3">
            money handled vs. customers wrongly written to
          </h3>
          <p className="mb-7 max-w-[58ch] text-[14.5px] leading-[1.72] text-paper-2">
            Up and to the left is where you want to be. The question is not who collects the
            most. It is what they spent in customer goodwill getting there.
          </p>

          <div className="relative" ref={plotRef}>
            <svg
              viewBox="0 0 900 440"
              className="h-auto w-full overflow-visible"
              role="img"
              aria-label="Money handled against wrong letters sent, by policy"
            >
              <defs>
                <radialGradient id="sb-good" cx="0" cy="0" r="1">
                  <stop offset="0%" stopColor={TONE.jade} stopOpacity="0.16" />
                  <stop offset="100%" stopColor={TONE.jade} stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* The quadrant we want to be in, shaded. Saying "up and to the left" and then
                  making the reader locate it themselves wastes the sentence. */}
              <rect x="96" y="40" width="290" height="140" fill="url(#sb-good)" />

              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <g key={f}>
                  <line
                    x1="96"
                    x2="884"
                    y1={40 + f * 320}
                    y2={40 + f * 320}
                    stroke="rgba(233,230,222,0.075)"
                    strokeDasharray={f === 1 ? undefined : "3 5"}
                    vectorEffect="non-scaling-stroke"
                  />
                  <text
                    x="88"
                    y={44 + f * 320}
                    textAnchor="end"
                    className="mono"
                    fontSize="10.5"
                    fill="var(--color-paper-3)"
                  >
                    {inrCompact(maxAddressed * (1 - f))}
                  </text>
                </g>
              ))}

              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <text
                  key={f}
                  x={96 + f * 788}
                  y="384"
                  textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}
                  className="mono"
                  fontSize="10.5"
                  fill="var(--color-paper-3)"
                >
                  {Math.round(maxFalse * f)}
                </text>
              ))}
              <text x="96" y="410" className="mono" fontSize="10.5" fill="var(--color-paper-3)">
                letters sent to customers who owed nothing →
              </text>
              <text
                x="26"
                y="200"
                className="mono"
                fontSize="10.5"
                fill="var(--color-paper-3)"
                transform="rotate(-90 26 200)"
                textAnchor="middle"
              >
                money handled →
              </text>

              {policies.map((p, i) => {
                const x = 96 + (Number(p.harm.false_chase_contacts) / (maxFalse || 1)) * 788;
                const y = 360 - (Number(p.money.addressed_paise) / (maxAddressed || 1)) * 320;
                const isAgent = p.policy === "agent";
                const tone = isAgent ? TONE.jade : p.policy === "b0" ? TONE.muted : TONE.leak;
                const flip = x > 660;
                return (
                  <g
                    key={p.policy}
                    className="pop-in"
                    style={{ animationDelay: `${i * 130}ms`, transformOrigin: `${x}px ${y}px` }}
                  >
                    {/* Drop lines to both axes: a scatter point without them makes the
                        reader estimate two coordinates by eye. */}
                    <line
                      x1={x}
                      x2={x}
                      y1={y}
                      y2="360"
                      stroke={tone}
                      strokeOpacity="0.22"
                      strokeDasharray="2 4"
                      vectorEffect="non-scaling-stroke"
                    />
                    <line
                      x1="96"
                      x2={x}
                      y1={y}
                      y2={y}
                      stroke={tone}
                      strokeOpacity="0.22"
                      strokeDasharray="2 4"
                      vectorEffect="non-scaling-stroke"
                    />
                    {isAgent && (
                      <>
                        <circle cx={x} cy={y} r="26" fill={tone} opacity="0.1" />
                        <circle
                          cx={x}
                          cy={y}
                          r="16"
                          fill="none"
                          stroke={tone}
                          strokeOpacity="0.5"
                          className="pulse-dot"
                        />
                      </>
                    )}
                    <circle cx={x} cy={y} r={isAgent ? 8.5 : 6} fill={tone} />
                    {/* A generous invisible target. Six pixels of dot is a hard thing to
                        hit with a mouse and impossible with a finger. */}
                    <circle
                      cx={x}
                      cy={y}
                      r="24"
                      fill="transparent"
                      style={{ cursor: "pointer" }}
                      onMouseMove={(ev) => show(ev, p)}
                      onMouseLeave={hide}
                    />
                    <text
                      x={x + (flip ? -16 : 16)}
                      y={y + 4}
                      textAnchor={flip ? "end" : "start"}
                      className="mono"
                      fontSize={isAgent ? "14" : "12"}
                      fontWeight={isAgent ? 600 : 400}
                      fill={isAgent ? "var(--color-paper)" : "var(--color-paper-2)"}
                    >
                      {POLICY_BLURB[p.policy]?.name ?? p.policy}
                    </text>
                    <text
                      x={x + (flip ? -16 : 16)}
                      y={y + 21}
                      textAnchor={flip ? "end" : "start"}
                      className="mono"
                      fontSize="10.5"
                      fill="var(--color-paper-3)"
                    >
                      {inrCompact(Number(p.money.addressed_paise))} ·{" "}
                      {num(Number(p.harm.false_chase_contacts))} wrong
                    </text>
                  </g>
                );
              })}
            </svg>

            {tip && (
              <TooltipCard x={tip.x} y={tip.y} width={252}>
                <TipBody
                  label={POLICY_BLURB[tip.data.policy]?.name ?? tip.data.policy}
                  value={inr(Number(tip.data.money.addressed_paise))}
                  tone={tip.data.policy === "agent" ? TONE.jade : TONE.leak}
                  rows={[
                    { k: "collected", v: inr(Number(tip.data.money.recovered_paise)) },
                    { k: "passed to a person", v: inr(Number(tip.data.money.queued_reachable_paise)) },
                    { k: "wrong letters", v: num(Number(tip.data.harm.false_chase_contacts)) },
                    { k: "letters in total", v: num(Number(tip.data.harm.contacts_total)) },
                  ]}
                  note={POLICY_BLURB[tip.data.policy]?.what}
                />
              </TooltipCard>
            )}
          </div>
        </Panel>
      </Reveal>

      {/* The row we lose, said plainly and first. */}
      <Reveal>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <Panel hover glow="leak" className="px-6 py-7">
            <div className="mono text-[10px] uppercase tracking-[0.2em]" style={{ color: TONE.leak }}>
              where we lose
            </div>
            <div className="mono mt-3.5 text-[1.8rem] font-semibold leading-none text-paper">
              {inr(Number(agent.money.recovered_paise))}
            </div>
            <p className="mt-3.5 text-[14px] leading-[1.72] text-paper-2">
              against {inr(Number(byPolicy.b1.money.recovered_paise))} for writing to everyone.
              Roughly half. There is no way to read the data that makes that go away.
            </p>
          </Panel>

          <Panel hover glow="jade" className="px-6 py-7">
            <div className="mono text-[10px] uppercase tracking-[0.2em]" style={{ color: TONE.jade }}>
              where we win
            </div>
            <div className="mono mt-3.5 text-[1.8rem] font-semibold leading-none text-paper">
              {inr(Number(agent.money.addressed_paise))}
            </div>
            <p className="mt-3.5 text-[14px] leading-[1.72] text-paper-2">
              collected, or on an analyst's desk with the evidence attached. That is{" "}
              {(
                Number(agent.money.addressed_paise) / Number(byPolicy.b1.money.addressed_paise)
              ).toFixed(1)}
              × what writing to everyone handles.
            </p>
          </Panel>

          <Panel hover glow="gold" className="px-6 py-7">
            <div className="mono text-[10px] uppercase tracking-[0.2em]" style={{ color: TONE.gold }}>
              what that cost
            </div>
            <div className="mono mt-3.5 text-[1.8rem] font-semibold leading-none text-paper">
              {num(Number(byPolicy.b1.harm.false_chase_contacts))} letters
            </div>
            <p className="mt-3.5 text-[14px] leading-[1.72] text-paper-2">
              to {num(Number(byPolicy.b1.harm.false_chase_cases))} customers who had done nothing
              wrong. {BRAND.name} sent {num(Number(agent.harm.false_chase_contacts))} —{" "}
              {(
                Number(byPolicy.b1.harm.false_chase_contacts) /
                Math.max(1, Number(agent.harm.false_chase_contacts))
              ).toFixed(0)}
              × fewer.
            </p>
          </Panel>
        </div>
      </Reveal>

      {/* ------------------------------------------------------------------ table ---- */}
      <Reveal>
        <div className="mt-14">
          <div className="mb-4 flex flex-wrap gap-1.5">
            {(["money", "harm", "accuracy", "operations"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="mono px-3 py-1.5 text-[10.5px] uppercase tracking-[0.14em] transition-colors"
                style={{
                  color: tab === t ? "var(--color-ink)" : "var(--color-paper-2)",
                  background: tab === t ? TONE.gold : "transparent",
                  border: `1px solid ${tab === t ? TONE.gold : "var(--color-rule-2)"}`,
                }}
              >
                {t}
              </button>
            ))}
          </div>

          <div className="overflow-x-auto border border-rule">
            <table className="w-full min-w-[720px] border-collapse text-left">
              <thead>
                <tr className="border-b border-rule">
                  <th className="mono px-4 py-3 text-[10px] uppercase tracking-[0.18em] font-normal text-paper-3">
                    metric
                  </th>
                  {policies.map((p) => (
                    <th
                      key={p.policy}
                      className="mono px-4 py-3 text-right text-[10.5px] uppercase tracking-[0.14em] font-normal"
                      style={{ color: p.policy === "agent" ? TONE.jade : "var(--color-paper-2)" }}
                    >
                      {p.policy}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ROWS[tab].map((row) => (
                  <tr key={row.label} className="border-b border-rule/60 last:border-0">
                    <td className={`px-4 py-2.5 text-[12.5px] ${row.strong ? "text-paper" : "text-paper-2"}`}>
                      {row.label}
                    </td>
                    {policies.map((p) => (
                      <td
                        key={p.policy}
                        className={`mono px-4 py-2.5 text-right text-[12.5px] ${
                          row.strong ? "font-semibold" : ""
                        }`}
                        style={{
                          color:
                            p.policy === "agent" && row.strong
                              ? TONE.jade
                              : row.strong
                                ? "var(--color-paper)"
                                : "var(--color-paper-2)",
                        }}
                      >
                        {row.get(p)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {policies.map((p) => (
              <div key={p.policy} className="border-l border-rule pl-3.5">
                <div className="mono text-[11px]" style={{ color: p.policy === "agent" ? TONE.jade : "var(--color-paper)" }}>
                  {p.policy} · {POLICY_BLURB[p.policy]?.name}
                </div>
                <p className="mt-1 text-[12px] leading-[1.6] text-paper-3">
                  {POLICY_BLURB[p.policy]?.what}
                </p>
              </div>
            ))}
          </div>

          <Note>
            The accuracy rows are identical in all five columns because classification runs once
            over the batch; the policies differ only in what they do with the answer. B3 is the one
            that prices the checking — same classifier, verification switched off,{" "}
            {num(Number(byPolicy.b3.harm.false_chase_contacts))} wrong letters against{" "}
            {num(Number(agent.harm.false_chase_contacts))}. That gap is what verification is
            worth.
          </Note>
        </div>
      </Reveal>
    </Section>
  );
}
