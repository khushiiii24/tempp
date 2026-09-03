import { useEffect, useMemo, useState } from "react";
import showcaseRaw from "../data/showcase.json";
import { BRAND } from "../lib/brand";
import {
  agent,
  byPolicy,
  caseIndex,
  loadCases,
  type CaseRecord,
  type CaseIndexRow,
} from "../lib/data";
import {
  bp,
  humanise,
  inr,
  inrCompact,
  num,
  pct,
  STATE_TONE,
  TONE,
  VERDICT_TONE,
  type Tone,
} from "../lib/format";
import { Badge, Note, Panel, Reveal, Section, SectionHead } from "./ui";

const showcase = showcaseRaw as { id: string; reason: string; note: string }[];

type Filter = "showcase" | "recovered" | "escalated" | "abstained" | "stopped" | "all";

const FILTERS: { key: Filter; label: string; test: (r: CaseIndexRow) => boolean }[] = [
  { key: "showcase", label: "showcase", test: (r) => Boolean(r.showcase_id) },
  { key: "recovered", label: "recovered", test: (r) => r.recovered_paise > 0 },
  { key: "escalated", label: "escalated", test: (r) => r.state === "human_queue" },
  { key: "abstained", label: "model abstained", test: (r) => r.code === "NEEDS_HUMAN" },
  { key: "stopped", label: "stopped", test: (r) => r.state === "stopped" },
  { key: "all", label: `all ${caseIndex.length}`, test: () => true },
];

/**
 * One deduction, end to end — raw text in, decision out, with the answer key behind a
 * switch.
 *
 * This is the single most useful thing on the site and the reason a front end exists at
 * all. A scoreboard proves the system works in aggregate; only this proves it is doing
 * anything defensible on any individual case. The ground-truth toggle is deliberately a
 * separate, explicit action: truth is quarantined from every agent module in the codebase
 * and enforced by a test that greps for it, and showing it side by side with the agent's
 * own output by default would blur exactly the line the project rests on.
 */
export default function CaseExplorer() {
  const [filter, setFilter] = useState<Filter>("showcase");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string>(
    caseIndex.find((c) => c.showcase_id === "SHOW-CONTRADICT")?.id ?? caseIndex[0]?.id ?? "",
  );
  const [cases, setCases] = useState<Map<string, CaseRecord> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadCases()
      .then((rows) => {
        if (alive) setCases(new Map(rows.map((r) => [r.id, r])));
      })
      .catch((e: Error) => alive && setLoadError(e.message));
    return () => {
      alive = false;
    };
  }, []);

  const rows = useMemo(() => {
    const test = FILTERS.find((f) => f.key === filter)!.test;
    const q = query.trim().toLowerCase();
    return caseIndex
      .filter(test)
      .filter(
        (r) =>
          !q ||
          r.id.toLowerCase().includes(q) ||
          r.buyer.toLowerCase().includes(q) ||
          r.code.toLowerCase().includes(q),
      )
      .sort((a, b) => b.amount_paise - a.amount_paise);
  }, [filter, query]);

  const record = cases?.get(selected) ?? null;

  return (
    <Section id="cases" backdrop="grain" tone="violet">
      <SectionHead
        kicker="the evidence"
        title="Open any case. Flip the answer key."
        lede={
          <>
            All {caseIndex.length} of them, with the text that came in, what the model made of
            it, what the source systems said, which rules fired and what got sent. The answer key
            is behind a switch, and nothing inside the agent is allowed to read it.
          </>
        }
      />

      {/* Seven scenarios that are guaranteed to be in the batch. Declared up front in
          config/generator.yaml so a demo never depends on the RNG producing a good case. */}
      <Reveal>
        <div className="mb-8">
          <div className="mono mb-3 text-[10px] uppercase tracking-[0.22em] text-paper-3">
            seven scenarios, fixed in config before the batch is generated
          </div>
          <div className="flex flex-wrap gap-2">
            {showcase.map((s) => {
              const target = caseIndex.find((c) => c.showcase_id === s.id);
              if (!target) return null;
              const on = selected === target.id;
              return (
                <button
                  key={s.id}
                  onClick={() => {
                    setFilter("showcase");
                    setSelected(target.id);
                  }}
                  title={s.note}
                  className="mono rounded-full px-3.5 py-1.5 text-[10.5px] uppercase tracking-[0.12em] transition-colors"
                  style={{
                    color: on ? "var(--color-ink)" : "var(--color-paper-2)",
                    background: on ? TONE.gold : "transparent",
                    border: `1px solid ${on ? TONE.gold : "var(--color-rule-2)"}`,
                  }}
                >
                  {s.id.replace("SHOW-", "").toLowerCase().replace(/-/g, " ")}
                </button>
              );
            })}
          </div>
        </div>
      </Reveal>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        {/* -------------------------------------------------------------- list ---- */}
        <div className="lg:sticky lg:top-24 lg:self-start">
          <div className="mb-3 flex flex-wrap gap-1.5">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className="mono px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] transition-colors"
                style={{
                  color: filter === f.key ? "var(--color-paper)" : "var(--color-paper-3)",
                  borderBottom: `1px solid ${filter === f.key ? TONE.gold : "transparent"}`,
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="case id, buyer or reason code"
            aria-label="Search cases"
            className="mono w-full rounded-xl border border-rule bg-ink-2 px-4 py-3 text-[12px] text-paper placeholder:text-paper-3 focus:border-rule-2 focus:outline-none"
          />

          <ul className="mt-3 max-h-[62vh] overflow-y-auto rounded-xl border border-rule">
            {rows.map((r) => {
              const on = selected === r.id;
              return (
                <li key={r.id}>
                  <button
                    onClick={() => setSelected(r.id)}
                    className="hairline w-full px-3.5 py-3 text-left transition-colors"
                    style={{ background: on ? "rgba(232,178,76,0.10)" : "transparent" }}
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="mono text-[11px]" style={{ color: on ? TONE.gold : "var(--color-paper-2)" }}>
                        {r.id}
                      </span>
                      <span className="mono text-[11.5px] text-paper">
                        {inrCompact(r.amount_paise)}
                      </span>
                    </div>
                    <div className="mt-1 truncate text-[12px] text-paper-2">{r.buyer}</div>
                    <div className="mono mt-1.5 flex items-center gap-2 text-[9.5px] uppercase tracking-[0.1em]">
                      <span style={{ color: TONE[STATE_TONE[r.state] ?? "muted"] }}>
                        {humanise(r.state)}
                      </span>
                      <span className="text-paper-3">·</span>
                      <span className="text-paper-3">{r.code}</span>
                    </div>
                  </button>
                </li>
              );
            })}
            {!rows.length && (
              <li className="px-3.5 py-6 text-[12.5px] text-paper-3">Nothing matches that.</li>
            )}
          </ul>
        </div>

        {/* ------------------------------------------------------------ detail ---- */}
        <div className="min-w-0">
          {loadError ? (
            <Panel className="p-6 text-[13px] text-paper-2">
              Could not load the case file ({loadError}). Run{" "}
              <span className="mono text-paper">python -m deduction_desk export-web</span> and
              reload.
            </Panel>
          ) : !record ? (
            <Panel className="animate-pulse p-6">
              <div className="h-4 w-40 bg-rule" />
              <div className="mt-4 h-3 w-full bg-rule" />
              <div className="mt-2 h-3 w-3/4 bg-rule" />
              <div className="mt-8 h-40 w-full bg-rule/60" />
            </Panel>
          ) : (
            <CaseDetail record={record} />
          )}
        </div>
      </div>
    </Section>
  );
}

/* ======================================================================================
   Detail
   ====================================================================================== */
function Step({
  n,
  title,
  children,
  tone = "gold",
}: {
  n: string;
  title: string;
  children: React.ReactNode;
  tone?: Tone;
}) {
  return (
    <section className="relative border-l border-rule pb-8 pl-6 last:pb-0">
      <span
        className="absolute -left-[5px] top-1 h-[9px] w-[9px] rounded-full"
        style={{ background: TONE[tone] }}
      />
      <div className="mono mb-3 flex items-center gap-3 text-[10px] uppercase tracking-[0.22em]">
        <span className="text-paper-3">{n}</span>
        <span className="text-paper">{title}</span>
      </div>
      {children}
    </section>
  );
}

function CaseDetail({ record }: { record: CaseRecord }) {
  const [showTruth, setShowTruth] = useState(false);
  const d = record.deduction;
  const v = d.verification ?? {};
  const truth = record.truth;

  // Reset the toggle whenever a different case is opened. Leaving it on would mean the
  // second case you look at shows you the answer before you have read the question.
  useEffect(() => setShowTruth(false), [record.id]);

  const evidence = Object.entries(v.evidence ?? {});
  const rulesFired = Array.from(
    new Set(record.trace.flatMap((t) => t.rules).filter(Boolean)),
  );

  return (
    <Panel className="p-6 md:p-8">
      <header className="mb-8 border-b border-rule pb-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mono text-[11px] tracking-[0.14em] text-paper-3">{record.id}</div>
            <h3 className="display mt-1.5 text-[clamp(1.5rem,3vw,2.25rem)] font-bold text-paper">
              {record.buyer.name}
            </h3>
            <div className="mono mt-2 flex flex-wrap items-center gap-2 text-[10px]">
              <Badge tone={STATE_TONE[record.state] ?? "muted"}>{humanise(record.state)}</Badge>
              <Badge tone={VERDICT_TONE[d.verdict] ?? "muted"}>{humanise(d.verdict)}</Badge>
              <Badge tone="muted">{record.buyer.segment}</Badge>
              <Badge tone="muted">{humanise(record.buyer.behaviour)}</Badge>
              {record.buyer.dnd && <Badge tone="leak">dnd</Badge>}
            </div>
          </div>
          <div className="text-right">
            <div className="mono text-[10px] uppercase tracking-[0.2em] text-paper-3">
              short paid
            </div>
            <div className="mono text-[1.7rem] font-semibold leading-none" style={{ color: TONE.gold }}>
              {inr(d.amount_paise)}
            </div>
            <div className="mono mt-1.5 text-[11px] text-paper-3">
              of {inr(record.invoice.total_paise)} · {record.invoice.no}
            </div>
          </div>
        </div>
      </header>

      <div>
        <Step n="01" title="what arrived">
          <div className="grid gap-4 md:grid-cols-2">
            <dl className="space-y-1.5">
              {[
                ["Invoice", record.invoice.no],
                ["Taxable", inr(record.invoice.taxable_paise)],
                ["GST", inr(record.invoice.gst_paise)],
                ["Invoice total", inr(record.invoice.total_paise)],
                ["Credit received", inr(record.payment.amount_paise)],
                ["UTR", record.payment.utr],
                ["Value date", record.payment.value_date],
              ].map(([k, val]) => (
                <div key={k} className="flex items-baseline justify-between gap-4 border-b border-rule/60 py-1.5">
                  <dt className="text-[12px] text-paper-3">{k}</dt>
                  <dd className="mono text-[12px] text-paper-2">{val}</dd>
                </div>
              ))}
            </dl>

            <div>
              <div className="mono mb-2 text-[10px] uppercase tracking-[0.18em] text-paper-3">
                bank narration
              </div>
              <pre className="mono mb-4 overflow-x-auto whitespace-pre-wrap border border-rule bg-ink-2 px-3 py-2.5 text-[11px] leading-[1.6] text-paper-2">
                {record.payment.narration || "—"}
              </pre>
              <div className="mono mb-2 text-[10px] uppercase tracking-[0.18em] text-paper-3">
                remittance advice {record.advice ? `· ${record.advice.format}` : ""}
              </div>
              <pre className="mono max-h-64 overflow-auto whitespace-pre-wrap border border-rule bg-ink-2 px-3 py-2.5 text-[11px] leading-[1.6] text-paper-2">
                {record.advice?.raw_text.trim() ||
                  "No advice was received for this payment. The reason for the deduction has to be inferred from the arithmetic alone."}
              </pre>
            </div>
          </div>
        </Step>

        <Step n="02" title="what the model said" tone="violet">
          {d.predicted_code === "NEEDS_HUMAN" ? (
            <div className="border border-violet/30 bg-violet/5 px-4 py-3.5">
              <div className="mono text-[12px]" style={{ color: TONE.violet }}>
                abstained
              </div>
              <p className="mt-2 max-w-[60ch] text-[13px] leading-[1.65] text-paper-2">
                Confidence came in under the floor, so nothing was done with the guess. A wrong
                code here turns into a wrong letter three steps later. Saying nothing is the
                cheaper mistake, and it shows up on the scoreboard as an abstention.
              </p>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-3">
                <span className="mono text-[15px] text-paper">{d.predicted_code}</span>
                <Badge tone="violet">{pct(d.confidence, 0)} confidence</Badge>
                <Badge tone="muted">{d.predicted_by}</Badge>
              </div>
              <p className="mt-3 max-w-[64ch] text-[13.5px] leading-[1.7] text-paper-2">
                {d.rationale || "—"}
              </p>
              {d.claimed_reason_text && (
                <p className="mt-3 max-w-[64ch] text-[12.5px] leading-[1.6] text-paper-3">
                  Buyer's own words: <span className="text-paper-2">"{d.claimed_reason_text}"</span>
                </p>
              )}
            </>
          )}
        </Step>

        <Step
          n="03"
          title="what verification found"
          tone={VERDICT_TONE[d.verdict] ?? "muted"}
        >
          <div className="flex flex-wrap items-center gap-3">
            <span
              className="mono text-[15px]"
              style={{ color: TONE[VERDICT_TONE[d.verdict] ?? "muted"] }}
            >
              {humanise(d.verdict)}
            </span>
            <span className="mono text-[12.5px] text-paper-2">
              recoverable {inr(d.recoverable_paise)}
            </span>
            {v.recheck_after_days ? (
              <Badge tone="slate">re-check in {v.recheck_after_days}d</Badge>
            ) : null}
          </div>

          {evidence.length > 0 && (
            <dl className="mt-4 grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
              {evidence.map(([k, val]) => (
                <div key={k} className="flex items-baseline justify-between gap-4 border-b border-rule/60 py-1.5">
                  <dt className="text-[12px] text-paper-3">{humanise(k)}</dt>
                  <dd className="mono text-[12px] text-paper-2">
                    {k.endsWith("_paise") ? inr(Number(val)) : k.endsWith("_bp") ? bp(Number(val)) : String(val)}
                  </dd>
                </div>
              ))}
            </dl>
          )}

          {(v.rules_fired?.length ?? 0) > 0 && (
            <div className="mt-4 flex flex-wrap gap-1.5">
              {v.rules_fired!.map((r) => (
                <span key={r} className="mono border border-rule px-2 py-1 text-[10px] text-paper-2">
                  {r}
                </span>
              ))}
            </div>
          )}

          <div className="mono mt-4 text-[11px] text-paper-3">
            Contract expects {record.contract.tds_section_expected} at{" "}
            {bp(record.contract.tds_rate_expected_bp)} · {record.contract.delivery_terms} ·{" "}
            {record.contract.payment_terms_days}-day terms
          </div>
        </Step>

        <Step n="04" title="what the policy decided" tone="gold">
          {rulesFired.length > 0 ? (
            <div className="mb-4 flex flex-wrap gap-1.5">
              {rulesFired.map((r) => (
                <span
                  key={r}
                  className="mono border px-2 py-1 text-[10px]"
                  style={{
                    color: r.startsWith("stopping") ? TONE.leak : TONE.gold,
                    borderColor: `color-mix(in srgb, ${
                      r.startsWith("stopping") ? TONE.leak : TONE.gold
                    } 30%, transparent)`,
                  }}
                >
                  {r}
                </span>
              ))}
            </div>
          ) : null}

          <ol className="space-y-2">
            {record.trace.map((t) => (
              <li key={t.seq} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule/60 py-1.5">
                <span className="mono w-[86px] shrink-0 text-[10.5px] text-paper-3">
                  {t.sim_date}
                </span>
                <span className="mono text-[11px] text-paper-2">{t.stage}</span>
                {typeof t.decision.action === "string" && (
                  <span className="mono text-[11px]" style={{ color: TONE.gold }}>
                    → {String(t.decision.action)}
                  </span>
                )}
                {typeof t.outcome?.case_state === "string" && (
                  <span className="mono text-[10.5px] text-paper-3">
                    {humanise(String(t.outcome.case_state))}
                  </span>
                )}
              </li>
            ))}
            {!record.trace.length && (
              <li className="text-[12.5px] text-paper-3">
                No decision-log rows for this case in this run.
              </li>
            )}
          </ol>

          {record.stop_reason && (
            <p className="mt-4 border-l-2 pl-4 text-[13px] leading-[1.65] text-paper-2" style={{ borderColor: TONE.leak }}>
              Stopped: <span className="mono">{record.stop_reason}</span>
            </p>
          )}
          {record.human_reason && (
            <p className="mt-4 border-l-2 pl-4 text-[13px] leading-[1.65] text-paper-2" style={{ borderColor: TONE.violet }}>
              Escalated to a person: <span className="mono">{record.human_reason}</span>
            </p>
          )}
        </Step>

        <Step n="05" title={`what was sent · ${record.contacts.length}`} tone="jade">
          {record.contacts.length === 0 ? (
            <p className="max-w-[60ch] text-[13px] leading-[1.7] text-paper-3">
              Nothing was sent, and that is the result rather than a gap in the record. Across
              the batch {BRAND.name} sent {num(Number(agent.harm.contacts_total))} messages where
              writing to everyone sent {num(Number(byPolicy.b1?.harm.contacts_total ?? 0))}. Almost
              all of the difference is letters about deductions that were legitimate.
            </p>
          ) : (
            <div className="space-y-4">
              {record.contacts.map((c, i) => (
                <div key={i} className="border border-rule bg-ink-2">
                  <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-4 py-2 text-[10px] uppercase tracking-[0.14em] text-paper-3">
                    <span style={{ color: TONE.jade }}>{c.channel}</span>
                    <span>{c.role}</span>
                    <span>{c.ts.slice(0, 16).replace("T", " ")}</span>
                    <span className="ml-auto">{c.template_id}</span>
                    <Badge tone={c.drafted_by === "llm" ? "violet" : "muted"}>{c.drafted_by}</Badge>
                  </div>
                  <div className="px-4 py-3">
                    {c.subject && (
                      <div className="mb-2 text-[13px] font-medium text-paper">{c.subject}</div>
                    )}
                    <pre className="mono max-h-56 overflow-auto whitespace-pre-wrap text-[11.5px] leading-[1.7] text-paper-2">
                      {c.body}
                    </pre>
                    {c.checks.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {c.checks.map((k) => (
                          <span key={k} className="mono text-[9.5px] tracking-[0.1em] text-paper-3">
                            ✓ {k}
                          </span>
                        ))}
                      </div>
                    )}
                    {c.response_kind && (
                      <div className="mt-3 border-t border-rule pt-3">
                        <div className="mono text-[10px] uppercase tracking-[0.16em] text-paper-3">
                          reply · {c.response_kind}
                        </div>
                        <p className="mt-1.5 text-[12.5px] leading-[1.6] text-paper-2">
                          {c.response_text}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Step>

        <Step n="06" title="outcome" tone={record.recovered_paise > 0 ? "jade" : "muted"}>
          <div className="grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
            {[
              ["Recovered", inr(record.recovered_paise)],
              ["Written off", inr(record.written_off_paise)],
              ["Credit note issued", inr(record.credit_note_paise)],
              ["Intervention cost", inr(record.cost_paise)],
              ["Contacts used", `${record.contacts_used}`],
              ["Closed", record.closed_at?.slice(0, 10) ?? "open"],
            ].map(([k, val]) => (
              <div key={k} className="flex items-baseline justify-between gap-4 border-b border-rule/60 py-1.5">
                <span className="text-[12px] text-paper-3">{k}</span>
                <span className="mono text-[12px] text-paper-2">{val}</span>
              </div>
            ))}
          </div>
        </Step>
      </div>

      {/* ----------------------------------------------------------- ground truth ---- */}
      <div className="mt-8 border-t border-rule pt-6">
        <button
          onClick={() => setShowTruth((v) => !v)}
          aria-expanded={showTruth}
          className="mono flex items-center gap-3 text-[11px] uppercase tracking-[0.2em] transition-colors"
          style={{ color: showTruth ? TONE.leak : "var(--color-paper-2)" }}
        >
          <span
            className="inline-flex h-4 w-8 items-center rounded-full border px-[2px] transition-colors"
            style={{
              borderColor: showTruth ? TONE.leak : "var(--color-rule-2)",
              background: showTruth ? "color-mix(in srgb, var(--color-leak) 20%, transparent)" : "transparent",
            }}
          >
            <span
              className="h-[10px] w-[10px] rounded-full transition-transform"
              style={{
                background: showTruth ? TONE.leak : "var(--color-paper-3)",
                transform: showTruth ? "translateX(14px)" : "none",
              }}
            />
          </span>
          reveal ground truth
        </button>

        {showTruth &&
          (truth ? (
            <div className="mt-5">
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <Badge tone={truth.code_correct ? "jade" : "leak"}>
                  {truth.code_correct ? "code correct" : "code wrong"}
                </Badge>
                <Badge tone={truth.is_valid ? "slate" : "gold"}>
                  {truth.is_valid ? "deduction was legitimate" : "deduction was not legitimate"}
                </Badge>
                {truth.will_pay_if_chased && <Badge tone="jade">would pay if chased</Badge>}
                {truth.will_dispute && <Badge tone="leak">would dispute</Badge>}
                {truth.opt_out && <Badge tone="leak">would opt out</Badge>}
                {truth.promise_then_default && <Badge tone="gold">promises, then defaults</Badge>}
              </div>
              <dl className="grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
                {[
                  ["True reason code", truth.true_reason_code],
                  ["Agent predicted", record.deduction.predicted_code],
                  ["Truly recoverable", inr(truth.recoverable_paise)],
                  ["Agent said recoverable", inr(record.deduction.recoverable_paise)],
                  ["Pays after n contacts", truth.pays_after_n_contacts === 99 ? "never" : String(truth.pays_after_n_contacts)],
                  ["Responds only at", truth.responds_only_at_role],
                  ["Responds to", truth.responds_to_channels.join(", ") || "—"],
                ].map(([k, val]) => (
                  <div key={k} className="flex items-baseline justify-between gap-4 border-b border-rule/60 py-1.5">
                    <dt className="text-[12px] text-paper-3">{k}</dt>
                    <dd className="mono text-[12px] text-paper-2">{val}</dd>
                  </div>
                ))}
              </dl>
              {truth.notes && (
                <p className="mt-4 text-[12.5px] leading-[1.65] text-paper-3">{truth.notes}</p>
              )}
            </div>
          ) : (
            <p className="mt-4 text-[12.5px] text-paper-3">No truth row for this case.</p>
          ))}
      </div>

      <Note>
        The generator wrote this answer key <em>before</em> the agent ran, into a table that no
        module under <span className="mono">ingest/</span>, <span className="mono">matching/</span>,{" "}
        <span className="mono">classify/</span>, <span className="mono">verify/</span>,{" "}
        <span className="mono">policy/</span> or <span className="mono">actions/</span> is allowed
        to import. A test greps those packages for it and fails the build if it turns up, so the
        agent cannot move its own score.
      </Note>
    </Panel>
  );
}
