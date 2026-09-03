import { BRAND } from "../lib/brand";
import { agent, guardrails as g, pipeline, scoreboard } from "../lib/data";
import { humanise, inr, num, titleCase, TONE } from "../lib/format";
import { Note, Panel, Reveal, Section, SectionHead } from "./ui";

/**
 * Human scale. The fest stops being a spectacle and becomes people you can call — here, the
 * system stops being a funnel and becomes the specific rules that stop it doing damage.
 *
 * Every value on this page is read out of `config/policy.yaml` through the snapshot, so the
 * page is a rendering of the actual configuration rather than a description of it. Change a
 * number in the YAML, re-export, and this section changes with it.
 */
export default function Guardrails() {
  const cards: { title: string; tone: keyof typeof TONE; body: React.ReactNode }[] = [
    {
      title: "When it may write to you",
      tone: "slate",
      body: (
        <>
          <p>
            {g.contact_window.start}–{g.contact_window.end} IST, {g.contact_days.map(titleCase).join(", ")}{" "}
            only. At most {g.max_contacts_per_case} contacts per case, {g.max_contacts_per_buyer_per_week}{" "}
            per buyer per week, and never less than {g.min_gap_hours} hours apart.
          </p>
          <p className="mt-2.5">
            A blocked contact is not quietly retried later. It is recorded as a decision, with
            the rule that blocked it.
          </p>
        </>
      ),
    },
    {
      title: "How it escalates",
      tone: "gold",
      body: (
        <>
          <p>
            Channels in order: <span className="mono text-paper">{g.channel_ladder.join(" → ")}</span>.
            Roles in order: <span className="mono text-paper">{g.escalation_ladder.map(humanise).join(" → ")}</span>.
          </p>
          <p className="mt-2.5">
            {g.require_consent_for.join(" and ")} need recorded consent, and DND overrides
            everything. If a buyer has consented to nothing we can use, the case goes to a person
            rather than out on a channel we are not allowed to use.
          </p>
        </>
      ),
    },
    {
      title: "What it may never say",
      tone: "leak",
      body: (
        <>
          <p>
            A draft containing any of these is rejected and replaced with the static template, and
            the rejection is counted rather than hidden:
          </p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {g.forbidden_phrases.map((p) => (
              <span key={p} className="mono border border-rule px-2 py-1 text-[10px] text-paper-2">
                "{p}"
              </span>
            ))}
          </div>
          <p className="mt-3">
            It also rejects any rupee figure that is not in the case record, any link we did not
            authorise, and any legal claim outside the template. The model writes the sentences.
            The policy engine decides what they are allowed to say.
          </p>
        </>
      ),
    },
    {
      title: "When it stops",
      tone: "violet",
      body: (
        <>
          <ul className="space-y-1.5">
            {Object.entries(g.stopping_rules).map(([k, v]) => (
              <li key={k} className="flex items-baseline justify-between gap-4 border-b border-rule/60 py-1">
                <span className="text-[12px] text-paper-3">{humanise(k)}</span>
                <span className="mono text-[11.5px] text-paper-2">
                  {typeof v === "boolean" ? (v ? "yes" : "no") : num(v)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3">
            Dispute raised, opt-out, checked and found valid, promise pending, not worth the
            cost, or small change owed by a very large account. Any one of them ends the case, and
            the log records which one did it.
          </p>
        </>
      ),
    },
    {
      title: "What a human must approve",
      tone: "gold",
      body: (
        <>
          <p>
            Credit holds always. Any claim above{" "}
            <span className="mono text-paper">{inr(Number(g.economics.human_review_threshold_paise))}</span>{" "}
            goes to a person before anything is sent. Credit notes above{" "}
            <span className="mono text-paper">{inr(Number(g.economics.cn_approval_threshold_paise))}</span>{" "}
            need sign-off.
          </p>
          <p className="mt-2.5">
            On this run: <span className="mono text-paper">{num(pipeline.approvals_requested)}</span>{" "}
            approvals requested, <span className="mono text-paper">{num(pipeline.approvals_granted)}</span>{" "}
            granted, and{" "}
            <span className="mono" style={{ color: TONE.jade }}>
              {num(Number(agent.harm.credit_holds_executed))}
            </span>{" "}
            credit holds executed. The queue is real. Nobody was standing at it during the run.
          </p>
        </>
      ),
    },
    {
      title: "What it can actually send",
      tone: "jade",
      body: (
        <>
          <p>
            Nothing. Every outbound message is queued with{" "}
            <span className="mono text-paper">dry_run=True</span>.{" "}
            <span className="mono text-paper">{num(pipeline.outbox)}</span> messages were queued on
            this run and{" "}
            <span className="mono" style={{ color: TONE.jade }}>
              {num(pipeline.outbox_sent_for_real)}
            </span>{" "}
            were sent — a figure that is zero by construction and stays zero unless both a{" "}
            <span className="mono text-paper">--live</span> flag and an environment variable are
            set.
          </p>
        </>
      ),
    },
  ];

  return (
    <Section id="guardrails" backdrop="orbit" tone="slate">
      <SectionHead
        kicker="the leash"
        title="Software that can email your customers needs rules."
        lede={
          <>
            Everything below is read from the same config file {BRAND.name} reads. This is not a
            description of the rules — it is the rules. Change a number and this page changes.
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {cards.map((c, i) => (
          <Reveal key={c.title} delay={i * 60} className="h-full">
            <Panel hover glow={c.tone} className="h-full px-6 py-7">
              <div className="mb-5 flex items-center gap-3">
                <span className="h-[3px] w-7 rounded-full" style={{ background: TONE[c.tone] }} />
                <h3 className="mono text-[10.5px] uppercase tracking-[0.2em] text-paper">
                  {c.title}
                </h3>
              </div>
              <div className="space-y-0 text-[13.5px] leading-[1.72] text-paper-2">{c.body}</div>
            </Panel>
          </Reveal>
        ))}
      </div>

      <Reveal>
        <div className="mt-12 grid gap-8 md:grid-cols-2">
          <div>
            <h3 className="mono mb-3 text-[10.5px] uppercase tracking-[0.22em]" style={{ color: TONE.jade }}>
              compliance is checked twice, by two implementations
            </h3>
            <p className="text-[14px] leading-[1.78] text-paper-2">
              One gate blocks the action before it happens. A separate auditor in{" "}
              <span className="mono text-paper-2">eval/compliance_audit.py</span> then rebuilds
              every violation from the contact log and the policy file, without importing that
              gate. A test makes sure it never can. If the same code both enforced the rules and
              checked them, "zero violations" would prove nothing.
            </p>
            <div className="mono mt-4 text-[12px]" style={{ color: TONE.jade }}>
              {num(
                scoreboard.policies.reduce((s, p) => s + Number(p.compliance?.violations ?? 0), 0),
              )}{" "}
              violations across all {scoreboard.policies.length} policies,{" "}
              {num(
                scoreboard.policies.reduce((s, p) => s + Number(p.harm.contacts_total ?? 0), 0),
              )}{" "}
              contacts audited
            </div>
          </div>
          <div>
            <h3 className="mono mb-3 text-[10.5px] uppercase tracking-[0.22em]" style={{ color: TONE.gold }}>
              every case replayable from the log alone
            </h3>
            <p className="text-[14px] leading-[1.78] text-paper-2">
              Decisions go into an append-only table, enforced by SQLite triggers rather than by
              good intentions. <span className="mono text-paper-2">replay --case CASE-0173-0</span>{" "}
              reads that table and nothing else. A test drops every other table in the database
              first, then checks the trace still rebuilds.
            </p>
          </div>
        </div>
      </Reveal>

      <Note>
        One number here differs from the spec:{" "}
        <span className="mono text-paper-2">stop_if_relationship_value_ratio_exceeds</span>. The
        spec sets it to 200, while the comment beside it describes ₹1,000 owed by a ₹2 crore
        customer, which is a ratio of 20,000. At 200 it stops 58% of every chaseable deduction on
        this batch. We went with the intent, set it to{" "}
        {num(Number(g.stopping_rules.stop_if_relationship_value_ratio_exceeds))}, and are saying so
        here.
      </Note>
    </Section>
  );
}
