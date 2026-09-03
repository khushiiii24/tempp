import { useEffect, useState } from "react";
import { BRAND } from "../lib/brand";
import Wordmark from "./Wordmark";
import { leak } from "../lib/data";
import { inr, num } from "../lib/format";
import { useReducedMotion } from "../lib/hooks";

/**
 * Three seconds to say what the word means and what the numbers are.
 *
 * It also earns its keep: it covers WebGL context creation and the font swap, which are
 * the two things that would otherwise flash.
 */
const STEPS = [
  { label: "invoices raised", value: () => num(leak.invoices) },
  { label: "credits received", value: () => num(leak.payments) },
  { label: "deductions found", value: () => num(leak.deductions) },
  { label: "unaccounted for", value: () => inr(leak.short_paid_paise) },
];

export default function Preloader({ onDone }: { onDone: () => void }) {
  const reduced = useReducedMotion();
  const [step, setStep] = useState(reduced ? STEPS.length : 0);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    if (reduced) {
      onDone();
      return;
    }
    const timers: number[] = [];
    STEPS.forEach((_, i) => {
      timers.push(window.setTimeout(() => setStep(i + 1), 320 + i * 270));
    });
    timers.push(window.setTimeout(() => setLeaving(true), 1720));
    timers.push(window.setTimeout(onDone, 2260));

    // Anyone who has seen it once can get past it.
    const skip = () => {
      timers.forEach(clearTimeout);
      setLeaving(true);
      window.setTimeout(onDone, 320);
    };
    window.addEventListener("keydown", skip, { once: true });
    window.addEventListener("pointerdown", skip, { once: true });

    return () => {
      timers.forEach(clearTimeout);
      window.removeEventListener("keydown", skip);
      window.removeEventListener("pointerdown", skip);
    };
  }, [onDone, reduced]);

  if (reduced) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-ink transition-opacity duration-500"
      style={{ opacity: leaving ? 0 : 1 }}
      role="status"
      aria-live="polite"
      aria-label={`Loading ${BRAND.name}`}
    >
      <div className="rules" />
      <div className="relative w-[min(540px,86vw)]">
        <Wordmark className="text-[40px] text-paper" />
        <div className="mono mt-2 text-[10.5px] uppercase tracking-[0.2em] text-paper-3">
          {BRAND.meaning}
        </div>

        <div className="mt-9 space-y-[7px]">
          {STEPS.map((s, i) => (
            <div
              key={s.label}
              className="mono flex items-baseline gap-3 text-[12.5px] transition-all duration-500"
              style={{
                opacity: step > i ? 1 : 0,
                transform: step > i ? "none" : "translateY(6px)",
              }}
            >
              <span className="text-paper-2">{s.label}</span>
              <span className="h-px flex-1 self-center gutter-rule" />
              <span className="text-paper">{s.value()}</span>
            </div>
          ))}
        </div>

        <div className="mt-8 h-px w-full overflow-hidden bg-rule">
          <div
            className="h-full bg-gold transition-[width] duration-300 ease-out"
            style={{ width: `${(step / STEPS.length) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
