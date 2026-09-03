import { useEffect, useRef, useState } from "react";

/** Honoured by the 3D scenes, the counters and every reveal. Live, not read once. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

/* ==========================================================================================
   Reveal-on-scroll, without depending on the frame loop.

   `IntersectionObserver` callbacks are delivered during the rendering steps, so anything
   that stops frames stops them: an occluded window, a throttled tab, a compositor that has
   given up. When that happens every `.reveal` stays at its initial `opacity: 0` and the
   whole site renders as a black screen over a perfectly healthy DOM (BROKE entry 16).

   The first attempt at a fix probed once on mount — request a frame, race it against a
   timeout — and it was not enough. Frames *start*, so the probe settles "alive", and then
   they stop once the window is occluded; the failsafe had already disarmed itself. The
   measured state was 0 frames in 600 ms with all 37 reveals still hidden.

   So the observer is now the fast path, not the only path. A shared, passive `scroll`
   listener re-checks anything still hidden using `getBoundingClientRect`. Scroll events are
   input-driven rather than frame-driven, so they arrive when frames do not, and one
   listener serves every reveal on the page. Whichever mechanism notices first wins.
   ========================================================================================== */
interface Pending {
  el: HTMLElement;
  show: () => void;
}

const pending = new Set<Pending>();
let listening = false;
let queued = false;

function checkPending() {
  if (!pending.size) return;
  const h = window.innerHeight || 0;
  for (const entry of [...pending]) {
    const r = entry.el.getBoundingClientRect();
    // "Has the reader reached it", not "is it on screen right now".
    //
    // The check is debounced, so a fast scroll lands between samples and an element that
    // was briefly visible is already above the viewport by the time we look. Testing for
    // intersection leaves those permanently blank; testing for *passed* cannot. Revealing
    // something slightly early costs an animation nobody sees — leaving it hidden costs
    // the content.
    if (r.top < h * 0.95) {
      pending.delete(entry);
      entry.show();
    }
  }
}

function schedule() {
  if (queued) return;
  queued = true;
  // setTimeout, not requestAnimationFrame — the entire point is to survive a dead loop.
  window.setTimeout(() => {
    queued = false;
    checkPending();
  }, 90);
}

function ensureFallbackListener() {
  if (listening) return;
  listening = true;
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule, { passive: true });
  // One sweep after layout settles, so whatever is on screen at load is never waiting on a
  // scroll that may not come.
  window.setTimeout(checkPending, 300);
}

export function useInView<T extends HTMLElement>(
  options?: { once?: boolean; threshold?: number; rootMargin?: string },
) {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  const once = options?.once ?? true;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const entry: Pending = { el, show: () => setInView(true) };
    pending.add(entry);
    ensureFallbackListener();

    if (typeof IntersectionObserver === "undefined") {
      checkPending();
      return () => {
        pending.delete(entry);
      };
    }

    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setInView(true);
          pending.delete(entry);
          if (once) io.disconnect();
          return;
        }
        // Under `once`, never un-reveal. The observer fires with `isIntersecting: false`
        // for anything off screen — including things the fallback has *already* revealed
        // and dropped from the pending set. Writing that `false` back put ten elements
        // into a state nothing could rescue them from: hidden, scrolled past, and no
        // longer watched by either mechanism.
        if (!once) setInView(false);
      },
      {
        threshold: options?.threshold ?? 0.12,
        rootMargin: options?.rootMargin ?? "0px 0px -8% 0px",
      },
    );
    io.observe(el);

    return () => {
      pending.delete(entry);
      io.disconnect();
    };
  }, [once, options?.threshold, options?.rootMargin]);

  return { ref, inView };
}

/**
 * Count up to a real figure.
 *
 * Takes the target it will end on, so the number rendered at rest is always the measured
 * one — an animated counter that lands anywhere other than the true figure is a chart that
 * lies for 900 milliseconds.
 */
export function useCountUp(target: number, active: boolean, duration = 1100): number {
  const reduced = useReducedMotion();
  const [value, setValue] = useState(reduced ? target : 0);

  useEffect(() => {
    if (!active) return;
    if (reduced) {
      setValue(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    // If frames never arrive the animation never runs, so land on the real number anyway.
    const failsafe = window.setTimeout(() => setValue(target), duration + 400);
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
      else setValue(target);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(failsafe);
    };
  }, [target, active, duration, reduced]);

  return value;
}

/** Which section the reader is in, for the nav's current-section marker. */
export function useActiveSection(ids: string[]): string {
  const [active, setActive] = useState(ids[0] ?? "");
  useEffect(() => {
    const sections = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => Boolean(el));
    if (!sections.length) return;
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: "-45% 0px -45% 0px", threshold: [0, 0.25, 0.6, 1] },
    );
    sections.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, [ids]);
  return active;
}
