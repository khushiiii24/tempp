import { useState } from "react";
import Preloader from "./components/Preloader";
import Nav from "./components/Nav";
import Hero from "./components/Hero";
import CodeTicker from "./components/CodeTicker";
import TheLeak from "./components/TheLeak";
import Pipeline from "./components/Pipeline";
import CaseExplorer from "./components/CaseExplorer";
import Scoreboard from "./components/Scoreboard";
import Guardrails from "./components/Guardrails";
import Footer from "./components/Footer";

/**
 * Rhythm: loud → quiet → structural → interactive → measured → careful → fade.
 *
 * The hero is maximum density on purpose and the section immediately after it is the
 * quietest thing here. That contrast does more work than any single element; a page that
 * stayed at hero intensity would be unreadable by the third screen.
 */
export default function App() {
  const [booted, setBooted] = useState(false);

  return (
    <>
      {!booted && <Preloader onDone={() => setBooted(true)} />}

      {/* The continuity kit: one grain and one set of rules over every section, hero to
          footer. Sections change; the paper does not. */}
      <div className="rules" aria-hidden />
      <div className="vignette" aria-hidden />
      <div className="texture" aria-hidden />

      <Nav />
      {/* No opacity transition on `main`.
          Fading the page in made the content's visibility depend on a CSS transition
          completing, and a transition needs the frame loop as much as a scroll reveal does
          — in a window not producing frames, `main` stayed at `opacity: 0` behind an
          already-unmounted preloader. Nothing here is invisible waiting for an animation
          (BROKE entry 16). */}
      <main className="relative">
        <Hero />
        <CodeTicker />
        <TheLeak />
        <Pipeline />
        <CaseExplorer />
        <Scoreboard />
        <Guardrails />
      </main>
      <Footer />
    </>
  );
}
