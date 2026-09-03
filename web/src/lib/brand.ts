/**
 * The name, in one place.
 *
 * It was **Katauti** — the Hindi trade word for the amount a buyer knocks off an invoice.
 * Accurate and distinctive, and it failed the only test that matters for a wordmark: a
 * reader who does not already know the word cannot guess what it means, cannot say it, and
 * in the drawn Devanagari-style treatment could not reliably read it either (the first cut
 * of that logo read as KATAUM).
 *
 * **Recoup** says the whole product in one word that needs no gloss. Anyone in finance
 * knows it, anyone else can guess it, and it sets cleanly at 14px in a nav bar. The
 * distinctiveness moved out of the letterforms and into the mark beside them, which is
 * where a logo can afford to be unusual without costing legibility.
 */
export const BRAND = {
  name: "Recoup",
  /** Sits under the wordmark on the loader and in the footer. */
  meaning: "get back what was taken off the invoice",
  /** The headline, reused as the sign-off. */
  tagline: "Every invoice was paid. Not every rupee arrived.",
  /** One line, for meta descriptions and the footer. */
  oneLiner:
    "An agent for Indian B2B receivables. It works out which deductions are genuinely owed back, and chases only those.",
} as const;
