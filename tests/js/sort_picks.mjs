// Pick ordering. Run with: node tests/js/sort_picks.mjs
//
// **This reads the comparators out of templates/app.html rather than defining
// its own copy.** A copied comparator is a test that keeps passing after the
// shipped one changes — the same shape as a green pipeline over an empty pipe.
// If the extraction below stops finding SORTS, that is a failure, not a skip.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../../templates/app.html", import.meta.url), "utf8");

const grab = (re, what) => {
  const m = html.match(re);
  if (!m) throw new Error(`could not find ${what} in templates/app.html — did it get renamed?`);
  return m[0];
};

const winProbSrc = grab(/function winProb\(p\)\{[^}]*\}/s, "winProb()");
const sortsSrc = grab(/ {2}SORTS: \[.*?\n {2}\],/s, "SORTS")
  .replace("  SORTS: [", "const SORTS = [")
  .replace(/,$/, "");

const { SORTS, winProb } = await import(
  "data:text/javascript," +
  encodeURIComponent(`${winProbSrc}\n${sortsSrc}\nexport { SORTS, winProb };`)
);

// CIN and DET deliberately share a kickoff, so the tie-break is exercised.
// TBD deliberately has no kickoff and the best numbers, so a sort that mishandles
// null can't hide behind also being last on merit.
const PICKS = [
  { id: "CIN", blended_prob: 0.533, edge_pct: 6.5, kelly_units: 0.010, tier: "B",
    kickoff_utc: "2026-09-13T17:00:00+00:00" },
  { id: "LV",  blended_prob: 0.640, edge_pct: 5.6, kelly_units: 0.012, tier: "B",
    kickoff_utc: "2026-09-13T20:05:00+00:00" },
  { id: "DET", blended_prob: 0.399, edge_pct: 5.4, kelly_units: 0.004, tier: "A",
    kickoff_utc: "2026-09-13T17:00:00+00:00" },
  { id: "TBD", blended_prob: 0.700, edge_pct: 9.9, kelly_units: 0.020, tier: "C",
    kickoff_utc: null },
];

// Mirrors visiblePicks(): chosen comparator, then win probability as tie-break.
const by = (key) => {
  const s = SORTS.find((x) => x.key === key);
  if (!s) throw new Error(`no sort named ${key}`);
  return [...PICKS].sort((a, b) => s.cmp(a, b) || (winProb(b) - winProb(a)))
                   .map((p) => p.id);
};

let failed = 0;
const t = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) failed++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${name}`);
  if (!ok) console.log(`          got  ${JSON.stringify(got)}\n          want ${JSON.stringify(want)}`);
};

t("best chance ranks by win probability", by("win"), ["TBD", "LV", "CIN", "DET"]);
t("best value ranks by edge", by("edge"), ["TBD", "CIN", "LV", "DET"]);
t("kickoff ties break by win%, unknown kickoff sorts LAST",
  by("time"), ["CIN", "DET", "LV", "TBD"]);
t("biggest stake ranks by kelly_units", by("stake"), ["TBD", "LV", "CIN", "DET"]);
t("tier puts A first", by("tier"), ["DET", "LV", "CIN", "TBD"]);

// Sorting must not reorder the array the day-chips count from.
t("sorting leaves the source array alone", PICKS.map((p) => p.id),
  ["CIN", "LV", "DET", "TBD"]);

// The control. Without this, comparators that all returned 0 would satisfy
// nothing above except by coincidence of input order.
t("chance and value are genuinely different orders",
  JSON.stringify(by("win")) !== JSON.stringify(by("edge")), true);

t("every sort explains itself", SORTS.every((s) => s.hint && s.hint.length > 10), true);
t("the default sort exists", SORTS.some((s) => s.key === "win"), true);

console.log(failed ? `\n${failed} failed` : `\n${SORTS.length} sorts, all passed`);
process.exit(failed ? 1 : 0);
