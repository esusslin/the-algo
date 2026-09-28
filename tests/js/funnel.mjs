// Props funnel readout. Run with: node tests/js/funnel.mjs
//
// The funnel is a chain, so every stage after a zero is also zero. Colouring
// them all red says "five things are broken" when one thing is. And the most
// common real state — props flowing fine, nothing clearing tonight's edge
// floor — must NOT read as a failure, because that is a slate, not a bug.
//
// On 28 September a diagnostic that could not see props was read as proof that
// props were broken. This file exists so the replacement cannot make the
// opposite mistake as confidently.
//
// Extracted from templates/app.html, not copied.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../../templates/app.html", import.meta.url), "utf8");

const grab = (re, what) => {
  const m = html.match(re);
  if (!m) throw new Error(`could not find ${what} in templates/app.html — renamed?`);
  return m[0];
};

const stagesSrc = grab(/ {2}funnelStages\(\)\{.*?\n {2}\},/s, "funnelStages()").replace(/,$/, "");
const verdictSrc = grab(/ {2}funnelVerdict\(\)\{.*?\n {2}\},/s, "funnelVerdict()").replace(/,$/, "");

const { make } = await import("data:text/javascript," + encodeURIComponent(`
const _c = { ${stagesSrc}, ${verdictSrc} };
export const make = (funnel) => {
  _c.funnel = funnel;
  return { stages: _c.funnelStages(), verdict: _c.funnelVerdict() };
};`));

let f = 0;
const t = (n, g, w) => {
  const ok = JSON.stringify(g) === JSON.stringify(w);
  if (!ok) { f++; console.log(`  FAIL  ${n}\n        got  ${JSON.stringify(g)}\n        want ${JSON.stringify(w)}`); }
  else console.log(`  ok    ${n}`);
};

const F = (over = {}) => ({
  enabled: true, min_edge_pct: 5.0, tier_min_books: { A: 8, B: 6, C: 4 },
  collection: { rows: 20573 }, pairing: { paired: 6936 },
  pricing: { rows: 6936 },
  opportunities: { total: 56, best_books: 7, median_books: 5,
                   curve: [{ edge: 5, n: 3 }] },
  picks: { live: 3, withdrawn: 1 },
  ...over,
});

// --- only the FIRST zero is marked --------------------------------------
{
  const { stages } = make(F({ pairing: { paired: 0 }, pricing: { rows: 0 },
                              opportunities: { total: 0, curve: [{ edge: 5, n: 0 }] },
                              picks: { live: 0 } }));
  t("only one stage is marked dead", stages.filter(s => s.dead).length, 1);
  t("and it is the earliest one", stages.find(s => s.dead).label, "Two-sided (priceable)");
}

// --- the healthy-but-quiet case -----------------------------------------
{
  // 20,573 quotes, 6,936 priced, 56 with an edge, none over 5%. This is what a
  // one-game Monday looks like and it must not read as breakage.
  const { verdict } = make(F({ opportunities: { total: 56, best_books: 7, median_books: 5,
                                                curve: [{ edge: 5, n: 0 }] },
                               picks: { live: 0 } }));
  t("a quiet slate is not reported as broken", verdict.bad, false);
  t("and it names the floor", /5% floor/.test(verdict.text), true);
}

// --- genuine breakage ----------------------------------------------------
{
  const { verdict } = make(F({ collection: { rows: 0 }, pairing: { paired: 0 },
                               pricing: { rows: 0 },
                               opportunities: { total: 0, curve: [{ edge: 5, n: 0 }] },
                               picks: { live: 0 } }));
  t("no quotes at all IS reported as bad", verdict.bad, true);
  t("and names the stage", /Prop quotes stored/.test(verdict.text), true);
}

// --- the one-sided trap (anytime TD) -------------------------------------
{
  // Quotes present, none two-sided. Historically the stage most likely to eat
  // everything silently, so it must be called out as a cause, not a symptom.
  const { verdict } = make(F({ pairing: { paired: 0 }, pricing: { rows: 0 },
                               opportunities: { total: 0, curve: [{ edge: 5, n: 0 }] },
                               picks: { live: 0 } }));
  t("all-one-sided is a cause, not a consequence", verdict.bad, true);
  t("named correctly", /Two-sided/.test(verdict.text), true);
}

// --- flag off ------------------------------------------------------------
{
  const { stages, verdict } = make(F({ enabled: false }));
  t("a disabled flag is the first thing reported", stages.find(s => s.dead).label, "Enabled");
  t("and it is bad", verdict.bad, true);
}

// --- fully healthy -------------------------------------------------------
{
  const { stages, verdict } = make(F());
  t("nothing marked dead when every stage populates", stages.some(s => s.dead), false);
  t("verdict is clean", verdict.bad, false);
}

// --- the readout must reconcile with what an admin sees ------------------
{
  // Live on 28 September: funnel said "Live prop picks 1", the board showed 3.
  // Both were right — an admin's board includes dark picks — but the readout
  // gave no way to see that, so it looked like one of them was lying.
  const { stages } = make(F({ picks: { live: 1, withdrawn: 2 } }));
  const row = stages.find(s => s.label === "Live prop picks");
  t("the dark count is shown alongside the live one", row.value, "1  (+2 dark)");
  t("and 1 live is not treated as a zero stage", row.dead, false);
}
{
  const { stages } = make(F({ picks: { live: 3, withdrawn: 0 } }));
  t("no parenthetical when nothing is withdrawn",
    stages.find(s => s.label === "Live prop picks").value, "3");
}

console.log(f ? `\n${f} FAILED` : "\nall passed");
process.exit(f ? 1 : 0);
