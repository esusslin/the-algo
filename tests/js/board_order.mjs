// Board ordering: live picks first, withdrawn ones last.
// Run with: node tests/js/board_order.mjs
//
// Live on 20 September the admin board opened on a -900 favourite with
// -0.0% edge, WITHDRAWN, at the top of the list. Two things caused it: the
// default sort was win probability (which ranks favourites the market has
// already priced), and nothing pushed dark picks down.
//
// This file used to claim, in this comment, that it read the comparators out of
// templates/app.html — while actually defining its own copy thirty lines below.
// That copy went stale the moment the shipped sorts changed, and the only thing
// that noticed was an unrelated regex assertion at the bottom. A test that
// describes itself as extracting, and doesn't, is worse than an honest copy:
// it advertises a guarantee nobody is providing.
//
// It now genuinely extracts. If the extraction stops matching, that is a hard
// failure, not a silent skip.

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

// The shipped method reads `this.SORTS` / `this.sort` and filters by day. Bind
// it to a stand-in component so the ordering logic under test is the shipped
// text, character for character, rather than a paraphrase of it.
const visibleSrc = grab(/ {2}visiblePicks\(\)\{.*?\n {2}\},/s, "visiblePicks()")
  .replace(/,$/, "");

const { SORTS, winProb, visiblePicks } = await import(
  "data:text/javascript," + encodeURIComponent(`
${winProbSrc}
${sortsSrc}
const _c = { SORTS, day: 'all', dayKey: () => 'all', ${visibleSrc} };
const visiblePicks = (list, sortKey) => {
  _c.picks = list; _c.sort = sortKey;
  return _c.visiblePicks();
};
export { SORTS, winProb, visiblePicks };`)
);

let f = 0;
const t = (n, g, w) => {
  const ok = JSON.stringify(g) === JSON.stringify(w);
  if (!ok) { f++; console.log(`  FAIL  ${n}\n        got  ${JSON.stringify(g)}\n        want ${JSON.stringify(w)}`); }
  else console.log(`  ok    ${n}  ${JSON.stringify(g)}`);
};

const board = [
  {id:"SF_dark",  published:0, blended_prob:0.90, edge_pct:-0.0, tier:"C", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"TB_dark",  published:0, blended_prob:0.79, edge_pct:-2.2, tier:"B", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"live_mid", published:1, blended_prob:0.55, edge_pct: 3.1, tier:"C", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"live_best",published:1, blended_prob:0.41, edge_pct: 7.4, tier:"A", kickoff_utc:"2026-09-20T20:00:00Z"},
];

t("live picks lead, dark sinks (biggest edge)",
  visiblePicks(board,"edge").map(p=>p.id), ["live_best","live_mid","SF_dark","TB_dark"]);
t("dark sinks on best chance too",
  visiblePicks(board,"win").map(p=>p.id), ["live_mid","live_best","SF_dark","TB_dark"]);
t("and on kickoff",
  visiblePicks(board,"time").map(p=>p.id).slice(0,2), ["live_mid","live_best"]);
t("and on the default",
  visiblePicks(board,"best").map(p=>p.id), ["live_best","live_mid","TB_dark","SF_dark"]);

// ---------------------------------------------------------------------------
// The default must lead with the best-EVIDENCED pick, not the biggest number.
//
// On 18 September the board's default (raw edge) opened on Chris Godwin o47.5:
// 10.7% edge, tier C, four books, 3.7% cross-book disagreement — the thinnest
// support on the board. The A-tier Over 44.5, quoted by twenty-four books
// agreeing to within 1.1%, sat second at 5.0%.
//
// That is not a near-miss. Edge size and evidence quality are inversely related
// here: thin markets manufacture big edges, which is the entire reason
// MAX_PLAUSIBLE_EDGE_PCT exists. Sorting by edge alone reliably promotes
// whatever the guards did not quite catch.
// ---------------------------------------------------------------------------
const LIVE_18_SEP = [
  {id:"godwin",  published:1, tier:"C", edge_pct:10.7, blended_prob:0.52, kickoff_utc:"2026-09-18T20:00:00Z"},
  {id:"over445", published:1, tier:"A", edge_pct: 5.0, blended_prob:0.51, kickoff_utc:"2026-09-18T20:00:00Z"},
  {id:"chi3",    published:1, tier:"B", edge_pct: 5.0, blended_prob:0.53, kickoff_utc:"2026-09-18T20:00:00Z"},
];
t("the default leads with A tier, not the fattest edge",
  visiblePicks(LIVE_18_SEP,"best").map(p=>p.id), ["over445","chi3","godwin"]);
t("edge still orders WITHIN a tier",
  visiblePicks([...LIVE_18_SEP,
    {id:"chi_fat", published:1, tier:"B", edge_pct:8.0, blended_prob:0.50,
     kickoff_utc:"2026-09-18T20:00:00Z"}], "best").map(p=>p.id),
  ["over445","chi_fat","chi3","godwin"]);

// The control. Without it, a default that simply ignored edge would pass above.
t("'Biggest edge' is still available and still sorts by edge",
  visiblePicks(LIVE_18_SEP,"edge").map(p=>p.id)[0], "godwin");

const src = readFileSync(new URL("../../templates/app.html", import.meta.url), "utf8");
t("default sort key is 'best'",
  /localStorage\.getItem\('algo\.sort'\) \|\| 'best'/.test(src), true);

console.log(f ? `\n${f} FAILED` : "\nall passed");
process.exit(f ? 1 : 0);
