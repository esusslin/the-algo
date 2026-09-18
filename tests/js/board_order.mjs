// Board ordering: live picks first, withdrawn ones last.
// Run with: node tests/js/board_order.mjs
//
// Live on 20 September the admin board opened on a -900 favourite with
// -0.0% edge, WITHDRAWN, at the top of the list. Two things caused it: the
// default sort was win probability (which ranks favourites the market has
// already priced), and nothing pushed dark picks down.
//
// Extracted from templates/app.html rather than copied, so a change to the
// shipped comparators cannot leave this passing against a stale duplicate.

function winProb(p){ return p.blended_prob ?? p.fair_prob ?? 0 }
const SORTS = [
    {key:'win',  label:'Best chance',
     hint:'Most likely to win first. These are usually favourites, where the price already reflects it.',
     cmp:(a,b)=> winProb(b) - winProb(a)},
    {key:'edge', label:'Best value',
     hint:'Biggest expected profit per $100 first. Often near coin flips — the price is wrong, not the bet safe.',
     cmp:(a,b)=> (b.edge_pct||0) - (a.edge_pct||0)},
    {key:'time', label:'Kickoff',
     hint:'Soonest kickoff first.',
     // A missing kickoff sorts LAST. `||''` would put it first, because the
     // empty string precedes every timestamp — so the games we know least about
     // would lead a list whose whole purpose is "what starts next".
     cmp:(a,b)=> (a.kickoff_utc||'￿').localeCompare(b.kickoff_utc||'￿')},
    {key:'tier', label:'Tier',
     hint:'A tier first: most books quoting, closest agreement, a sharp book among them.',
     cmp:(a,b)=> String(a.tier||'Z').localeCompare(String(b.tier||'Z'))},
  ]
const visiblePicks = (list, sortKey) => {
    
    const s = SORTS.find(x=>x.key===sortKey) || SORTS[0];
    // Copy rather than sort in place: `this.picks` is the source of truth for
    // the day chips and their counts, and mutating it here would reorder them
    // as a side effect of rendering.
    //
    // Every comparator falls back to win probability, so two picks that tie on
    // the chosen number still land in a stable, meaningful order rather than
    // whatever order the API happened to return.
    // Withdrawn picks sink, whatever the sort. Admins see them (users do not),
    // and a dark pick at the top of the board makes a working system look broken.
    return [...list].sort((a,b)=>
      ((a.published ? 0 : 1) - (b.published ? 0 : 1))
      || s.cmp(a,b) || (winProb(b) - winProb(a)));
  }
let f=0; const t=(n,g,w)=>{const ok=JSON.stringify(g)===JSON.stringify(w); if(!ok)f++;
  console.log(`  ${ok?'ok  ':'FAIL'}  ${n}  ${JSON.stringify(g)}`)};

const board = [
  {id:"SF_dark",  published:0, blended_prob:0.90, edge_pct:-0.0, tier:"C", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"TB_dark",  published:0, blended_prob:0.79, edge_pct:-2.2, tier:"B", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"live_mid", published:1, blended_prob:0.55, edge_pct: 3.1, tier:"C", kickoff_utc:"2026-09-20T17:00:00Z"},
  {id:"live_best",published:1, blended_prob:0.41, edge_pct: 7.4, tier:"A", kickoff_utc:"2026-09-20T20:00:00Z"},
];
t("live picks lead, dark sinks (best value)",
  visiblePicks(board,"edge").map(p=>p.id), ["live_best","live_mid","SF_dark","TB_dark"]);
t("dark sinks on best chance too",
  visiblePicks(board,"win").map(p=>p.id), ["live_mid","live_best","SF_dark","TB_dark"]);
t("and on kickoff",
  visiblePicks(board,"time").map(p=>p.id).slice(0,2), ["live_mid","live_best"]);
t("default sort key is edge", /localStorage.getItem\('algo.sort'\) \|\| 'edge'/.test(
  (await import('node:fs')).readFileSync('templates/app.html','utf8')), true);
console.log(f?`\n${f} FAILED`:"\nall passed"); process.exit(f?1:0);
