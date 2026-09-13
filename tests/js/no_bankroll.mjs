// The app must not know, ask, store or opine on anyone's bankroll.
// Run with: node tests/js/no_bankroll.mjs
//
// This is a product rule, not a style preference, so it gets a test. HTML
// comments and JS line comments are stripped before checking, because the
// rationale is allowed to explain what was removed — the rendered app is not
// allowed to mention it.

function winProb(p){ return p.blended_prob ?? p.fair_prob ?? 0 }
const fmtOdds = (p) => {return p>0?'+'+p:''+p}
const whyText = (p) => {
    // `book_count` is not a column on picks — it rides inside the
    // `prob_components` JSON, which is also what the "where does this number
    // come from?" panel reads. Taking it from there rather than adding a column
    // keeps one source for the figure; two would eventually disagree.
    let books = 0;
    try{ books = (JSON.parse(p.prob_components||'{}').book_count) || 0 }catch(e){}
    const gap = (winProb(p)*100).toFixed(1);
    return (books ? `${books} books say ${gap}%` : `Market says ${gap}%`)
         + ` · this price is ${p.edge_pct.toFixed(1)}% better than fair`;
  }
import { readFileSync } from 'node:fs';
const html = readFileSync('templates/app.html','utf8');
let failed=0;
const t=(n,g,w)=>{const ok=JSON.stringify(g)===JSON.stringify(w); if(!ok)failed++;
  console.log(`  ${ok?'ok  ':'FAIL'}  ${n}${ok?'':`\n          got ${JSON.stringify(g)}`}`)};

const CIN={description:'CIN', blended_prob:0.674, edge_pct:2.9, best_price:-190, best_book:'betmgm',
  prob_components:JSON.stringify({market_fair:0.674,anchor:'sharp',book_count:25,dispersion:0.012})};
console.log("the card:");
console.log(`          ${CIN.description} at ${fmtOdds(CIN.best_price)} on ${CIN.best_book}`);
console.log(`          ${whyText(CIN)}\n`);

console.log("bankroll is gone from the UI:");
// strip HTML comments and JS line comments before checking — the rationale is
// allowed to mention what was removed; the rendered app is not.
const stripped = html.replace(/<!--[\s\S]*?-->/g,'').replace(/^\s*\/\/.*$/gm,'');
t("no 'bankroll' anywhere user-visible", /bankroll/i.test(stripped), false);
t("no 'of roll'",                        /of roll/i.test(stripped), false);
t("no Kelly shown",                      /Kelly/.test(stripped), false);
t("no kelly_units rendered",             /kelly_units/.test(stripped), false);
t("no stake suggestion on the card",     /stakePct|stakeLabel/.test(stripped), false);
t("no 'biggest stake' sort",             /Biggest stake/.test(stripped), false);
t("stake field is not prefilled",        /stake:''/.test(html), true);

console.log("\nstill intact:");
t("4 sorts remain", (html.match(/SORTS: \[[\s\S]*?\n  \],/)[0].match(/key:'/g)||[]).length, 4);
t("why line still shows book count", whyText(CIN), "25 books say 67.4% · this price is 2.9% better than fair");
t("edge still on the card", /p\.edge_pct\.toFixed\(1\)\+'% edge'/.test(html), true);
console.log(failed?`\n${failed} FAILED`:"\nall passed");
process.exit(failed?1:0);
