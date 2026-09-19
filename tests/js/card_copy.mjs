// The card must say what to bet, not prove it did its homework.
// Run with: node tests/js/card_copy.mjs
//
// Every version of this card before 20 September led with the evidence:
//   "Fair value 53.0% from sharp book across 24 books; best available price
//    implies 50.5%. Cross-book disagreement 1.1%."
//
// True, and nobody asked. A person opening a betting app wants to know what
// to bet, where, and at what price. The evidence is not deleted -- it lives
// behind "where does this number come from?" for the one reader in fifty who
// wants it. It is just no longer in front of the other forty-nine.
//
// Extracted from the template so the shipped copy is what gets checked.

function winProb(p){ return p.blended_prob ?? p.fair_prob ?? 0 }
const fmtOdds = (p) => {return p>0?'+'+p:''+p}
const instruction = (p) => {
    const book = (p.best_book||'').replace(/_/g,' ')
                    .replace(/\b\w/g, c => c.toUpperCase());
    return `Take ${p.description} at ${fmtOdds(p.best_price)} on ${book}`;
  }
const reason = (p) => {
    const books = (()=>{ try{ return JSON.parse(p.prob_components||'{}').book_count || 0 }
                         catch(e){ return 0 } })();
    const pct = Math.round(winProb(p)*100);
    const book = (p.best_book||'').replace(/_/g,' ')
                    .replace(/\b\w/g, c => c.toUpperCase());
    const who = books ? `${books} books say` : 'The market says';
    const price = Number(p.best_price);
    const pays = price === 100 ? 'is paying even money'
               : price > 0 ? `is paying +${price}`
               : `has it at ${price}`;
    return `${who} this hits ${pct}% of the time. ${book} ${pays}.`;
  }
const picks = [
 {description:"Over 44.5", best_price:-102, best_book:"lowvig", blended_prob:0.530,
  prob_components:JSON.stringify({book_count:24})},
 {description:"CHI -3", best_price:110, best_book:"lowvig", blended_prob:0.500,
  prob_components:JSON.stringify({book_count:15})},
 {description:"LV +7", best_price:-108, best_book:"fanduel", blended_prob:0.537,
  prob_components:JSON.stringify({book_count:21})},
 {description:"Christian McCaffrey over 4.5 receptions", best_price:-105,
  best_book:"betrivers", blended_prob:0.536, prob_components:JSON.stringify({book_count:4})},
 {description:"Deshaun Watson over 17.5 completions", best_price:100,
  best_book:"betmgm", blended_prob:0.525, prob_components:JSON.stringify({book_count:4})},
 {description:"NO +8.5", best_price:105, best_book:"lowvig", blended_prob:0.506,
  prob_components:null},
];
console.log("THE CARD, for tonight's real picks:\n");
for (const p of picks) console.log(`  ${instruction(p)}\n  ${reason(p)}\n`);

let f=0; const t=(n,c)=>{ if(!c){f++;console.log("  FAIL "+n)} };
t("no glossary vocabulary leaks in", !picks.some(p =>
  /fair value|implies|disagreement|edge|tier|consensus/i.test(instruction(p)+reason(p))));
t("every instruction starts with Take", picks.every(p => instruction(p).startsWith("Take ")));
t("book names are capitalised", instruction(picks[0]).includes("Lowvig"));
t("even money is said, not +100", reason(picks[4]).includes("even money"));
t("missing book_count degrades", reason(picks[5]).startsWith("The market says"));
console.log(f?`${f} FAILED`:"all checks passed");
process.exit(f?1:0);
