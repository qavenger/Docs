// 一次性：请求生成再加两类说法——只读工具的「提问式」（…是多少 / 查一下… / how many…），以及日期参数（昨天 / 上周 / 最近 7 天）
const fs = require('fs');
const path = require('path');
const p = path.join(__dirname, '..', 'tools_catalog.js');
let s = fs.readFileSync(p, 'utf8');
function rep(a, b) { const n = s.split(a).length - 1; if (n !== 1) throw new Error(`expected 1 × «${a.slice(0, 70)}», found ${n}`); s = s.replace(a, () => b); }
rep("    const NO_BA = new Set(", `    // 只读工具的提问式说法（查数据、看状态、列清单的请求常常是问句）
    const ZH_Q = [['{app}里的{o}现在是多少{x}', '{o}现在是多少{x}'], ['查一下{app}的{o}{x}', '查一下{o}{x}'], ['{app}里有多少{o}', '有多少{o}'], ['{o}是什么情况{x}', '{o}是什么情况{x}'], ['看看{app}里的{o}{x}', '看看{o}{x}']];
    const ZH_QP = [['{app}里{v}是多少{x}', '{v}是多少{x}'], ['查一下{app}的{v}{x}', '查一下{v}{x}'], ['{v}怎么样了{x}', '{v}怎么样了{x}']];
    const EN_Q = [['how many {o} are there in {app}', 'how many {o} are there'], ['what is the current {o} in {app}{x}', 'what is the current {o}{x}'], ['show me the {o}{x}', 'show me the {o}{x}']];
    const EN_QP = [['what is the {v} in {app}{x}', 'what is the {v}{x}'], ['check the {v}{x}', 'check the {v}{x}']];
    const NO_BA = new Set(`);
rep("        if (has('content') || has('body') || has('text') || has('query')) opts.push(", "        if (has('start_date') || has('cohort_start') || has('date')) opts.push(zh ? pick(['，昨天的', '，上周的', '，最近 7 天', '，2024-06-01 到 2024-06-07', '，这个月']) : pick([' for yesterday', ' for last week', ' over the last 7 days', ' from 2024-06-01 to 2024-06-07']));\n        if (has('content') || has('body') || has('text') || has('query')) opts.push(");
rep("            if (!o || (t.phrase && rand() < 0.45)) {", `            if (t.annotations.readOnlyHint && rand() < 0.3) {   // 只读工具：三成用提问式
                text = o && !t.phrase ? fill(pick(zh ? ZH_Q : EN_Q), appName, v, o, x) : fill(pick(zh ? ZH_QP : EN_QP), appName, v, '', x);
            } else if (!o || (t.phrase && rand() < 0.45)) {`);
fs.writeFileSync(p, s);
console.log('ok');
