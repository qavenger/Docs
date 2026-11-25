// p06_test.js — 在 Node 里检查第 06 篇 P03 命名空间的逻辑（不需要浏览器）
// 用法：node pages/tools/p06_test.js
const fs = require('fs'), path = require('path');
const LM = require('../llm_common.js');
const LM_WEIGHTS = require('../llm_weights.js');
const src = fs.readFileSync(path.join(__dirname, '..', '06_tokenizer_embedding.lab.js'), 'utf8');
const P03 = new Function('LM', 'LM_WEIGHTS', 'PipelineLab', 'PL', src.split('PipelineLab.mount(')[0] + '\nreturn P03;')(LM, LM_WEIGHTS, { mount() {} }, {});
const assert = (c, m) => { if (!c) { console.log('FAIL', m); process.exitCode = 1; } };

// 1. UTF-8
for (const s of P03.SENTS) assert(JSON.stringify(P03.strBytes(s)) === JSON.stringify(Array.from(new TextEncoder().encode(s))), 'utf8 ' + s);
console.log('utf8 ok; bytes of', P03.SENTS.map((s) => [...s].length + '/' + P03.strBytes(s).length).join(' '));

// 2. corpus + BPE
const rand = LM.rng(20251123);
const zh = [], en = [], u = [];
for (let i = 0; i < P03.NSENT; i++) { zh.push(LM.world.sentence(rand)); en.push(P03.enSentence(rand)); u.push(rand()); }
const zhHeld = [], enHeld = [];
for (let i = 0; i < P03.NHELD; i++) { zhHeld.push(LM.world.sentence(rand)); enHeld.push(P03.enSentence(rand)); }
const corpusOf = (mix) => zh.map((s, i) => (u[i] < mix ? en[i] + ' ' : s)).join('');
const held = [zhHeld.join(''), enHeld.join(' ') + ' '];
for (const mix of P03.MIXES) {
    const t0 = Date.now(), R = P03.trainBPE(corpusOf(mix), held, P03.MAXM);
    const ms = Date.now() - t0;
    const at = (k) => `k${k}: zh ${(R.heldChars[0] / R.hTok[0][Math.min(k, R.merges.length)]).toFixed(2)} en ${(R.heldChars[1] / R.hTok[1][Math.min(k, R.merges.length)]).toFixed(2)}`;
    console.log(`mix ${mix} train ${ms} ms merges ${R.merges.length} chunks ${R.words.length} |`, [0, 50, 100, 200, 400].map(at).join(' | '));
    if (mix === 0.1) {
        console.log('first 40 merges:', R.merges.slice(0, 40).map((m, r) => `${r}:${P03.segText(P03.segsOf(R.vocab[m.id]))}(${m.count})`).join(' '));
        // top pair at state k = merge k
        for (const k of [0, 1, 7, 33, 120, 250, 399]) {
            const st = P03.stateAt(R, k), top = P03.pairStats(st, R.counts)[0], m = R.merges[k];
            assert(top && top.a === m.a && top.b === m.b && top.n === m.count, 'top pair at ' + k);
            // the state must equal encoding each chunk word with k merges
            const words = R.words.slice(0, 30);
            words.forEach((w, i) => { const e = P03.encodeChunk(P03.strBytes(w), R, k); assert(JSON.stringify(e) === JSON.stringify(Array.from(st[i])), `state vs encode k=${k} ${w}`); });
        }
        // held-out counts == encode
        for (const k of [0, 60, 200]) { const e = P03.encode(held[0], R, k); assert(e.ids.length === R.hTok[0][k], 'held tok ' + k + ' ' + e.ids.length + ' vs ' + R.hTok[0][k]); }
        // round trip
        for (const s of P03.SENTS) for (const k of [0, 30, 80, 200, 400]) { const e = P03.encode(s, R, k); assert(P03.decode(e.ids, R) === s, 'roundtrip ' + s + ' k' + k); }
        for (const k of [20, 40, 60, 80, 120, 200]) { const e = P03.encode(P03.SENTS[0], R, k); console.log('k', k, e.ids.length, 'tokens:', e.ids.map((i) => P03.segText(P03.segsOf(R.vocab[i]))).join(' | ')); }
        const kinds = { part: 0, char: 0, multi: 0 }; R.merges.slice(0, 100).forEach((m) => kinds[m.kind] = (kinds[m.kind] || 0) + 1); console.log('kinds in first 100', kinds);
        console.log('sentence 3 k=80:', P03.encode(P03.SENTS[3], R, 80).ids.map((i) => P03.segText(P03.segsOf(R.vocab[i]))).join(' | '));
    }
}

// 3. word2vec
const sents = [], wr = LM.rng(5);
for (let i = 0; i < P03.W2V.nSent; i++) sents.push(P03.segmentWords(LM.world.sentence(wr)));
console.log('w2v sentences', sents.length, 'max len', Math.max(...sents.map((s) => s.length)), 'V', P03.LEX.V, P03.LEX.words.join(' '));
const tab = P03.unigramTable(sents, P03.LEX.V, P03.W2V.table);
const t1 = Date.now();
for (const win of P03.W2V.wins) {
    const pairs = P03.buildPairs(sents, win, LM.rng(100 + win)), M = P03.w2vInit(11 + win, P03.W2V.d);
    let L = 0; const chunk = 1000;
    while (M.done < P03.W2V.total) L = P03.w2vTrain(M, sents, pairs, tab.table, chunk, 5, P03.W2V.d);
    const Wn = P03.normRows(M.Win, P03.LEX.V, P03.W2V.d), acc = P03.analogyAcc(Wn, P03.W2V.d);
    console.log(`win ${win} pairs/epoch ${pairs.length} loss ${L.toFixed(3)} sep ${P03.separation(Wn, P03.W2V.d).toFixed(3)} analogy food ${acc.food.toFixed(2)} home ${acc.home.toFixed(2)} color ${acc.color.toFixed(2)} all ${acc.all.toFixed(2)} (${acc.ok}/${acc.n})`);
}
console.log('w2v total ms (6 models x 120k pairs)', Date.now() - t1);

// 4. base model
const base = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64);
console.log('base wte range', Math.min(...base.wte.data).toFixed(2), Math.max(...base.wte.data).toFixed(2), 'wpe range', Math.min(...base.wpe.data).toFixed(2), Math.max(...base.wpe.data).toFixed(2));

// 5. RoPE relative property
const qr = LM.rng(3), q = Float64Array.from({ length: 32 }, () => LM.randn(qr)), k = Float64Array.from({ length: 32 }, () => LM.randn(qr));
let md = 0; for (let D = -40; D <= 40; D++) { const s0 = P03.dot(P03.rope(q, 40, 1e4), P03.rope(k, 40 + D, 1e4)), s1 = P03.dot(P03.rope(q, 1000, 1e4), P03.rope(k, 1000 + D, 1e4)); md = Math.max(md, Math.abs(s0 - s1)); }
console.log('rope max diff between m=40 and m=1000 curves', md.toExponential(2));
