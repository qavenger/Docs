// p06_bpe.js — 第 06 篇原型：在玩具世界语料上训练字节级 BPE，看合并顺序、压缩率
// 用法：node pages/tools/p06_bpe.js [enMix=0] [nSent=2000] [merges=400]
const LM = require('../llm_common.js');

const enMix = +(process.argv[2] || 0), nSent = +(process.argv[3] || 2000), K = +(process.argv[4] || 400);

const EN = [
    { name: 'cat', food: 'fish', home: 'house', color: 'white' },
    { name: 'dog', food: 'bones', home: 'yard', color: 'yellow' },
    { name: 'rabbit', food: 'carrots', home: 'meadow', color: 'gray' },
    { name: 'bird', food: 'bugs', home: 'tree', color: 'blue' },
    { name: 'panda', food: 'bamboo', home: 'mountains', color: 'black and white' },
    { name: 'monkey', food: 'bananas', home: 'forest', color: 'brown' },
    { name: 'sheep', food: 'grass', home: 'hillside', color: 'white' },
    { name: 'tiger', food: 'meat', home: 'forest', color: 'orange' },
];
const ENT = [
    (a) => `The ${a.name} likes to eat ${a.food}.`,
    (a) => `The ${a.name} lives in the ${a.home}.`,
    (a) => `The ${a.name} is ${a.color}.`,
    (a, b) => `The ${a.name} and the ${b.name} are good friends.`,
    (a) => `The ${a.name} likes ${a.food}, so it eats ${a.food} every day.`,
];
function enSentence(rand) {
    const a = EN[Math.floor(rand() * EN.length)]; let b = EN[Math.floor(rand() * EN.length)]; if (b === a) b = EN[(EN.indexOf(a) + 1) % EN.length];
    return ENT[Math.floor(rand() * ENT.length)](a, b);
}
function corpus(rand, n, mix) {
    let s = '';
    for (let i = 0; i < n; i++) s += rand() < mix ? enSentence(rand) + ' ' : LM.world.sentence(rand);
    return s;
}
const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const enc = new TextEncoder(), dec = new TextDecoder('utf-8', { fatal: true });
function chunksOf(text) {
    const m = new Map();
    for (const w of text.match(PAT) || []) m.set(w, (m.get(w) || 0) + 1);
    const chunks = [], counts = [];
    for (const [w, c] of m) { chunks.push(Array.from(enc.encode(w))); counts.push(c); }
    return { chunks, counts, words: [...m.keys()] };
}
function trainBPE(text, K) {
    const { chunks, counts } = chunksOf(text);
    const vocab = []; for (let i = 0; i < 256; i++) vocab.push([i]);
    const merges = [], ntok = [];
    let tot = 0; chunks.forEach((c, i) => { tot += c.length * counts[i]; }); ntok.push(tot);
    for (let k = 0; k < K; k++) {
        const stats = new Map();
        chunks.forEach((c, i) => { for (let j = 0; j + 1 < c.length; j++) { const key = c[j] * 4096 + c[j + 1]; stats.set(key, (stats.get(key) || 0) + counts[i]); } });
        let best = -1, bc = 0; for (const [key, v] of stats) if (v > bc) { bc = v; best = key; }
        if (best < 0) break;
        const a = Math.floor(best / 4096), b = best % 4096, id = 256 + k;
        for (let i = 0; i < chunks.length; i++) {
            const c = chunks[i], out = [];
            for (let j = 0; j < c.length; j++) { if (j + 1 < c.length && c[j] === a && c[j + 1] === b) { out.push(id); j++; } else out.push(c[j]); }
            chunks[i] = out;
        }
        merges.push({ a, b, id, count: bc });
        vocab.push(vocab[a].concat(vocab[b]));
        tot = 0; chunks.forEach((c, i) => { tot += c.length * counts[i]; }); ntok.push(tot);
    }
    return { merges, vocab, ntok, chunks, counts };
}
function label(bytes) {
    try { return JSON.stringify(dec.decode(Uint8Array.from(bytes))); } catch (e) {
        // partial: show valid prefix + hex
        return '<' + bytes.map((x) => x.toString(16).toUpperCase().padStart(2, '0')).join(' ') + '>';
    }
}
const text = corpus(LM.rng(2024), nSent, enMix);
const nbytes = enc.encode(text).length;
console.log('chars', [...text].length, 'bytes', nbytes);
const { chunks } = chunksOf(text); console.log('unique chunks', chunks.length);
const t0 = Date.now();
const R = trainBPE(text, K);
console.log('train ms', Date.now() - t0, 'merges', R.merges.length);
R.merges.slice(0, 80).forEach((m, k) => console.log(k, label(R.vocab[m.a]), '+', label(R.vocab[m.b]), '->', label(R.vocab[m.id]), m.count));
console.log('...');
[100, 150, 200, 250, 300, 350, 399].forEach((k) => { if (R.merges[k]) { const m = R.merges[k]; console.log(k, label(R.vocab[m.a]), '+', label(R.vocab[m.b]), '->', label(R.vocab[m.id]), m.count); } });
[0, 10, 20, 50, 100, 200, 300, 400].forEach((k) => { if (R.ntok[k] != null) console.log('k', k, 'tokens', R.ntok[k], 'bytes/token', (nbytes / R.ntok[k]).toFixed(2)); });
