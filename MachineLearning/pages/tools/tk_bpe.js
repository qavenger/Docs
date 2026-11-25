// tk_bpe.js — 训练第 19–21 篇用来「数 token」的字节级 BPE（16k 词表；与第 06 篇同一算法，换成增量更新以便跑到 16k 条规则）。
// 语料 = 本系列全部讲义源文件（中文正文 + JS / C++ 代码 + 英文术语，约 4 MB）。输出 tools/tk_bpe.json：{ merges: base64(Uint16 a,b 对), K }
// 用法（在 pages/ 下）：node tools/tk_bpe.js [K=16128]
const fs = require('fs');
const path = require('path');
const pages = path.join(__dirname, '..');
const K = +(process.argv[2] || 16128);
const files = fs.readdirSync(pages).filter((f) => /\.lab\.js$/.test(f) || ['llm_common.js', 'agent_common.js', 'agent_cpp.js'].includes(f)).map((f) => path.join(pages, f))
    .concat(fs.readdirSync(path.join(pages, 'cpp')).map((f) => path.join(pages, 'cpp', f)));
const text = files.map((f) => fs.readFileSync(f, 'utf8')).join('\n');
// 预切分（cl100k 风格）：单词（带前导空格）、1–3 位数字、标点串、空白
const PRE = /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
const t0 = Date.now();
const freq = new Map();
for (const m of text.matchAll(PRE)) { const w = m[0]; freq.set(w, (freq.get(w) || 0) + 1); }
const words = [], counts = [];
for (const [w, c] of freq) { words.push(Array.from(Buffer.from(w, 'utf8'))); counts.push(c); }
console.log(`${(text.length / 1e6).toFixed(2)} M chars · ${freq.size} unique chunks · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
// 相邻对计数 + 「哪些块里有这个对」的索引 + 惰性删除的最大堆
const key = (a, b) => a * 65536 + b;
const pc = new Map(), where = new Map();
const addPairs = (wi, sign) => { const w = words[wi], c = counts[wi] * sign; for (let j = 0; j + 1 < w.length; j++) { const k = key(w[j], w[j + 1]); pc.set(k, (pc.get(k) || 0) + c); if (sign > 0) { let s = where.get(k); if (!s) where.set(k, (s = new Set())); s.add(wi); } } };
for (let i = 0; i < words.length; i++) addPairs(i, 1);
const heap = [];   // [count, key]
const hpush = (c, k) => { heap.push([c, k]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] > heap[i][0] || (heap[p][0] === heap[i][0] && heap[p][1] < heap[i][1])) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
const hpop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; const better = (a, b) => heap[a][0] > heap[b][0] || (heap[a][0] === heap[b][0] && heap[a][1] < heap[b][1]); if (l < heap.length && better(l, m)) m = l; if (r < heap.length && better(r, m)) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
for (const [k, c] of pc) hpush(c, k);
const merges = new Uint16Array(2 * K);
for (let r = 0; r < K; r++) {
    let best = null;
    while (heap.length) { const [c, k] = hpop(); if (pc.get(k) === c && c > 0) { best = k; break; } }
    if (best === null) { console.log('ran out of pairs at', r); break; }
    const a = Math.floor(best / 65536), b = best % 65536, id = 256 + r;
    merges[2 * r] = a; merges[2 * r + 1] = b;
    const touched = new Set();
    for (const wi of where.get(best) || []) {
        const w = words[wi];
        let has = false; for (let j = 0; j + 1 < w.length; j++) if (w[j] === a && w[j + 1] === b) { has = true; break; }
        if (!has) continue;
        addPairs(wi, -1);
        const out = []; for (let j = 0; j < w.length; j++) { if (j + 1 < w.length && w[j] === a && w[j + 1] === b) { out.push(id); j++; } else out.push(w[j]); }
        words[wi] = out;
        addPairs(wi, 1);
        for (let j = 0; j + 1 < out.length; j++) touched.add(key(out[j], out[j + 1]));
    }
    where.delete(best); pc.delete(best);
    for (const k of touched) { const c = pc.get(k); if (c > 0) hpush(c, k); }
    if ((r + 1) % 2000 === 0) console.log(`${r + 1} merges · ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
const total = words.reduce((s, w, i) => s + w.length * counts[i], 0);
console.log(`done: ${K} merges · corpus ${text.length} chars → ${total} tokens (${(text.length / total).toFixed(2)} chars/token) · ${((Date.now() - t0) / 1000).toFixed(0)} s`);
fs.writeFileSync(path.join(__dirname, 'tk_bpe.json'), JSON.stringify({ K, merges: Buffer.from(merges.buffer).toString('base64'), pre: PRE.source }));
