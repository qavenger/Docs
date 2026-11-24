// p12_train.js — 离线训练第 12 篇的两个模型（与页面同一份代码：P12.trainBatch / P12.lossStep / P12.createModel / P12.inflate），
// 数据并行：每个 worker 线程算一个子 batch 的梯度，主线程取平均 → 裁剪 → AdamW → EMA。权重（EMA，float16 base64）写进 12_video_generation.lab.js 的 <P12_W> 段。
// 用法（在 pages/ 下）：
//   node tools/p12_train.js image  [--steps 6000] [--B 64] [--lr 2e-3] [--workers 6]
//   node tools/p12_train.js video  [--steps 8000] [--B 32] [--lr 1e-3] [--workers 8] [--freeze]   （从图像模型「膨胀」初始化）
//   node tools/p12_train.js write                                                                  （把 tools/p12_ckpt_*.json 写进页面）
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');

if (!isMainThread) {
    const { LM, P12 } = require('./p12_load.js');
    Object.assign(P12.RECIPE, workerData.recipe || {});
    const m = P12.createModel(workerData.cfg, 1);
    const total = m.params.reduce((s, p) => s + p.size, 0);
    parentPort.on('message', (msg) => {
        let o = 0;
        for (const p of m.params) { p.data.set(msg.params.subarray(o, o + p.size)); o += p.size; p.grad = null; }
        const bt = P12.trainBatch(m, msg.B, LM.rng(msg.seed));
        const loss = P12.lossStep(m, bt);
        const g = new Float32Array(total); o = 0;
        for (const p of m.params) { if (p.grad) g.set(p.grad, o); o += p.size; }
        // 每个 t 区间的损失（画「损失 vs 噪声水平」用）：重新算一遍逐样本误差太贵，这里只回传 batch 的 t 与总损失
        parentPort.postMessage({ loss, g, ts: Array.from(bt.ts) }, [g.buffer]);
    });
    return;
}

const { LM, P12, file } = require('./p12_load.js');
const args = process.argv.slice(2), kind = args[0];
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true) : d; };
const TAG = opt('tag', ''), CK0 = (k) => path.join(__dirname, `p12_ckpt_${k}.json`), CK = (k) => path.join(__dirname, `p12_ckpt_${k}${TAG}.json`);
const CFG = { image: { d: 32, L: 2, H: 4, mlp: 4, video: false, pred: 'x0' }, video: { d: 32, L: 2, H: 4, mlp: 4, video: true, pred: 'x0' } };
if (opt('recipe', null)) Object.assign(P12.RECIPE, JSON.parse(opt('recipe', '{}')));

function writePage() {
    const W = {};
    for (const k of ['image', 'video']) if (fs.existsSync(CK0(k))) W[k] = JSON.parse(fs.readFileSync(CK0(k), 'utf8'));
    const src = fs.readFileSync(file, 'utf8');
    const line = (k) => { const w = W[k]; if (!w) return `    ${k}: null,`; return `    ${k}: { cfg: ${JSON.stringify(w.cfg)}, steps: ${w.steps}, batch: ${w.batch}, lr: ${w.lr}, init: ${JSON.stringify(w.init)}, sec: ${w.sec}, evals: ${JSON.stringify(w.evals)},\n        curve: ${JSON.stringify(w.curve)},\n        b64: '${w.b64}' },`; };
    const block = `// <P12_W> 离线训练好的权重（tools/p12_train.js 生成，请勿手改）：EMA 权重，float16 小端序 base64；curve = [步, 平均 loss]；evals = 训练中的采样评估\nconst P12_W = {\n${line('image')}\n${line('video')}\n};\n// </P12_W>`;
    const out = src.replace(/\/\/ <P12_W>[\s\S]*?\/\/ <\/P12_W>/, block);
    if (out === src && !src.includes(block)) throw new Error('marker block not found');
    fs.writeFileSync(file, out);
    console.log('wrote P12_W into', file, Object.keys(W).map((k) => `${k}: ${(W[k].b64.length / 1024).toFixed(0)} KB`).join(', '));
}
if (kind === 'write') { writePage(); process.exit(0); }
if (!CFG[kind]) { console.error('usage: node tools/p12_train.js image|video|write [...]'); process.exit(2); }

const steps = +opt('steps', kind === 'image' ? 6000 : 8000), B = +opt('B', kind === 'image' ? 64 : 32), lr = +opt('lr', kind === 'image' ? 2e-3 : 1e-3);
const nW = +opt('workers', 6), seed = +opt('seed', 1), freeze = !!opt('freeze', false), evalEvery = +opt('eval', 1000);
const cfg = Object.assign({}, CFG[kind], JSON.parse(opt('cfg', '{}')));
const m = P12.createModel(cfg, seed);
let init = 'random';
if (kind === 'video' && !opt('scratch', false)) {
    const ck = JSON.parse(fs.readFileSync(opt('from', CK0('image')), 'utf8')), img = P12.createModel(ck.cfg, 1);
    P12.loadParams(img, ck.b64); P12.inflate(m, img); init = `inflated from image model (${ck.steps} steps)`;
}
const trainable = m.params.filter((p) => !freeze || /^(b\d+\.t\.|pos\.t|cls|patch\.W)/.test(p.name) || /adaLN/.test(p.name));
const ema = P12.createModel(cfg, seed); P12.copyParams(ema, m);
const adam = new LM.AdamW(trainable, { lr, b1: 0.9, b2: 0.99, wd: 0 });
const total = m.params.reduce((s, p) => s + p.size, 0), flat = new Float32Array(total);
console.log(`${kind}: ${total} params (${trainable.reduce((s, p) => s + p.size, 0)} trainable), init ${init}, ${steps} steps × B ${B}, lr ${lr}, ${nW} workers`);

const workers = Array.from({ length: nW }, () => new Worker(__filename, { workerData: { cfg, recipe: P12.RECIPE } }));
const call = (w, msg) => new Promise((res) => { w.once('message', res); w.postMessage(msg); });
function evaluate(model) {
    if (!cfg.video) {
        const n = 64, x = P12.sampleAll(model, { B: n, K: 16, rand: LM.rng(99) });
        let one = 0, mass = 0; for (let i = 0; i < n; i++) { const s = P12.frameStats(x, i * 256); if (s.comps === 1) one++; mass += s.m; }
        return { one: +(one / n).toFixed(3), mass: +(mass / n).toFixed(2) };
    }
    const n = 16, out = {};
    for (const [name, o] of [['uncond', {}], ['i2v', { known: 1 }], ['ctx2', { known: 2 }]]) {
        const rr = LM.rng(99), cond = new Float32Array(n * P12.VD);
        for (let b = 0; b < n; b++) P12.renderVideo(P12.sampleBall(rr), cond, b * P12.VD);
        const x = P12.sampleAll(model, Object.assign({ B: n, K: 16, rand: rr, cond }, o));
        let one = 0, res = 0, nr = 0; for (let i = 0; i < n; i++) { const mm = P12.videoMetrics(x, i * P12.VD); one += mm.one; if (isFinite(mm.res)) { res += mm.res; nr++; } }
        out[name] = { one: +(one / n).toFixed(3), res: +(res / Math.max(1, nr)).toFixed(3) };
    }
    return out;
}
(async () => {
    const t0 = Date.now(), curve = [], evals = [];
    let acc = 0, na = 0;
    for (let it = 0; it < steps; it++) {
        let o = 0; for (const p of m.params) { flat.set(p.data, o); o += p.size; }
        const per = Math.ceil(B / nW), jobs = [];
        for (let k = 0; k < nW; k++) { const b = Math.min(per, B - k * per); if (b > 0) jobs.push(call(workers[k], { params: flat, seed: (seed * 1000003 + it * 7919 + k * 104729) >>> 0, B: b }).then((r) => Object.assign(r, { b }))); }
        const res = await Promise.all(jobs);
        m.params.forEach((p) => { p.grad = p.grad || new Float32Array(p.size); p.grad.fill(0); });
        let loss = 0;
        for (const r of res) {
            const wgt = r.b / B; loss += wgt * r.loss; o = 0;
            for (const p of m.params) { const g = p.grad; for (let i = 0; i < p.size; i++) g[i] += wgt * r.g[o + i]; o += p.size; }
        }
        LM.clipGradNorm(trainable, 1.0);
        adam.step(LM.cosineLR(it, { maxLR: lr, minLR: lr * 0.05, warmup: 200, total: steps }));
        P12.emaUpdate(ema, m, Math.min(0.999, (1 + it) / (10 + it)));
        acc += loss; na++;
        if ((it + 1) % 50 === 0) { curve.push([it + 1, +(acc / na).toFixed(4)]); acc = 0; na = 0; }
        if ((it + 1) % 250 === 0) console.log(`step ${it + 1} loss ${curve[curve.length - 1][1]}  ${((Date.now() - t0) / (it + 1)).toFixed(0)} ms/step`);
        if ((it + 1) % evalEvery === 0 || it + 1 === steps) {
            const e = evaluate(ema); evals.push([it + 1, e]);
            console.log(`  eval @${it + 1}: ${JSON.stringify(e)}`);
            const ck = { cfg, steps: it + 1, batch: B, lr, init, sec: Math.round((Date.now() - t0) / 1000), evals, curve, b64: P12.exportParams(ema) };
            fs.writeFileSync(CK(kind + (it + 1 === steps ? '' : '_partial')), JSON.stringify(ck));
            if (opt('keep', false)) fs.writeFileSync(CK(`${kind}_${it + 1}`), JSON.stringify(ck));
        }
    }
    workers.forEach((w) => w.terminate());
    console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s → ${CK(kind)}`);
})();
