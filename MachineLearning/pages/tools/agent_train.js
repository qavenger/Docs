// agent_train.js — 离线训练第 16–18 篇共用的「会用工具的迷你 GPT」（与页面同一份代码：llm_common.js 的 LM.gpt + agent_common.js 的数据与 Harness）。
// 数据并行：每个 worker 线程算一个子 batch 的梯度，主线程取平均 → 裁剪 → AdamW → EMA。权重（EMA，float16 base64）写进 pages/agent_weights.js。
// 用法（在 pages/ 下）：node tools/agent_train.js [--steps 6000] [--workers 8] [--tag pilot] [--eval 500]
//                     node tools/agent_train.js write [--tag ...]      （把 tools/agent_ckpt<tag>.json 写进 agent_weights.js）
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
global.LM = LM;
const AGT = require('../agent_common.js');
const CFG = { V: AGT.V, T: 192, d: 64, L: 3, H: 4, ff: 256 };
const QA = { B: 32, T: 48 }, HOUSE = { B: 16, T: 192 };

if (!isMainThread) {
    const m = LM.gpt.create(CFG, 1), total = m.params.reduce((s, p) => s + p.size, 0);
    parentPort.on('message', (msg) => {
        let o = 0;
        for (const p of m.params) { p.data.set(msg.params.subarray(o, o + p.size)); o += p.size; p.grad = null; }
        const rand = LM.rng(msg.seed), qa = [], hs = [];
        for (let i = 0; i < msg.nqa; i++) qa.push(AGT.sample(rand, { qa: 1 }));
        for (let i = 0; i < msg.nh; i++) hs.push(AGT.sample(rand, { qa: 0 }));
        const g = new Float32Array(total), add = (scale) => { let k = 0; for (const p of m.params) { if (p.grad) for (let i = 0; i < p.size; i++) g[k + i] += scale * p.grad[i]; k += p.size; p.grad = null; } };
        const b1 = AGT.batch(qa, QA.T), out1 = LM.gpt.forward(m, b1.x, msg.nqa, QA.T, { targets: b1.y }); LM.AG.backward(out1.loss); add(0.5);
        const b2 = AGT.batch(hs, HOUSE.T), out2 = LM.gpt.forward(m, b2.x, msg.nh, HOUSE.T, { targets: b2.y }); LM.AG.backward(out2.loss); add(0.5);
        parentPort.postMessage({ lqa: out1.loss.item(), lh: out2.loss.item(), g }, [g.buffer]);
    });
    return;
}

const args = process.argv.slice(2), kind = args[0] && !args[0].startsWith('--') ? args[0] : 'train';
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const TAG = opt('tag', ''), CK = path.join(__dirname, `agent_ckpt${TAG}.json`), OUT = path.join(__dirname, '..', 'agent_weights.js');

/** 评估（贪心解码）：单轮问答按类型 × 可用工具，房子任务按类型；与页面里 Harness 同一份 runEpisode */
function evaluate(model, n) {
    const rr = LM.rng(4242), res = {};
    const qa = (kind, tools) => {
        let ok = 0, calls = 0;
        for (let i = 0; i < n; i++) {
            const task = AGT.qaTask(rr, kind), world = AGT.newWorld(rr), want = AGT.qaExpert(task, tools, world).filter((t) => t.role === 'bot').pop().text.replace('<eos>', '');
            const r = AGT.runEpisode(model, AGT.sysText(tools), task.q, AGT.qaEnv(tools, world), { greedy: true, maxCalls: 3 });
            if (r.ok && r.answer === want) ok++; calls += r.calls;
        }
        return { acc: +(ok / n).toFixed(3), calls: +(calls / n).toFixed(2) };
    };
    for (const k of ['add', 'sub', 'mul']) { res[k + '+tool'] = qa(k, ['算', '查']); res[k + '-tool'] = qa(k, []); }
    res['fact+tool'] = qa('fact', ['算', '查']); res['where+tool'] = qa('where', ['算', '查']); res['where-tool'] = qa('where', []);
    for (const k of ['find', 'move', 'list']) {
        let ok = 0, steps = 0;
        for (let i = 0; i < n; i++) { const task = AGT.houseTask(rr, k), H = AGT.newHouse(rr), r = AGT.runHouse(model, task, H, { greedy: true, maxCalls: 20 }); if (r.success) ok++; steps += r.calls; }
        res[k] = { succ: +(ok / n).toFixed(3), calls: +(steps / n).toFixed(1) };
    }
    return res;
}

function writePage() {
    const ck = JSON.parse(fs.readFileSync(CK, 'utf8'));
    const body = `// agent_weights.js — 由 tools/agent_train.js 用 llm_common.js + agent_common.js 离线训练生成，请勿手改。
// 第 16–18 篇共用的「会用工具的迷你 GPT」：权重 = EMA，全部参数按 model.params 顺序的 float16 小端序，base64 编码。
const AGT_W = {
    main: { cfg: ${JSON.stringify(ck.cfg)}, steps: ${ck.steps}, batch: ${JSON.stringify(ck.batch)}, lr: ${ck.lr}, sec: ${ck.sec}, workers: ${ck.workers},
        curve: ${JSON.stringify(ck.curve)},
        evals: ${JSON.stringify(ck.evals)},
        b64: '${ck.b64}' },
};
if (typeof module !== 'undefined' && module.exports) module.exports = AGT_W;
`;
    fs.writeFileSync(OUT, body);
    console.log('wrote', OUT, (ck.b64.length / 1024).toFixed(0), 'KB');
}
if (kind === 'write') { writePage(); process.exit(0); }

const steps = +opt('steps', 6000), nW = +opt('workers', 8), lr = +opt('lr', 2e-3), evalEvery = +opt('eval', 500), seed = +opt('seed', 1);
const m = LM.gpt.create(CFG, seed), ema = LM.gpt.create(CFG, seed);
const total = m.params.reduce((s, p) => s + p.size, 0), flat = new Float32Array(total);
const adam = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.95, wd: 0.05 });
console.log(`agent model: ${total} params, cfg ${JSON.stringify(CFG)}, ${steps} steps × (QA ${QA.B}×${QA.T} + house ${HOUSE.B}×${HOUSE.T}), lr ${lr}, ${nW} workers`);
const workers = Array.from({ length: nW }, () => new Worker(__filename));
const call = (w, msg) => new Promise((res) => { w.once('message', res); w.postMessage(msg); });
(async () => {
    const t0 = Date.now(), curve = [], evals = [];
    let aq = 0, ah = 0, na = 0;
    for (let it = 0; it < steps; it++) {
        let o = 0; for (const p of m.params) { flat.set(p.data, o); o += p.size; }
        const res = await Promise.all(workers.map((w, k) => call(w, { params: flat, seed: (seed * 1000003 + it * 7919 + k * 104729) >>> 0, nqa: QA.B / nW, nh: HOUSE.B / nW })));
        m.params.forEach((p) => { p.grad = p.grad || new Float32Array(p.size); p.grad.fill(0); });
        let lq = 0, lh = 0;
        for (const r of res) { o = 0; for (const p of m.params) { const g = p.grad; for (let i = 0; i < p.size; i++) g[i] += r.g[o + i] / nW; o += p.size; } lq += r.lqa / nW; lh += r.lh / nW; }
        LM.clipGradNorm(m.params, 1.0);
        adam.step(LM.cosineLR(it, { maxLR: lr, minLR: lr * 0.05, warmup: 200, total: steps }));
        const dec = Math.min(0.999, (1 + it) / (10 + it));
        for (let k = 0; k < m.params.length; k++) { const a = ema.params[k].data, b = m.params[k].data; for (let i = 0; i < a.length; i++) a[i] = dec * a[i] + (1 - dec) * b[i]; }
        aq += lq; ah += lh; na++;
        if ((it + 1) % 50 === 0) { curve.push([it + 1, +(aq / na).toFixed(4), +(ah / na).toFixed(4)]); aq = ah = na = 0; }
        if ((it + 1) % 250 === 0) { const c = curve[curve.length - 1]; console.log(`step ${it + 1}  loss qa ${c[1]}  house ${c[2]}  ${((Date.now() - t0) / (it + 1)).toFixed(0)} ms/step`); }
        if ((it + 1) % evalEvery === 0 || it + 1 === steps) {
            const e = evaluate(ema, it + 1 === steps ? 200 : 60); evals.push([it + 1, e]);
            console.log(`  eval @${it + 1}: ${JSON.stringify(e)}`);
            fs.writeFileSync(CK, JSON.stringify({ cfg: CFG, steps: it + 1, batch: { qa: QA, house: HOUSE }, lr, sec: Math.round((Date.now() - t0) / 1000), workers: nW, curve, evals, b64: LM.gpt.exportWeights(ema) }));
        }
    }
    workers.forEach((w) => w.terminate());
    console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s → ${CK}`);
})();
