// CLIP 训练实验：node p07_clip_exp.js N=8 tau=0.1 lr=3e-3 steps=1500 d=32 Lv=1 Lt=1 ff=128
const { LM, P07 } = require('./p07_load.js');
const { AG } = LM;
const A = Object.fromEntries(process.argv.slice(2).map((s) => s.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const N = A.N ?? 8, tau = A.tau ?? 0.1, lr = A.lr ?? 3e-3, steps = A.steps ?? 1500, d = A.d ?? 32, Lv = A.Lv ?? 1, Lt = A.Lt ?? 1, ff = A.ff ?? 4 * d, pS = A.pS ?? 0.4, wd = A.wd ?? 0.1;
if (A.snap != null) P07.DATA.snap = A.snap; if (A.r1a) P07.DATA.r1 = [A.r1a, A.r1b]; if (A.r2a) P07.DATA.r2 = [A.r2a, A.r2b]; if (A.jit != null) P07.DATA.jit = A.jit;
const T = P07.MAXCAP + 2;
const cfg = { img: { d, L: Lv, H: 4, ff, e: 32, init: A.init || 'clip' }, txt: { V: P07.vocab.size, T, d, L: Lt, H: 4, ff, e: 32, embStd: A.es, init: A.init || 'clip' } };
if (A.sq) P07.DATA.sq = A.sq;
const clip = P07.createCLIP(cfg, A.seed ?? 1);
const opt = new LM.AdamW(clip.params, { lr, wd, b2: 0.98 });
const rand = LM.rng(1234 + (A.ds ?? 0));
function hardBatch(n, r) {   // 一半随机场景 + 每个配一个「只改一个图形的形状」的困难负样本
    const seen = new Set(), out = [];
    while (out.length < n) {
        const sc = P07.sampleScene(r, { pSingle: pS }); if (seen.has(sc.caption)) continue;
        const j = Math.floor(r() * sc.objs.length), ob = sc.objs.map((o) => Object.assign({}, o)), u = r();
        if (A.mix && sc.objs.length === 2 && u < 0.3) { const t = ob[0]; ob[0] = Object.assign({}, ob[1], { x: t.x, y: t.y, r: t.r }); ob[1] = Object.assign({}, t, { x: ob[1].x, y: ob[1].y, r: ob[1].r }); }
        else if (A.mix && u < 0.6) ob[j].c = (ob[j].c + 1 + Math.floor(r() * 3)) % 4;
        else ob[j].s = (ob[j].s + 1 + Math.floor(r() * 2)) % 3;
        const sc2 = { objs: ob, single: sc.single, caption: P07.captionOf(ob), cls: sc.single ? ob[0].c * 3 + ob[0].s : -1, seed: Math.floor(r() * 2 ** 31) };
        if (seen.has(sc2.caption)) continue;
        seen.add(sc.caption); seen.add(sc2.caption); out.push(sc, sc2);
    }
    return out.slice(0, n);
}
function batch(n, r) { if (A.hard) return hardBatch(n, r); const seen = new Set(), out = []; while (out.length < n) { const sc = P07.sampleScene(r, { pSingle: pS }); if (seen.has(sc.caption)) continue; seen.add(sc.caption); out.push(sc); } return out; }
// eval data
const er = LM.rng(999), testS = [], testD = [];
for (let k = 0; k < P07.NCLS; k++) for (let i = 0; i < 10; i++) testS.push(P07.sampleScene(er, { single: true, cls: k }));
for (let i = 0; i < 120; i++) testD.push(P07.sampleScene(er, { single: false }));
const allCaps = []; for (let k = 0; k < P07.NCLS; k++) allCaps.push(P07.captionOf([{ c: Math.floor(k / 3), s: k % 3 }]));
for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) allCaps.push(P07.captionOf([{ c: Math.floor(a / 3), s: a % 3 }, { c: Math.floor(b / 3), s: b % 3 }]));
const swap = (sc) => P07.captionOf([sc.objs[1], sc.objs[0]]);
function embT(caps) { return AG.noGrad(() => { const { ids, eos } = P07.textBatch(caps, T); return P07.textFwd(clip.txt, ids, eos, caps.length).emb; }); }
function embI(scs) { return AG.noGrad(() => P07.vitFwd(clip.img, P07.imageBatch(scs).X, scs.length).emb); }
const dot = (a, i, b, j, e) => { let s = 0; for (let k = 0; k < e; k++) s += a.data[i * e + k] * b.data[j * e + k]; return s; };
function evaluate() {
    const E = 32, tc = embT(allCaps), is = embI(testS), id = embI(testD);
    let zs = 0, zc = 0, zh = 0; testS.forEach((sc, i) => { let b = 0; for (let k = 1; k < 12; k++) if (dot(is, i, tc, k, E) > dot(is, i, tc, b, E)) b = k; if (b === sc.cls) zs++; if (Math.floor(b / 3) === Math.floor(sc.cls / 3)) zc++; if (b % 3 === sc.cls % 3) zh++; });
    let ret = 0, bind = 0; const all = testS.concat(testD), ia = [is, id];
    const capIdx = new Map(allCaps.map((c, i) => [c, i]));
    all.forEach((sc, i) => { const src = i < testS.length ? is : id, ii = i < testS.length ? i : i - testS.length; let b = 0; for (let k = 1; k < allCaps.length; k++) if (dot(src, ii, tc, k, E) > dot(src, ii, tc, b, E)) b = k; if (b === capIdx.get(sc.caption)) ret++; });
    testD.forEach((sc, i) => { if (sc.caption === swap(sc)) { bind += 0.5; return; } const c = capIdx.get(sc.caption), s = capIdx.get(swap(sc)); if (dot(id, i, tc, c, E) > dot(id, i, tc, s, E)) bind++; });
    return { zc: zc / testS.length, zh: zh / testS.length, zs: zs / testS.length, ret: ret / all.length, bind: bind / testD.length };
}
{ const e = embT(['一个红色的圆形', '一个红色的方形', '一个蓝色的圆形']); const ii = embI([P07.sampleScene(LM.rng(5), { single: true, cls: 0 }), P07.sampleScene(LM.rng(6), { single: true, cls: 4 })]); console.log('init img cos', dot(ii, 0, ii, 1, 32).toFixed(3)); console.log('init cos(红圆,红方)', dot(e, 0, e, 1, 32).toFixed(3), 'cos(红圆,蓝圆)', dot(e, 0, e, 2, 32).toFixed(3)); }
const t0 = Date.now(); let ema = null;
for (let it = 1; it <= steps; it++) {
    const sc = batch(N, rand);
    const { X } = P07.imageBatch(sc), { ids, eos } = P07.textBatch(sc.map((s) => s.caption), T);
    opt.zeroGrad();
    const I = P07.vitFwd(clip.img, X, N).emb, Tx = P07.textFwd(clip.txt, ids, eos, N).emb;
    const r = P07.clipLoss(I, Tx, tau); AG.backward(r.loss); LM.clipGradNorm(clip.params, 1);
    const l = A.cos ? LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 50, total: steps }) : (it < 50 ? lr * it / 50 : lr);
    opt.step(l);
    ema = ema == null ? r.loss.item() : 0.98 * ema + 0.02 * r.loss.item();
    if (it % (A.every ?? 250) === 0 || it === steps) { const e = evaluate(); console.log(`it ${it} loss ${ema.toFixed(3)} zs ${(e.zs * 100).toFixed(1)}% (col ${(e.zc * 100).toFixed(0)} shp ${(e.zh * 100).toFixed(0)}) ret ${(e.ret * 100).toFixed(1)}% bind ${(e.bind * 100).toFixed(1)}%  ${((Date.now() - t0) / it).toFixed(1)} ms/it`); }
}
module.exports = { clip };
