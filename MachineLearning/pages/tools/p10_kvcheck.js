// KV-cache 贪心解码（P07.decode）与完整前向（P07.llavaFwd）逐位一致性检查：随机扰动过的解码器、页面配置 T = 32
const { LM, P07 } = require('./p10_load.js');
const { AG } = LM;
const rand = LM.rng(11), vit = P07.bakedViT(), mm = P07.createLLaVA(P07.LLAVA_CFG, 5);
mm.lm.params.forEach((p) => { for (let i = 0; i < p.size; i++) p.data[i] += 0.3 * (rand() - 0.5); });   // 让分布不那么平
let worst = 0, toks = 0;
for (let trial = 0; trial < 8; trial++) {
    const sc = [P07.sampleScene(rand)], vis = P07.visFeatures(vit, sc, 0), pv = AG.noGrad(() => P07.projFwd(mm.proj, vis));
    const g = P07.decode(mm.lm, pv.data, 16, P07.MAXCAP + 1, { probs: true });
    const txt = new Int32Array(16).fill(P07.PAD); [P07.BOS].concat(g.ids).slice(0, 16).forEach((t, k) => { txt[k] = t; });
    const out = AG.noGrad(() => P07.llavaFwd(mm, vis, txt, 1, {})), V = P07.LLAVA_CFG.V;
    g.probs.forEach((p, k) => { const row = LM.softmax(out.logits.data, new Float32Array(V), 1, (16 + k) * V, V); for (let v = 0; v < V; v++) worst = Math.max(worst, Math.abs(p[v] - row[v])); toks++; });
}
console.log(`KV-cache decode vs full forward: ${toks} positions, max |Δp| = ${worst.toExponential(2)}`);
