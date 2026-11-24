// 在页面里把 CLIP 和 LLaVA 都训练到终点，然后导出每张预览（--eval 用；输出 { id: dataURL }）
PipelineLab.setParam('spf', 4);
const S = PipelineLab.S, D = PipelineLab._dbg, sleep = (ms) => new Promise((r) => setTimeout(r, ms));
document.querySelector('#step-train').scrollIntoView();
const t0 = performance.now();
while (!(S.cr && S.cr.tr.it >= 2000 && S.cr.queue.length === 0 && S.lr && S.lr.tr.it >= 3000 && S.lr.queue.length === 0) && performance.now() - t0 < 400000) await sleep(500);
const out = { _it: [S.cr.tr.it, S.lr && S.lr.tr.it], _ms: Math.round(performance.now() - t0) };
for (const id of D.previews.keys()) {
    const pv = D.previews.get(id); pv.canvas.scrollIntoView({ block: 'center' }); await sleep(250);
    try { pv.render(pv.canvas, S, PipelineLab); out[id] = pv.canvas.toDataURL('image/png'); } catch (e) { out[id] = 'ERR ' + e.message; }
}
return out;
