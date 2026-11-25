// p03_unet_act.js — 第 03 篇 Step 08：U-Net 用 ReLU 还是 SiLU？训练 1200 步后的 PSNR，以及把解码器那一级（h₂）置零后的 PSNR（看粗尺度那条路有没有在用）
const { P08, LM } = require('./p03_proto.js'), { AG } = LM;
const seed = 20251123, data = P08.makeData(LM.rng(seed));
for (const sigma of [0.5, 0.3, 0.8]) for (const act of ['relu', 'silu']) for (const ms of [seed + 91, seed + 92]) {
    const ns = P08.noisySet(data, 48, sigma, seed + 17), out = [];
    for (const skip of [true, false]) {
        const m = P08.makeUNet({ c1: 4, c2: 8, skip, act }, LM.rng(ms));
        const R = P08.makeDenoiseRun({ model: m, lr: 3e-3, cosine: true, B: 8, cap: 1200, seed: seed + 13, sigma, data });
        let t = 0; while (R.step < R.cap) { const t0 = process.hrtime.bigint(); P08.denoiseStep(R); t += Number(process.hrtime.bigint() - t0) / 1e6; }
        const tr = {}, y = AG.noGrad(() => P08.unetForward(m, AG.tensor(48, 256, ns.Xn), tr)).data, full = P08.psnr(P08.mseOf(y, ns.X0, y.length));
        let abl = '';
        if (skip) { const u1 = AG.concatCols(AG.tensor(48, 4 * 256), tr.s1), y0 = AG.noGrad(() => P08.conv(u1, m.out.w, m.out.b, { C: 8, H: 16, W: 16, K: 3 })).data; abl = ` (h2 zeroed ${P08.psnr(P08.mseOf(y0, ns.X0, y0.length)).toFixed(2)})`; }
        out.push(`${skip ? 'skip' : 'noskip'} ${full.toFixed(2)}${abl} ${(t / R.cap).toFixed(2)}ms`);
    }
    console.log(`σ ${sigma} ${act.padEnd(4)} seed+${ms - seed}: ${out.join(' | ')}`);
}
