// p11_bake.js — 把 tools/p11_ckpt/final_*.json 写进 11_diffusion.lab.js 开头的 P11_W 块（// <P11_W> … // </P11_W>）
const fs = require('fs'), path = require('path');
const dir = path.join(__dirname, 'p11_ckpt'), file = path.join(__dirname, '..', '11_diffusion.lab.js');
const rd = (n) => JSON.parse(fs.readFileSync(path.join(dir, `final_${n}.json`), 'utf8'));
const m = { vp: rd('vp'), cfg: rd('cfg'), fm: rd('fm'), rf: rd('rf'), latent: rd('latent') };
const line = (k, o) => `    ${k}: { cfg: ${JSON.stringify(o.cfg)}, recipe: ${JSON.stringify(o.recipe)},\n        curve: ${JSON.stringify(o.curve)},\n        b64: '${o.b64}' },`;
const L = m.latent;
const block = `// <P11_W> 离线训练好的权重（tools/p11_train.js 训练、tools/p11_bake.js 写入，请勿手改）。float16 小端序 base64，按 params 顺序；curve = 每 100 步的平均训练损失
const P11_W = {
${line('vp', m.vp)}
${line('cfg', m.cfg)}
${line('fm', m.fm)}
${line('rf', m.rf)}
    latent: { aeSteps: ${L.aeSteps}, ldmSteps: ${L.ldmSteps}, mu: ${JSON.stringify(L.mu.map((v) => +v.toFixed(5)))}, sd: ${JSON.stringify(L.sd.map((v) => +v.toFixed(5)))}, cfg: ${JSON.stringify(L.cfg)},
        ae: '${L.ae}',
        b64: '${L.b64}' },
};
// </P11_W>
`;
let s = fs.readFileSync(file, 'utf8');
if (s.includes('// <P11_W>')) s = s.replace(/\/\/ <P11_W>[\s\S]*?\/\/ <\/P11_W>\n/, () => block);
else s = s.replace('const P11 = (() => {', () => block + '\nconst P11 = (() => {');
fs.writeFileSync(file, s);
console.log('baked', Object.keys(m).join(', '), (block.length / 1024).toFixed(0) + ' KB');
