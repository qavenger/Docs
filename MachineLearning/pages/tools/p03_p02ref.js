// p03_p02ref.js — 第 03 篇引用的第 02 篇数字：用第 02 篇自己的代码和默认设置（H = 64、ReLU、He、SGD lr 0.1、batch 32、3000 步）训练 MLP，报告测试准确率
const path = require('path'), fs = require('fs');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../02_mlp_backprop.lab.js'), 'utf8');
const P02 = new Function('LM', src.slice(0, src.indexOf('PipelineLab.mount(')) + '\nreturn P02;')(LM);
const S = { seed: 20251123, lr: 0.1, hidden: 64, act: 'relu', init: 'he' };
S.data = P02.makeData(LM.rng(S.seed));
const R = P02.makeRun(P02.mlpCfg(S));
P02.evalNow(R);
while (R.step < R.cap) P02.tick(R, 4, 1e9);
const H = R.hist;
console.log(`page 02 MLP (SGD lr 0.1, ${R.step} steps): test ${(100 * H.testA[H.testA.length - 1]).toFixed(1)}%  train(500) ${(100 * H.trainA[H.trainA.length - 1]).toFixed(1)}%`);
