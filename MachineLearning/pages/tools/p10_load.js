// p10_load.js — 加载 10_reinforcement_learning.lab.js 里 PipelineLab.mount( 之前的部分（P10 命名空间 + P10CPP 代码块，与页面同一份代码）
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../10_reinforcement_learning.lab.js'), 'utf8');
const cut = src.indexOf('PipelineLab.mount(');
const head = cut < 0 ? src : src.slice(0, cut);
const tail = '\nreturn { P10, P10CPP: typeof P10CPP === "undefined" ? null : P10CPP, P10D: typeof P10D === "undefined" ? null : P10D };';
const ns = new Function('LM', head + tail)(LM);
module.exports = { P10: ns.P10, P10CPP: ns.P10CPP, P10D: ns.P10D, LM };
