// p14_load.js — 在 Node 里加载 14_world_models.lab.js 里的 P14 命名空间（与页面同一份代码），供原型实验 / 离线训练脚本使用。
// 用法：const { LM, P14 } = require('./p14_load.js');
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const file = path.join(__dirname, '..', '14_world_models.lab.js');
const src = fs.readFileSync(file, 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const P14 = new Function('LM', head + '\nreturn P14;')(LM);
module.exports = { LM, P14, file };
