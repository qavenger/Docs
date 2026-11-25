// 在 Node 里加载 10_multimodal.lab.js 的参考实现（P07 命名空间），供原型实验 / 离线训练脚本使用。
// 用法：const { LM, P07, SPEC } = require('./p10_load.js');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.LM = require('../llm_common.js');
global.PipelineLab = { mount(spec) { global.P07_SPEC = spec; } };
global.PL = global.PipelineLab;
const file = path.join(__dirname, '..', '10_multimodal.lab.js');
const src = fs.readFileSync(file, 'utf8');
vm.runInThisContext(src + '\n;globalThis.P07 = P07;', { filename: file });
module.exports = { LM: global.LM, P07: global.P07, SPEC: global.P07_SPEC, file };
