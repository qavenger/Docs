// 在 Node 里加载 11_diffusion.lab.js 的参考实现（P11 命名空间，与页面同一份代码），供原型实验 / 离线训练脚本使用。
// 用法：const { LM, P11, SPEC } = require('./p11_load.js');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
global.LM = require('../llm_common.js');
global.PipelineLab = { mount(spec) { global.P11_SPEC = spec; } };
global.PL = global.PipelineLab;
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
const src = fs.readFileSync(file, 'utf8');
vm.runInThisContext(src + '\n;globalThis.P11 = P11;' + (src.includes('const P11_W') ? 'globalThis.P11_W = P11_W;' : ''), { filename: file });
module.exports = { LM: global.LM, P11: global.P11, SPEC: global.P11_SPEC, P11_W: global.P11_W, file };
