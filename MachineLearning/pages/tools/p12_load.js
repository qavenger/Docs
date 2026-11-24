// 在 Node 里加载 12_video_generation.lab.js 的参考实现（P12 命名空间与离线权重 P12_W），供原型实验 / 离线训练脚本使用。
// 用法：const { LM, P12, P12_W } = require('./p12_load.js');
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const file = path.join(__dirname, '..', '12_video_generation.lab.js');
const src = fs.readFileSync(file, 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const { P12, P12_W } = new Function('LM', head + '\nreturn { P12, P12_W };')(LM);
module.exports = { LM, P12, P12_W, file };
