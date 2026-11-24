// gen_cpp.js — 把 pages/cpp/minml_*.h 写进 llm_common.js 的 LM.CPP（两个标记之间的内容整体替换）。
// 修改 C++ 库时：只改 cpp/*.h → 编译运行 cpp/minml_test.cpp 自检 → node tools/gen_cpp.js → node tools/build_all.mjs
const fs = require('fs');
const path = require('path');
const pages = path.join(__dirname, '..');
const MODS = [['core', 'minml_core.h'], ['autograd', 'minml_autograd.h'], ['nn', 'minml_nn.h'], ['gpt', 'minml_gpt.h']];
const BS = String.fromCharCode(92);   // 反斜杠
const esc = (s) => s.split(BS).join(BS + BS).split('`').join(BS + '`').split('${').join(BS + '${');
const NL = String.fromCharCode(10);
const lines = [];
lines.push('    // <CPP-BEGIN> 由 tools/gen_cpp.js 从 pages/cpp/*.h 生成，请勿手改');
lines.push("    // 本系列自带的极简 C++ 机器学习库（只用 C++17 标准库）。每篇把用到的部分放进 graph.common：blocks: LM.CPP.blocks(['core', 'autograd', ...])");
lines.push('    LM.CPP = {');
MODS.forEach(([k, f]) => {
    const src = fs.readFileSync(path.join(pages, 'cpp', f), 'utf8').split(String.fromCharCode(13)).join('');
    lines.push(`        ${k}: \`${NL}${esc(src)}\`,`);
});
lines.push('        files: ' + JSON.stringify(Object.fromEntries(MODS)) + ',');
lines.push('    };');
lines.push("    /** graph.common 用的 blocks：LM.CPP.blocks(['core', 'autograd', 'nn', 'gpt'])（按依赖顺序，缺的依赖自动补上） */");
lines.push('    LM.CPP.blocks = function (mods) {');
lines.push("        const order = ['core', 'autograd', 'nn', 'gpt'], need = new Set(mods || order), deps = { autograd: ['core'], nn: ['autograd'], gpt: ['nn'] };");
lines.push('        const addDeps = (m) => (deps[m] || []).forEach((d) => { need.add(d); addDeps(d); });');
lines.push('        [...need].forEach(addDeps);');
lines.push("        return order.filter((m) => need.has(m)).map((m) => ({ label: `━ cpp/${LM.CPP.files[m]} ━`, code: LM.CPP[m], lang: 'cpp' }));");
lines.push('    };');
lines.push('    // <CPP-END>');
const block = lines.join(NL) + NL;
const lc = path.join(pages, 'llm_common.js');
let s = fs.readFileSync(lc, 'utf8');
const BEGIN = '    // <CPP-BEGIN>', END = '    // <CPP-END>' + NL;
const a = s.indexOf(BEGIN), b = s.indexOf(END);
if (a >= 0 && b > a) s = s.slice(0, a) + block + s.slice(b + END.length);
else {
    const anchor = '    // ═════════════════════════════════ 共享的 Python 源码（代码块） ═════════════════════════════════';
    if (!s.includes(anchor)) throw new Error('anchor not found');
    s = s.replace(anchor, block + NL + anchor);
}
fs.writeFileSync(lc, s);
delete require.cache[require.resolve(lc)];
const LM = require(lc);
console.log('LM.CPP modules:', Object.keys(LM.CPP).filter((k) => typeof LM.CPP[k] === 'string').map((k) => `${k} ${LM.CPP[k].split(NL).length} lines`).join(', '));
// 自检：生成的字符串必须与源文件逐字相同
MODS.forEach(([k, f]) => { const src = fs.readFileSync(path.join(pages, 'cpp', f), 'utf8').split(String.fromCharCode(13)).join(''); if (LM.CPP[k] !== NL + src) throw new Error('round-trip mismatch: ' + f); });
console.log('round-trip OK');
