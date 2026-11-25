// tk_cpp_check.js — 单独编译 tools_cpp.js / coding_cpp.js 里的代码块（按给定顺序），与 check_cpp.js 同样的头文件和编译选项。
// 用法：node tools/tk_cpp_check.js tooldef bpecount bm25            （默认模块 tools_cpp.js）
//       node tools/tk_cpp_check.js bpecount coding_cpp.js:agenttools coding_cpp.js:repotokens …
const fs = require('fs'), path = require('path'), os = require('os'), { execFileSync } = require('child_process');
const mods = {};
const mod = (f) => (mods[f] = mods[f] || require(path.join(__dirname, '..', f)));
const CLANG = process.env.CLANGXX || 'C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe';
const names = process.argv.slice(2);
let tu = '#include "minml_gpt.h"\n#include <map>\n#include <set>\n#include <functional>\n#include <algorithm>\n#include <unordered_map>\n#include <numeric>\n#include <climits>\n#include <memory>\n#include <cmath>\n#include <tuple>\n#include <regex>\n#include <filesystem>\n#include <fstream>\n#include <sstream>\n#include <cstdio>\n';
for (const n of names) {
    const [f, b] = n.includes(':') ? n.split(':') : ['tools_cpp.js', n];
    const M = mod(f);
    if (!M[b]) throw new Error(`no block ${b} in ${f}`);
    tu += `\n// ===== ${f}:${b} =====\n` + M[b].replace(/^\n/, '') + '\n';
}
const tmp = path.join(os.tmpdir(), 'tk_cpp_check.cpp'); fs.writeFileSync(tmp, tu);
try { execFileSync(CLANG, ['-std=c++17', '-fsyntax-only', '-Wall', '-Wno-unused-variable', '-Wno-unused-function', '-ferror-limit=30', '-I', path.join(__dirname, '..', 'cpp'), tmp], { stdio: 'pipe' }); console.log('✔ compiles:', names.join(' ')); }
catch (e) { console.log(String(e.stderr || e.stdout).split('\n').slice(0, 40).join('\n')); process.exit(1); }
