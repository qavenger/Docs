// agent_cpp_check.js — 单独编译 agent_cpp.js 里的代码块（按给定顺序），与 check_cpp.js 同样的头文件和编译选项。用法：node tools/agent_cpp_check.js chat train session loop grammar
const fs = require('fs'), path = require('path'), os = require('os'), { execFileSync } = require('child_process');
const AGT_CPP = require('../agent_cpp.js');
const CLANG = process.env.CLANGXX || 'C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe';
const names = process.argv.slice(2);
let tu = '#include "minml_gpt.h"\n#include <map>\n#include <set>\n#include <functional>\n#include <algorithm>\n';
for (const n of names) { if (!AGT_CPP[n]) throw new Error('no block ' + n); tu += `\n// ===== ${n} =====\n` + AGT_CPP[n].replace(/^\n/, '') + '\n'; }
const tmp = path.join(os.tmpdir(), 'agent_cpp_check.cpp'); fs.writeFileSync(tmp, tu);
try { execFileSync(CLANG, ['-std=c++17', '-fsyntax-only', '-Wall', '-Wno-unused-variable', '-Wno-unused-function', '-ferror-limit=30', '-I', path.join(__dirname, '..', 'cpp'), tmp], { stdio: 'pipe' }); console.log('✔ compiles:', names.join(' ')); }
catch (e) { console.log(String(e.stderr || e.stdout).split('\n').slice(0, 40).join('\n')); process.exit(1); }
