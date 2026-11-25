// 列出「既写了 blocks，又写了 formula / code / previews / notes / warn」的步骤（后者会被框架忽略）
const fs = require('fs'), path = require('path'), vm = require('vm');
const pages = path.join(__dirname, '..');
const stub = new Proxy({}, { get: () => () => ({}) });
for (const f of fs.readdirSync(pages).filter((x) => /\.lab\.js$/.test(x)).sort()) {
    let spec = null;
    const ctx = { console, Math, Float32Array, Float64Array, Int32Array, Int8Array, Uint8Array, Uint16Array, Uint32Array, Int16Array, ArrayBuffer, DataView, Array, Object, Map, Set, JSON, Number, String, Date, Promise, Proxy, Symbol, Error, RegExp, isFinite, isNaN, parseInt, parseFloat, atob, btoa, Buffer, performance, TextDecoder, TextEncoder, setTimeout, clearTimeout };
    ctx.window = ctx; ctx.PipelineLab = ctx.PL = { mount(s) { spec = s; }, draw: stub, mesh: stub, scene: stub, esc: (s) => String(s), rng: () => () => 0.5, gauss: () => 0, fft: stub, cmap: stub, v3: stub, buf: () => new Float32Array(1), clamp: (x) => x, lerp: (a) => a, smoothstep: () => 0, fract: (x) => x, hexToRgb: () => [0, 0, 0] };
    vm.createContext(ctx);
    const srcs = ['llm_common.js'].concat(fs.existsSync(path.join(pages, 'llm_weights.js')) ? ['llm_weights.js'] : []);
    const src = fs.readFileSync(path.join(pages, f), 'utf8');
    if (/\bAGT(_CPP|_W)?\b/.test(src)) srcs.push('agent_common.js', 'agent_cpp.js', 'agent_weights.js');
    if (/\bTK(_CPP|_W|CAT)?\b/.test(src)) srcs.push('tools_catalog.js', 'tools_common.js', 'tools_weights.js', 'tools_cpp.js');
    try { vm.runInContext(srcs.map((s) => fs.readFileSync(path.join(pages, s), 'utf8')).join('\n;\n') + '\n;\n' + src, ctx, { filename: f }); } catch (e) { console.log(f, 'ERR', e.message.slice(0, 80)); continue; }
    for (const [i, st] of (spec.steps || []).entries()) {
        if (!st.blocks) continue;
        const lost = ['formula', 'formulaHtml', 'code', 'previews', 'notes', 'note', 'warn', 'warns'].filter((k) => st[k] != null && (!Array.isArray(st[k]) || st[k].length));
        if (lost.length) console.log(`${f} step ${String(i + 1).padStart(2, '0')} ${st.id}: blocks + ${lost.join(', ')}${st.previews ? ` (previews: ${st.previews.map((p) => p.id || p).join(', ')})` : ''}`);
    }
}
