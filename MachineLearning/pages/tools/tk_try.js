// 临时：用页面同一套代码试几条自由输入
global.TKCAT = require('../tools_catalog.js');
const TK = require('../tools_common.js'), W = require('../tools_weights.js');
const R = TK.buildRetriever(W), A = TK.loadAction(W.act), B = TK.loadBPE(require('../tk_bpe.js'));
const qs = process.argv.slice(2).length ? process.argv.slice(2) : ['把关卡里所有点光源的强度减半', '帮我在 Blender 里把选中的模型导出成 glb', '给这个 PR 写个 review 然后合并', '昨天的 DAU 和次留是多少', '把 Hero.fbx 导入到 /Game/Characters', '重启一下线上的 api 服务', 'Steam 上把 1.2.0 的包设为默认分支', '飞书群里通知大家今晚 8 点停服维护', 'what is a draw call', '这个老工具没有接口，帮我点一下导出按钮', '帮我弄一下那个', 'compress all the png textures in the project', '在 Unity 里给主角加一个刚体', '把 TAPD 上这个 bug 指派给小王'];
for (const q of qs) {
    const r = R.search(q, { k: 5, docZh: !process.env.EN }), act = TK.classifyAction(A, q), name = (i) => R.tools[i].mcpName.replace('mcp__', '');
    console.log(`\n「${q}」  action=${act.best} (${act.p.map((p, i) => TKCAT.ACTIONS[i][0] + (100 * p).toFixed(0)).join(' ')})  ${r.ms.toFixed(1)} ms`);
    console.log('  bm25  :', r.top.bm25.slice(0, 3).map(name).join('  '));
    console.log('  dense :', r.top.dense.slice(0, 3).map((i) => `${name(i)} ${r.dense[i].toFixed(2)}`).join('  '));
    console.log('  hybrid:', r.top.hybrid.slice(0, 3).map(name).join('  '));
    console.log('  apps  :', r.appTop.slice(0, 3).map((i) => `${R.apps[i].ns} ${(100 * r.appP[i]).toFixed(0)}%`).join('  '), '→ router:', r.top.router.slice(0, 2).map(name).join('  '));
    const t = R.tools[r.top.hybrid[0]], ex = TK.extractArgs(t, q);
    console.log('  args  :', JSON.stringify(ex.args), 'missing', JSON.stringify(ex.missing));
}
const all = TK.promptParts(B, 'all', {}); console.log('\nall tools tokens:', all.map((p) => `${p.part}=${p.tokens}`).join(' '));
