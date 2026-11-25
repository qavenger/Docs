global.TKCAT = require('../tools_catalog.js');
const TK = require('../tools_common.js'), W = require('../tools_weights.js'), R = TK.buildRetriever(W);
for (const q of process.argv.slice(2)) {
    const r = R.search(q, { k: 5, docZh: true }), nm = (i) => R.tools[i].mcpName.replace('mcp__', '');
    console.log(`「${q}」\n   dense ${r.top.dense.map(nm).join(' ')}\n   bm25  ${r.top.bm25.map(nm).join(' ')}\n   hyb   ${r.top.hybrid.map(nm).join(' ')}\n   apps  ${r.appTop.slice(0, 3).map((i) => R.apps[i].ns + ' ' + (100 * r.appP[i]).toFixed(0) + '%').join(' ')}`);
}
