// coding_common.js — 第 20 篇（Coding Agent）共用的「真东西」：一个小游戏逻辑项目（带一个真实的 bug）、虚拟文件系统、
// 编程 Agent 的通用工具（Read / Write / Edit / Glob / Grep / Bash / Task）的真实语义、迷你 Shell（管道、git status/diff/log、npm test）、
// CommonJS 测试运行器（带死循环保护）、Aider 式仓库地图（定义 / 引用图 + 个性化 PageRank + token 预算）、脚本化的 Agent 轨迹。
// 依赖：TK（tools_common.js，用于 BPE 计数）。浏览器与 Node 通用。
const CA = (() => {
    // ═════════════════════════ 1. 项目：一个小的游戏逻辑库（背包、掉落、伤害、存档、任务）═════════════════════════
    const ITEM_NAMES = ['Potion', 'Elixir', 'Arrow', 'Bolt', 'Herb', 'Ore', 'Gem', 'Scroll', 'Rune', 'Feather', 'Bone', 'Pelt', 'Ingot', 'Shard', 'Seed', 'Mushroom'];
    const ITEM_ADJ = ['Minor', 'Lesser', 'Common', 'Greater', 'Superior', 'Ancient', 'Cursed', 'Blessed', 'Frozen', 'Burning', 'Shadow', 'Golden', 'Iron', 'Silver', 'Crystal'];
    function itemsJson() {
        let s = 2025, r = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
        const out = { potion: { name: 'Health Potion', maxStack: 99, rarity: 'common', value: 25 }, arrow: { name: 'Arrow', maxStack: 999, rarity: 'common', value: 1 }, sword: { name: 'Iron Sword', maxStack: 1, rarity: 'common', value: 120 } };
        for (const a of ITEM_ADJ) for (const n of ITEM_NAMES) {
            if (r() < 0.45) continue;
            const id = `${a}_${n}`.toLowerCase(), rar = ['common', 'uncommon', 'rare', 'epic', 'legendary'][Math.min(4, Math.floor(r() * r() * 5))];
            out[id] = { name: `${a} ${n}`, maxStack: [1, 10, 20, 50, 99][Math.floor(r() * 5)], rarity: rar, value: Math.round(5 + r() * 500), tags: [n.toLowerCase(), rar] };
        }
        return JSON.stringify(out, null, 2) + '\n';
    }
    const REPO = {
        'package.json': `{
  "name": "tiny-rpg-core",
  "version": "0.4.2",
  "description": "Inventory, loot, damage and save logic for a small RPG.",
  "main": "src/index.js",
  "scripts": { "test": "node tests/run.js" },
  "license": "MIT"
}
`,
        'README.md': `# tiny-rpg-core

Gameplay logic shared by the client and the server: inventory with stacking,
weighted loot tables, damage formulas, quests and versioned save files.

## Layout
- \`src/items.js\` — item database (loaded from \`data/items.json\`)
- \`src/inventory.js\` — slots, stacking, add / remove
- \`src/loot.js\` — weighted loot rolls (seeded RNG)
- \`src/damage.js\` — damage, armor and crits
- \`src/save.js\` — save / load with version migration
- \`src/quests.js\` — quest progress

Run the tests with \`npm test\`.
`,
        'CHANGELOG.md': `# Changelog

## 0.4.2
- Inventory: fill partially filled stacks before opening new slots.
- Save: migrate v2 saves (gold moved into the inventory as "coin").

## 0.4.1
- Damage: crits ignore 50% of armor.
- Loot: tables can contain nested tables.

## 0.4.0
- New quest tracker.
`,
        'src/index.js': `const { Inventory } = require('./inventory');
const { getItem, allItems } = require('./items');
const { rollLoot } = require('./loot');
const { computeDamage } = require('./damage');
const { serialize, deserialize } = require('./save');
const { QuestLog } = require('./quests');

module.exports = { Inventory, getItem, allItems, rollLoot, computeDamage, serialize, deserialize, QuestLog };
`,
        'src/items.js': `// Item database. Every item has a maxStack: how many fit in one inventory slot.
const DB = require('../data/items.json');

function getItem(id) {
    const item = DB[id];
    if (!item) throw new Error(\`unknown item "\${id}"\`);
    return item;
}

function allItems() {
    return Object.keys(DB);
}

module.exports = { getItem, allItems };
`,
        'src/inventory.js': `const { getItem } = require('./items');

// A fixed number of slots; each slot holds up to maxStack items of one kind.
class Inventory {
    constructor(capacity = 20) {
        this.capacity = capacity;
        this.slots = [];
    }

    // Put n items in: fill existing slots of the same kind first, then open new slots.
    // Returns false if they do not fit.
    addItem(id, n = 1) {
        const max = getItem(id).maxStack;
        for (const slot of this.slots) {
            if (slot.id !== id || slot.count >= max) continue;
            const room = max - slot.count + 1;
            const k = Math.min(room, n);
            slot.count += k;
            n -= k;
            if (n === 0) return true;
        }
        while (n > 0) {
            if (this.slots.length >= this.capacity) return false;
            const k = Math.min(max, n);
            this.slots.push({ id, count: k });
            n -= k;
        }
        return true;
    }

    // Take n items out, starting from the last slot. Returns false if there are not enough.
    removeItem(id, n = 1) {
        if (this.count(id) < n) return false;
        for (let i = this.slots.length - 1; i >= 0 && n > 0; i--) {
            const slot = this.slots[i];
            if (slot.id !== id) continue;
            const k = Math.min(slot.count, n);
            slot.count -= k;
            n -= k;
            if (slot.count === 0) this.slots.splice(i, 1);
        }
        return true;
    }

    count(id) {
        let total = 0;
        for (const slot of this.slots) if (slot.id === id) total += slot.count;
        return total;
    }

    toJSON() {
        return { capacity: this.capacity, slots: this.slots.map((s) => ({ id: s.id, count: s.count })) };
    }

    static fromJSON(data) {
        const inv = new Inventory(data.capacity);
        for (const s of data.slots) inv.slots.push({ id: s.id, count: s.count });
        return inv;
    }
}

module.exports = { Inventory };
`,
        'src/rng.js': `// mulberry32: small, fast, seedable. Same sequence on every platform.
function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

module.exports = { mulberry32 };
`,
        'src/loot.js': `const { mulberry32 } = require('./rng');

// A loot table is a list of { weight, item, min, max } or { weight, table } (nested).
function rollLoot(table, seed, rolls = 1) {
    const rand = mulberry32(seed);
    const drops = [];
    for (let r = 0; r < rolls; r++) pick(table, rand, drops);
    return drops;
}

function pick(table, rand, drops) {
    const total = table.reduce((s, e) => s + e.weight, 0);
    let x = rand() * total;
    for (const entry of table) {
        x -= entry.weight;
        if (x > 0) continue;
        if (entry.table) return pick(entry.table, rand, drops);
        const count = entry.min + Math.floor(rand() * (entry.max - entry.min + 1));
        drops.push({ id: entry.item, count });
        return drops;
    }
    return drops;
}

module.exports = { rollLoot };
`,
        'src/damage.js': `const { clamp } = require('./utils/clamp');

// damage = attack * skill multiplier * (100 / (100 + effective armor)); crits double and ignore half the armor.
function computeDamage(attacker, defender, skill = { multiplier: 1 }, crit = false) {
    const armor = crit ? defender.armor * 0.5 : defender.armor;
    const raw = attacker.attack * skill.multiplier * (100 / (100 + armor));
    const dmg = crit ? raw * 2 : raw;
    return clamp(Math.round(dmg), 1, 9999);
}

module.exports = { computeDamage };
`,
        'src/save.js': `const { Inventory } = require('./inventory');

const VERSION = 3;

function serialize(player) {
    return JSON.stringify({ version: VERSION, name: player.name, level: player.level, inventory: player.inventory.toJSON() });
}

// Older saves are migrated step by step: v1 had no capacity, v2 kept gold outside the inventory.
function deserialize(text) {
    const data = JSON.parse(text);
    if (data.version === 1) { data.inventory.capacity = 20; data.version = 2; }
    if (data.version === 2) {
        if (data.gold) data.inventory.slots.push({ id: 'arrow', count: data.gold });
        delete data.gold;
        data.version = 3;
    }
    if (data.version !== VERSION) throw new Error(\`unsupported save version \${data.version}\`);
    return { name: data.name, level: data.level, inventory: Inventory.fromJSON(data.inventory) };
}

module.exports = { serialize, deserialize, VERSION };
`,
        'src/quests.js': `// Quest progress: each quest needs some items or kills; progress never exceeds the goal.
class QuestLog {
    constructor() {
        this.quests = new Map();
    }

    start(id, goals) {
        this.quests.set(id, { goals, progress: Object.fromEntries(Object.keys(goals).map((k) => [k, 0])), done: false });
    }

    record(id, key, n = 1) {
        const q = this.quests.get(id);
        if (!q || q.done || !(key in q.goals)) return false;
        q.progress[key] = Math.min(q.goals[key], q.progress[key] + n);
        q.done = Object.keys(q.goals).every((k) => q.progress[k] >= q.goals[k]);
        return true;
    }

    isDone(id) {
        const q = this.quests.get(id);
        return !!q && q.done;
    }
}

module.exports = { QuestLog };
`,
        'src/utils/clamp.js': `function clamp(x, lo, hi) {
    return x < lo ? lo : x > hi ? hi : x;
}

module.exports = { clamp };
`,
        'src/utils/format.js': `const { getItem } = require('../items');

// "Health Potion x3"
function formatStack(slot) {
    return \`\${getItem(slot.id).name} x\${slot.count}\`;
}

module.exports = { formatStack };
`,
        'tests/harness.js': `// A tiny test framework: describe / it / expect. Results are collected and printed by run.js.
const results = [];
let scope = '';

function describe(name, fn) {
    const prev = scope;
    scope = prev ? prev + ' > ' + name : name;
    fn();
    scope = prev;
}

function it(name, fn) {
    try { fn(); results.push({ name: scope + ' > ' + name, ok: true }); }
    catch (error) { results.push({ name: scope + ' > ' + name, ok: false, error }); }
}

const show = (v) => JSON.stringify(v);
function expect(actual) {
    return {
        toBe(expected) { if (actual !== expected) throw new Error(\`expected \${show(actual)} to be \${show(expected)}\`); },
        toEqual(expected) { if (show(actual) !== show(expected)) throw new Error(\`expected \${show(actual)} to equal \${show(expected)}\`); },
        toBeLessThanOrEqual(limit) { if (!(actual <= limit)) throw new Error(\`expected \${show(actual)} <= \${show(limit)}\`); },
        toBeTruthy() { if (!actual) throw new Error(\`expected \${show(actual)} to be truthy\`); },
    };
}

module.exports = { describe, it, expect, results };
`,
        'tests/run.js': `const { results } = require('./harness');

for (const file of ['./inventory.test.js', './loot.test.js', './damage.test.js', './save.test.js']) require(file);

let failed = 0;
for (const r of results) {
    if (r.ok) { console.log('  ✓ ' + r.name); continue; }
    failed++;
    console.log('  ✗ ' + r.name);
    console.log('      ' + r.error.message);
    const at = String(r.error.stack || '').split('\\n').find((line) => line.includes('.test.js'));
    if (at) console.log('      ' + at.trim());
}
console.log(\`\\n\${results.length - failed} passed, \${failed} failed\`);
process.exitCode = failed ? 1 : 0;
`,
        'tests/inventory.test.js': `const { describe, it, expect } = require('./harness');
const { Inventory } = require('../src/inventory');

describe('inventory', () => {
    it('stacks items of the same kind', () => {
        const inv = new Inventory();
        inv.addItem('potion', 3);
        inv.addItem('potion', 4);
        expect(inv.slots).toEqual([{ id: 'potion', count: 7 }]);
    });

    it('opens a new slot when a stack is full', () => {
        const inv = new Inventory();
        inv.addItem('potion', 120);
        expect(inv.slots.length).toBe(2);
    });

    it('never puts more than maxStack in one slot', () => {
        const inv = new Inventory();
        inv.addItem('potion', 98);
        inv.addItem('potion', 5);
        for (const slot of inv.slots) expect(slot.count).toBeLessThanOrEqual(99);
    });

    it('refuses items that do not fit', () => {
        const inv = new Inventory(1);
        expect(inv.addItem('sword', 1)).toBe(true);
        expect(inv.addItem('sword', 1)).toBe(false);
    });

    it('removes from the last slot first', () => {
        const inv = new Inventory();
        inv.addItem('arrow', 10);
        inv.removeItem('arrow', 4);
        expect(inv.count('arrow')).toBe(6);
    });
});
`,
        'tests/loot.test.js': `const { describe, it, expect } = require('./harness');
const { rollLoot } = require('../src/loot');

const TABLE = [
    { weight: 70, item: 'arrow', min: 5, max: 10 },
    { weight: 25, item: 'potion', min: 1, max: 2 },
    { weight: 5, table: [{ weight: 1, item: 'sword', min: 1, max: 1 }] },
];

describe('loot', () => {
    it('is deterministic for a seed', () => {
        expect(rollLoot(TABLE, 42, 5)).toEqual(rollLoot(TABLE, 42, 5));
    });
    it('respects min / max', () => {
        for (const d of rollLoot(TABLE, 7, 200)) if (d.id === 'arrow') expect(d.count >= 5 && d.count <= 10).toBeTruthy();
    });
});
`,
        'tests/damage.test.js': `const { describe, it, expect } = require('./harness');
const { computeDamage } = require('../src/damage');

describe('damage', () => {
    it('armor reduces damage', () => {
        expect(computeDamage({ attack: 100 }, { armor: 100 })).toBe(50);
    });
    it('crits double damage and ignore half the armor', () => {
        expect(computeDamage({ attack: 100 }, { armor: 100 }, { multiplier: 1 }, true)).toBe(133);
    });
    it('never deals less than 1', () => {
        expect(computeDamage({ attack: 0 }, { armor: 10 })).toBe(1);
    });
});
`,
        'tests/save.test.js': `const { describe, it, expect } = require('./harness');
const { serialize, deserialize } = require('../src/save');
const { Inventory } = require('../src/inventory');

describe('save', () => {
    it('round-trips a player', () => {
        const inv = new Inventory();
        inv.addItem('potion', 3);
        const p = deserialize(serialize({ name: 'Ava', level: 7, inventory: inv }));
        expect(p.inventory.count('potion')).toBe(3);
    });
    it('migrates v1 saves', () => {
        const p = deserialize(JSON.stringify({ version: 1, name: 'Old', level: 1, inventory: { slots: [] } }));
        expect(p.inventory.capacity).toBe(20);
    });
});
`,
    };
    // 不在仓库里的「隐藏测试」：评测时才跑（SWE-bench 式：FAIL_TO_PASS 与 PASS_TO_PASS）
    const HIDDEN = {
        'tests/hidden/inventory.hidden.test.js': `const { describe, it, expect } = require('../harness');
const { Inventory } = require('../../src/inventory');

describe('inventory (hidden)', () => {
    it('never loses items when stacking', () => {
        const inv = new Inventory();
        inv.addItem('potion', 98);
        inv.addItem('potion', 5);
        expect(inv.count('potion')).toBe(103);
    });
    it('fills a partial stack before opening a new slot', () => {
        const inv = new Inventory();
        inv.addItem('potion', 50);
        inv.addItem('potion', 60);
        expect(inv.slots).toEqual([{ id: 'potion', count: 99 }, { id: 'potion', count: 11 }]);
    });
});
`,
        'tests/hidden/run_hidden.js': `const { results } = require('../harness');
for (const file of ['../inventory.test.js', '../loot.test.js', '../damage.test.js', '../save.test.js', './inventory.hidden.test.js']) require(file);
for (const r of results) console.log((r.ok ? 'PASS ' : 'FAIL ') + r.name + (r.ok ? '' : ' :: ' + r.error.message));
`,
    };
    // 三种「修法」（第 09 步）：正确的一行修复、只治症状的修复（截断，丢东西）、什么都不改
    const FIXES = {
        good: { file: 'src/inventory.js', old: '            const room = max - slot.count + 1;', new: '            const room = max - slot.count;', label: '正确修复：room = max − count' },
        minus1: { file: 'src/inventory.js', old: '            const room = max - slot.count + 1;', new: '            const room = max - slot.count - 1;', label: '过度修正：room = max − count − 1' },
        clamp: { file: 'src/inventory.js', old: '            slot.count += k;\n            n -= k;\n            if (n === 0) return true;', new: '            slot.count = Math.min(slot.count + k, max);\n            n -= k;\n            if (n === 0) return true;', label: '只治症状：把数量截到 maxStack（多出来的直接丢了）' },
    };

    // ═════════════════════════ 2. 虚拟文件系统 ═════════════════════════
    function newFS(files) {
        const m = new Map(Object.entries(files || REPO));
        if (!m.has('data/items.json')) m.set('data/items.json', itemsJson());
        return { files: m, read: new Set(), base: new Map(m) };
    }
    const cloneFS = (F) => ({ files: new Map(F.files), read: new Set(F.read), base: F.base });
    const norm = (p) => { const out = []; for (const s of String(p || '.').replace(/\\/g, '/').split('/')) { if (!s || s === '.') continue; if (s === '..') out.pop(); else out.push(s); } return out.join('/'); };
    const isDir = (F, p) => { p = norm(p); if (!p) return true; for (const k of F.files.keys()) if (k.startsWith(p + '/')) return true; return false; };
    const listDir = (F, p) => { p = norm(p); const pre = p ? p + '/' : '', set = new Set(); for (const k of F.files.keys()) if (k.startsWith(pre)) { const rest = k.slice(pre.length), i = rest.indexOf('/'); set.add(i < 0 ? rest : rest.slice(0, i) + '/'); } return [...set].sort((a, b) => (a.endsWith('/') === b.endsWith('/') ? a.localeCompare(b) : a.endsWith('/') ? -1 : 1)); };
    function globToRe(g) { let re = ''; for (let i = 0; i < g.length; i++) { const c = g[i]; if (c === '*') { if (g[i + 1] === '*') { re += '.*'; i++; if (g[i + 1] === '/') i++; } else re += '[^/]*'; } else if (c === '?') re += '[^/]'; else if (c === '{') { const j = g.indexOf('}', i); re += '(' + g.slice(i + 1, j).split(',').map((x) => x.replace(/[.+^$()|[\]\\]/g, '\\$&')).join('|') + ')'; i = j; } else re += c.replace(/[.+^$()|[\]\\]/g, '\\$&'); } return new RegExp('^' + re + '$'); }

    // ═════════════════════════ 3. 编程 Agent 的工具（语义仿照 Claude Code 这类工具：先读后改、精确替换、输出截断）═════════════════════════
    const LIMITS = { readLines: 2000, lineChars: 2000, bashChars: 30000, grepHead: 250 };
    /** Read：cat -n 格式（行号 + 制表符），默认最多 2000 行；记录「读过」，Edit 之前必须读过 */
    function toolRead(F, path, offset, limit) {
        path = norm(path);
        if (!F.files.has(path)) return { error: `File does not exist: ${path}` };
        const lines = F.files.get(path).split('\n'); if (lines[lines.length - 1] === '') lines.pop();
        const start = Math.max(1, offset || 1), end = Math.min(lines.length, start - 1 + (limit || LIMITS.readLines));
        F.read.add(path);
        let out = '';
        for (let i = start; i <= end; i++) { const t = lines[i - 1]; out += `${String(i).padStart(6)}\t${t.length > LIMITS.lineChars ? t.slice(0, LIMITS.lineChars) + '…' : t}\n`; }
        if (end < lines.length) out += `… (${lines.length - end} more lines; use offset / limit to read them)\n`;
        return { out, lines: end - start + 1, total: lines.length };
    }
    /** Edit：old_string 必须在文件里恰好出现一次（或 replace_all），并且这个文件在本次对话里读过 */
    function toolEdit(F, path, oldS, newS, replaceAll) {
        path = norm(path);
        if (!F.files.has(path)) return { error: `File does not exist: ${path}` };
        if (!F.read.has(path)) return { error: 'File has not been read yet. Read it first before writing to it.' };
        if (oldS === newS) return { error: 'No changes to make: old_string and new_string are exactly the same.' };
        const text = F.files.get(path);
        if (!oldS) return { error: 'old_string is empty.' };
        const n = text.split(oldS).length - 1;
        if (n === 0) return { error: 'String to replace not found in file.' };
        if (n > 1 && !replaceAll) return { error: `Found ${n} matches of the string to replace, but replace_all is false. Provide more surrounding context to make the match unique, or set replace_all.` };
        const out = replaceAll ? text.split(oldS).join(newS) : text.replace(oldS, () => newS);
        F.files.set(path, out);
        return { out: `The file ${path} has been updated.`, count: replaceAll ? n : 1 };
    }
    function toolWrite(F, path, content) { path = norm(path); if (F.files.has(path) && !F.read.has(path)) return { error: 'File has not been read yet. Read it first before overwriting it.' }; F.files.set(path, content); F.read.add(path); return { out: `File written: ${path}` }; }
    function toolGlob(F, pattern) { const re = globToRe(norm(pattern)); return { out: [...F.files.keys()].filter((k) => re.test(k)).sort().join('\n') || 'No files found' }; }
    /** Grep：ripgrep 式；output_mode = files_with_matches（默认）| content | count */
    function toolGrep(F, pattern, o) {
        o = o || {};
        let re; try { re = new RegExp(pattern, o.i ? 'i' : ''); } catch (e) { return { error: `Invalid regex: ${e.message}` }; }
        const base = norm(o.path || ''), g = o.glob ? globToRe(o.glob.includes('/') ? o.glob : '**/' + o.glob) : null, mode = o.output_mode || 'files_with_matches';
        const out = [];
        for (const [k, text] of [...F.files].sort((a, b) => a[0].localeCompare(b[0]))) {
            if (base && k !== base && !k.startsWith(base + '/')) continue;
            if (g && !g.test(k) && !g.test(k.split('/').pop())) continue;
            const lines = text.split('\n'); let c = 0;
            lines.forEach((l, i) => { if (re.test(l)) { c++; if (mode === 'content') out.push(`${k}:${o.n === false ? '' : i + 1 + ':'}${l}`); } });
            if (c && mode === 'files_with_matches') out.push(k); if (c && mode === 'count') out.push(`${k}:${c}`);
        }
        const head = o.head_limit || LIMITS.grepHead;
        return { out: (out.slice(0, head).join('\n') || 'No matches found') + (out.length > head ? `\n… (${out.length - head} more)` : ''), matches: out.length };
    }

    // ═════════════════════════ 4. 迷你 Shell：管道、权限分级、输出截断 ═════════════════════════
    /** 按 shell 规则切词：引号内的空格、|、&&、; 不切；记下每个词是否带引号（带引号的不做通配符展开） */
    function tokenize(cmd) {
        const out = []; let cur = '', q = null, quoted = false, has = false;
        const flush = () => { if (has) out.push({ t: cur, q: quoted }); cur = ''; quoted = false; has = false; };
        const s = String(cmd);
        for (let i = 0; i < s.length; i++) {
            const ch = s[i];
            if (q) { if (ch === q) q = null; else cur += ch; continue; }
            if (ch === '"' || ch === "'") { q = ch; quoted = true; has = true; continue; }
            if (/\s/.test(ch)) { flush(); continue; }
            if (ch === '|' && s[i + 1] === '|') { flush(); out.push({ t: '||', op: true }); i++; continue; }
            if (ch === '|') { flush(); out.push({ t: '|', pipe: true }); continue; }
            if (ch === '&' && s[i + 1] === '&') { flush(); out.push({ t: '&&', op: true }); i++; continue; }
            if (ch === ';') { flush(); out.push({ t: ';', op: true }); continue; }
            cur += ch; has = true;
        }
        flush();
        return out;
    }
    /** 命令列表：a && b ; c || d → [{ op, toks }]（op = 前面的连接符） */
    const lists = (toks) => { const L = [{ op: null, toks: [] }]; for (const k of toks) { if (k.op) L.push({ op: k.t, toks: [] }); else L[L.length - 1].toks.push(k); } return L.filter((l) => l.toks.length); };
    const segments = (toks) => { const segs = [[]]; for (const k of toks) { if (k.pipe) segs.push([]); else if (!k.op) segs[segs.length - 1].push(k); else segs.push([]); } return segs.filter((s) => s.length); };
    const isRedirect = (k) => !k.q && /^>{1,2}/.test(k.t) && !/^>&\d$/.test(k.t), isFdDup = (k) => !k.q && /^\d?>&\d$/.test(k.t);
    const READ_ONLY = new Set(['ls', 'cat', 'head', 'tail', 'grep', 'rg', 'find', 'wc', 'pwd', 'echo', 'tree']), EXEC = new Set(['npm', 'node', 'npx', 'python', 'make', 'cmake']), DENY = new Set(['rm', 'curl', 'wget', 'sudo', 'chmod', 'chown', 'dd', 'mkfs', 'mv', 'cp', 'kill', 'shutdown', 'scp', 'ssh']);
    /** 权限：read（只读，自动放行）/ exec（会执行代码或改状态，需要确认或在白名单里）/ deny（默认拒绝） */
    function classify(cmd) {
        const toks = tokenize(cmd);
        if (!toks.length) return { perm: 'read', why: '空命令' };
        let worst = 'read', why = '只读命令';
        if (toks.some(isRedirect)) { worst = 'exec'; why = '> 重定向会写文件'; }
        for (const s of segments(toks)) {
            const c = s[0].t, a1 = s[1] && s[1].t;
            if (DENY.has(c)) return { perm: 'deny', why: `${c}：会删除 / 移动文件、联网或提权` };
            if (c === 'git') { if (!['status', 'diff', 'log', 'show', 'blame'].includes(a1)) { worst = 'exec'; why = `git ${a1 || ''}：会改仓库状态`; } continue; }
            if ((c === 'npm' || c === 'pip' || c === 'yarn') && /^(i|install|add|ci)$/.test(a1 || '')) { worst = 'exec'; why = `${c} ${a1}：会联网下载并执行安装脚本`; continue; }
            if (EXEC.has(c)) { worst = 'exec'; why = `${c}：会执行项目里的代码`; continue; }
            if (!READ_ONLY.has(c)) { worst = 'exec'; why = `${c}：不在只读白名单里（不认识的命令一律当成会改东西）`; }
        }
        return { perm: worst, why };
    }
    /** 一套简单的放行策略：工具 × 模式 × 用户写的放行规则（前缀匹配，如 "npm test"）→ allow / ask / deny */
    const MODES = { default: '默认', acceptEdits: '自动接受改动', plan: '只做计划', bypass: '全部放行' };
    function policy(tool, input, mode, allow) {
        if (tool === 'Read' || tool === 'Grep' || tool === 'Glob') return { decision: 'allow', why: '只读工具' };
        if (tool === 'Edit' || tool === 'Write') return mode === 'plan' ? { decision: 'deny', why: '计划模式下不改文件' } : mode === 'default' ? { decision: 'ask', why: '改文件前先给用户看 diff' } : { decision: 'allow', why: '这个模式自动接受改动' };
        if (tool !== 'Bash') return { decision: 'ask', why: '未知工具' };
        const c = classify(input.command), cmd = String(input.command).trim();
        if (c.perm === 'deny') return { decision: 'deny', why: c.why + '（拒绝名单在任何模式下都生效）' };
        if (c.perm === 'read') return { decision: 'allow', why: c.why };
        if (mode === 'plan') return { decision: 'deny', why: '计划模式下不执行会改东西的命令' };
        if (mode === 'bypass') return { decision: 'allow', why: '全部放行（只应在一次性的容器 / 虚拟机里用）' };
        if ((allow || []).some((p) => p && (cmd === p || cmd.startsWith(p + ' ')))) return { decision: 'allow', why: `匹配放行规则「${(allow || []).find((p) => p && (cmd === p || cmd.startsWith(p + ' ')))}」` };
        return { decision: 'ask', why: c.why };
    }
    // Bash 一个工具 = 机器上所有命令行程序。第 19 篇目录里这些应用的「专用工具」，在终端里各是一条命令
    const CLI = [
        { ns: 'git', cmd: 'git log --oneline -5 -- src/inventory.js', what: '查某个文件的提交历史' },
        { ns: 'github', cmd: 'gh pr create --fill --label bug', what: '开一个 PR' },
        { ns: 'docker', cmd: 'docker build -t game-server:0.4.2 .', what: '构建服务器镜像' },
        { ns: 'k8s', cmd: 'kubectl rollout status deploy/game-server', what: '看部署进度' },
        { ns: 'aws', cmd: 'aws s3 sync build/ s3://game-cdn/0.4.2/', what: '上传到 CDN' },
        { ns: 'ffmpeg', cmd: 'ffmpeg -i trailer.mov -c:v libx264 -crf 20 trailer.mp4', what: '转码宣传片' },
        { ns: 'imagemagick', cmd: 'magick icon.png -resize 256x256 icon_256.png', what: '缩放图标' },
        { ns: 'unreal', cmd: 'UnrealEditor-Cmd Game.uproject -run=pythonscript -script=fix_lights.py', what: '无界面跑虚幻编辑器 Python' },
        { ns: 'unity', cmd: 'Unity -batchmode -quit -projectPath . -executeMethod Build.Android', what: '命令行打 Android 包' },
        { ns: 'blender', cmd: 'blender -b hero.blend --python export_glb.py', what: '后台导出 glb' },
        { ns: 'godot', cmd: 'godot --headless --export-release "Windows Desktop" build/game.exe', what: '导出 Windows 版' },
        { ns: 'sql', cmd: 'psql -d game -c "select count(*) from players"', what: '查数据库' },
        { ns: 'redis', cmd: 'redis-cli --scan --pattern "session:*"', what: '列出会话键' },
    ];
    function lineDiff(a, b) {   // 最长公共子序列 → 行级编辑脚本（文件很小，O(nm) 足够）
        const n = a.length, m = b.length, L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
        for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
        const ops = []; let i = 0, j = 0;
        while (i < n && j < m) { if (a[i] === b[j]) { ops.push([' ', a[i], i, j]); i++; j++; } else if (L[i + 1][j] >= L[i][j + 1]) { ops.push(['-', a[i], i, j]); i++; } else { ops.push(['+', b[j], i, j]); j++; } }
        while (i < n) { ops.push(['-', a[i], i, j]); i++; } while (j < m) { ops.push(['+', b[j], i, j]); j++; }
        return ops;
    }
    /** git diff：统一格式，3 行上下文 */
    function unifiedDiff(path, oldT, newT) {
        const a = oldT.split('\n'), b = newT.split('\n'), ops = lineDiff(a, b), out = [`diff --git a/${path} b/${path}`, `--- a/${path}`, `+++ b/${path}`];
        const ch = ops.map((o, k) => (o[0] !== ' ' ? k : -1)).filter((k) => k >= 0);
        if (!ch.length) return '';
        let k = 0;
        while (k < ch.length) {
            let s = Math.max(0, ch[k] - 3), e = Math.min(ops.length - 1, ch[k] + 3);
            while (k + 1 < ch.length && ch[k + 1] - 3 <= e + 1) { k++; e = Math.min(ops.length - 1, ch[k] + 3); }
            const seg = ops.slice(s, e + 1), a0 = seg.find((o) => o[0] !== '+'), b0 = seg.find((o) => o[0] !== '-');
            out.push(`@@ -${(a0 ? a0[2] : 0) + 1},${seg.filter((o) => o[0] !== '+').length} +${(b0 ? b0[3] : 0) + 1},${seg.filter((o) => o[0] !== '-').length} @@`);
            for (const o of seg) out.push(o[0] + o[1]);
            k++;
        }
        return out.join('\n');
    }
    const GIT_LOG = ['a41c9e2 Save: migrate v2 saves (gold → arrows, see CHANGELOG)', '7d03b15 Inventory: fill partial stacks before opening new slots', 'c2e88f0 Damage: crits ignore half the armor', '19ab4d7 Loot: nested tables', '5f0e3a1 Initial import'];
    /** 不带引号、含 * ? 的词按仓库里的路径展开（与 bash 一样：一个都匹配不上就原样保留） */
    function expandGlob(F, word) {
        if (!/[*?]/.test(word)) return [word];
        const re = globToRe(norm(word)), dirs = new Set(); for (const k of F.files.keys()) { const p = k.split('/'); for (let i = 1; i < p.length; i++) dirs.add(p.slice(0, i).join('/')); }
        const hits = [...F.files.keys(), ...dirs].filter((k) => re.test(k)).sort();
        return hits.length ? hits : [word];
    }
    function runShell(F, cmd, B) {
        const perm = classify(cmd), toks = tokenize(cmd);
        if (perm.perm === 'deny') return { out: `（Harness 拒绝执行：${perm.why}）\n`, code: 126, perm, tokens: B ? TK.countTokens(B, perm.why) : null };
        let all = '', code = 0;
        for (const L of lists(toks)) {
            if ((L.op === '&&' && code !== 0) || (L.op === '||' && code === 0)) continue;
            let stdin = null, redirect = null;
            for (const seg of segments(L.toks)) {
                const argv = [];
                for (let i = 0; i < seg.length; i++) {
                    const k = seg[i];
                    if (isFdDup(k)) continue;                                                                  // 2>&1：演示 Shell 本来就把 stderr 并进 stdout
                    if (isRedirect(k)) { const app = k.t.startsWith('>>'), rest = k.t.replace(/^>{1,2}/, ''); redirect = { app, path: norm(rest || (seg[++i] && seg[i].t)) }; continue; }
                    if (k.q) argv.push(k.t); else argv.push(...expandGlob(F, k.t));
                }
                if (!argv.length) continue;
                const r = runOne(F, argv, stdin); stdin = r.out; code = r.code; if (code === 127) break;
            }
            let out = stdin || '';
            if (redirect && redirect.path) { F.files.set(redirect.path, (redirect.app && F.files.has(redirect.path) ? F.files.get(redirect.path) : '') + out); out = ''; }
            all += out;
        }
        if (all.length > LIMITS.bashChars) all = all.slice(0, LIMITS.bashChars) + `\n… [输出 ${all.length} 个字符，超过 ${LIMITS.bashChars} 已截断]`;
        return { out: all, code, perm, tokens: B ? TK.countTokens(B, all) : null };
    }
    function flags(argv) { const f = {}, pos = []; for (let i = 1; i < argv.length; i++) { const a = argv[i]; if (/^-n$/.test(a) && /^\d+$/.test(argv[i + 1] || '')) { f.N = +argv[++i]; continue; } if (/^-\d+$/.test(a)) { f.N = +a.slice(1); continue; } if (a === '-name' || a === '-type' || a === '--glob' || a === '-g') { f[a.replace(/^-+/, '')] = argv[++i]; continue; } if (/^-[a-zA-Z]+$/.test(a)) { for (const c of a.slice(1)) f[c] = true; continue; } if (/^--/.test(a)) { f[a.slice(2)] = true; continue; } pos.push(a); } return { f, pos }; }
    /** grep 默认是 POSIX 基本正则（BRE）：( ) { } | + ? 是普通字符，前面加反斜杠才是特殊含义；rg 与 grep -E 用扩展正则 */
    function breToJs(p) { let out = ''; for (let i = 0; i < p.length; i++) { const ch = p[i]; if (ch === '\\' && i + 1 < p.length) { const nx = p[++i]; out += '(){}|+?'.includes(nx) ? nx : '\\' + nx; continue; } out += '(){}|+?'.includes(ch) ? '\\' + ch : ch; } return out; }
    function runOne(F, argv, stdin) {
        const [c] = argv, { f, pos } = flags(argv), files = F.files, lines = (t) => { const L = t.split('\n'); if (L[L.length - 1] === '') L.pop(); return L; };
        const need = (p) => { const k = norm(p); if (!files.has(k)) throw new Error(`${c}: ${p}: No such file or directory`); return files.get(k); };
        try {
            if (c === 'pwd') return { out: '/workspace/tiny-rpg-core\n', code: 0 };
            if (c === 'echo') return { out: pos.join(' ') + '\n', code: 0 };
            if (c === 'ls' || c === 'tree') {
                const one = (p) => { const items = listDir(F, p); return (f.l ? items.map((x) => { const full = norm((p === '.' ? '' : p + '/') + x.replace(/\/$/, '')); return x.endsWith('/') ? `drwxr-xr-x         - ${x}` : `-rw-r--r-- ${String(files.get(full).length).padStart(9)} ${x}`; }) : items).join('\n') + '\n'; };
                const ps = pos.length ? pos : ['.'], fl = ps.filter((p) => !isDir(F, p)), ds = ps.filter((p) => isDir(F, p));
                fl.forEach(need);
                return { out: (fl.length ? fl.map(norm).join('\n') + '\n' : '') + ds.map((p) => (ps.length > 1 ? `\n${norm(p)}:\n` : '') + one(p)).join(''), code: 0 };
            }
            if (c === 'cat') { if (!pos.length && stdin != null) return { out: stdin, code: 0 }; let o = ''; for (const p of pos) { const L = lines(need(p)); o += f.n ? L.map((l, i) => `${String(i + 1).padStart(6)}\t${l}`).join('\n') + '\n' : L.join('\n') + '\n'; F.read.add(norm(p)); } return { out: o, code: 0 }; }
            if (c === 'head' || c === 'tail') { const N = f.N || 10, src = pos.length ? need(pos[0]) : stdin || '', L = lines(src); if (pos.length) F.read.add(norm(pos[0])); return { out: (c === 'head' ? L.slice(0, N) : L.slice(-N)).join('\n') + '\n', code: 0 }; }
            if (c === 'wc') { const src = pos.length ? pos.map((p) => [p, need(p)]) : [['', stdin || '']]; const o = src.map(([p, t]) => `${String(lines(t).length).padStart(7)}${f.l ? '' : ' ' + String(t.split(/\s+/).filter(Boolean).length).padStart(7) + ' ' + String(t.length).padStart(7)}${p ? ' ' + p : ''}`); if (src.length > 1) o.push(`${String(src.reduce((s, [, t]) => s + lines(t).length, 0)).padStart(7)} total`); return { out: o.join('\n') + '\n', code: 0 }; }
            if (c === 'find') { const p = norm(pos[0] || '.'), re = f.name ? globToRe(f.name) : null; const o = [...files.keys()].filter((k) => (!p || k.startsWith(p + '/') || k === p) && (!re || re.test(k.split('/').pop()))).sort(); if (f.type === 'd') { const ds = new Set(); for (const k of o) { const parts = k.split('/'); for (let i = 1; i < parts.length; i++) ds.add(parts.slice(0, i).join('/')); } return { out: [...ds].sort().join('\n') + '\n', code: 0 }; } return { out: o.map((k) => './' + k).join('\n') + '\n', code: 0 }; }
            if (c === 'grep' || c === 'rg') {
                const pat = pos[0]; if (pat == null) throw new Error(`${c}: missing pattern`);
                const re = new RegExp(c === 'grep' && !f.E ? breToJs(pat) : pat, f.i ? 'i' : ''), rec = c === 'rg' || f.r || f.R, showN = c === 'rg' || f.n;
                if (!pos.slice(1).length && stdin != null) { const L = lines(stdin).filter((l) => re.test(l) !== !!f.v); return { out: f.c ? L.length + '\n' : L.join('\n') + (L.length ? '\n' : ''), code: L.length ? 0 : 1 }; }
                const targets = pos.slice(1).length ? pos.slice(1) : ['.'], o = [];
                for (const t of targets) {
                    const keys = isDir(F, t) ? (rec ? [...files.keys()].filter((k) => !norm(t) || k.startsWith(norm(t) + '/')).sort() : []) : [norm(t)];
                    if (isDir(F, t) && !rec) throw new Error(`grep: ${t}: Is a directory`);
                    for (const k of keys) { if (!files.has(k)) throw new Error(`${c}: ${k}: No such file or directory`); const L = lines(files.get(k)); let cnt = 0; L.forEach((l, i) => { if (re.test(l) !== !!f.v) { cnt++; if (!f.l && !f.c) o.push(`${keys.length > 1 || rec ? k + ':' : ''}${showN ? i + 1 + ':' : ''}${l}`); } }); if (cnt && f.l) o.push(k); if (f.c) o.push(`${k}:${cnt}`); }
                }
                return { out: o.join('\n') + (o.length ? '\n' : ''), code: o.length ? 0 : 1 };
            }
            if (c === 'git') {
                const sub = argv[1];
                if (sub === 'status') {
                    const mod = [...files.keys()].filter((k) => F.base.has(k) && F.base.get(k) !== files.get(k)), add = [...files.keys()].filter((k) => !F.base.has(k)).sort(), del = [...F.base.keys()].filter((k) => !files.has(k));
                    let o = 'On branch main\n';
                    if (mod.length || del.length) o += 'Changes not staged for commit:\n' + mod.map((k) => `\tmodified:   ${k}\n`).join('') + del.map((k) => `\tdeleted:    ${k}\n`).join('');
                    if (add.length) o += 'Untracked files:\n' + add.map((k) => `\t${k}\n`).join('');
                    if (!mod.length && !del.length && !add.length) o += 'nothing to commit, working tree clean\n';
                    return { out: o, code: 0 };
                }
                if (sub === 'diff') { const only = argv.slice(2).filter((a) => !a.startsWith('-')).map(norm); const ds = [...files.keys()].filter((k) => F.base.has(k) && F.base.get(k) !== files.get(k) && (!only.length || only.includes(k))).map((k) => unifiedDiff(k, F.base.get(k), files.get(k))); return { out: ds.join('\n') + (ds.length ? '\n' : ''), code: 0 }; }
                if (sub === 'log') return { out: (argv.includes('--oneline') ? GIT_LOG : GIT_LOG.map((l) => `commit ${l.slice(0, 7)}…\n    ${l.slice(8)}\n`)).join('\n') + '\n', code: 0 };
                return { out: `git ${sub}: 演示 Shell 只支持 status / diff / log\n`, code: 1 };
            }
            if ((c === 'npm' && (argv[1] === 'test' || (argv[1] === 'run' && argv[2] === 'test'))) || (c === 'node' && norm(argv[1]) === 'tests/run.js')) { const r = runTests(F, 'tests/run.js'); return { out: (c === 'npm' ? '\n> tiny-rpg-core@0.4.2 test\n> node tests/run.js\n\n' : '') + r.stdout + (r.error ? r.error + '\n' : ''), code: r.exitCode }; }
            if (c === 'npm') return { out: `npm ${argv[1] || ''}: 演示 Shell 不联网，只支持 npm test\n`, code: 1 };
            if (c === 'node') { if (!argv[1]) throw new Error('node: 演示 Shell 需要一个脚本路径'); const r = runTests(F, norm(argv[1])); return { out: r.stdout + (r.error ? r.error + '\n' : ''), code: r.exitCode }; }
            return { out: `${c}: command not found（演示 Shell 支持：ls cat head tail grep rg find wc git npm node echo pwd）\n`, code: 127 };
        } catch (e) { return { out: e.message + '\n', code: 1 }; }
    }

    // ═════════════════════════ 5. 测试运行器：CommonJS 加载 + 死循环保护 ═════════════════════════
    /** 在每个 for / while 循环体开头插入 __g()：执行超过 200 万次就抛错（防止改坏的代码把页面卡死） */
    function guardLoops(code) {
        let out = '', i = 0;
        const re = /\b(for|while)\s*\(/g; let m;
        while ((m = re.exec(code))) {
            let j = m.index + m[0].length, depth = 1;
            while (j < code.length && depth) { if (code[j] === '(') depth++; else if (code[j] === ')') depth--; j++; }
            let k = j; while (k < code.length && /\s/.test(code[k])) k++;
            if (code[k] === '{') { out += code.slice(i, k + 1) + '__g();'; i = k + 1; re.lastIndex = k + 1; }
        }
        return out + code.slice(i);
    }
    function runTests(F, entry, extra) {
        const files = new Map(F.files); if (extra) for (const [k, v] of Object.entries(extra)) files.set(k, v);
        let stdout = '', steps = 0;
        const con = { log: (...a) => { stdout += a.map(String).join(' ') + '\n'; }, error: (...a) => { stdout += a.map(String).join(' ') + '\n'; } };
        con.warn = con.log; con.info = con.log;
        const proc = { exitCode: 0, env: {} }, __g = () => { if (++steps > 2e6) { tripped = true; throw new Error('Timeout: more than 2,000,000 loop iterations (infinite loop?)'); } };
        let tripped = false;
        const cache = new Map();
        function load(path) {
            if (cache.has(path)) return cache.get(path).exports;
            if (!files.has(path)) throw new Error(`Cannot find module '${path}'`);
            const src = files.get(path), mod = { exports: {} };
            cache.set(path, mod);
            if (path.endsWith('.json')) { mod.exports = JSON.parse(src); return mod.exports; }
            const dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';
            const req = (p) => { const base = norm((dir ? dir + '/' : '') + p); for (const cand of [base, base + '.js', base + '.json', base + '/index.js']) if (files.has(cand)) return load(cand); throw new Error(`Cannot find module '${p}' from '${path}'`); };
            // 与 Node 的 CommonJS 一样把代码包进一个函数；包装写在第一行，报错的行号与文件一致
            const fn = (0, eval)(`(function (module, exports, require, console, process, __g) {${guardLoops(src)}\n})\n//# sourceURL=${path}`);
            fn(mod, mod.exports, req, con, proc, __g);
            return mod.exports;
        }
        let error = null;
        try { load(entry); } catch (e) { error = tripped ? '[Harness] 命令超时：循环执行了 200 万次还没结束，进程被终止（真实的 Harness 按时间算，例如 2 分钟）' : `${e.name}: ${e.message}${(String(e.stack || '').split('\n').find((l) => /(^|[\s(])(src|tests|data)\/[\w./-]+\.js:\d+/.test(l)) || '').replace(/^\s*/, '\n    ')}`; proc.exitCode = 1; }
        if (tripped && !error) error = '[Harness] 命令超时：循环执行了 200 万次还没结束，进程被终止';
        return { stdout, exitCode: proc.exitCode || (error ? 1 : 0), error, steps, timeout: tripped };
    }
    /** SWE-bench 式评测：跑可见测试 + 隐藏测试，逐条给出 PASS / FAIL */
    function evaluate(F) { const r = runTests(F, 'tests/hidden/run_hidden.js', HIDDEN), res = {}; for (const line of r.stdout.split('\n')) { const m = /^(PASS|FAIL) (.*?)( :: (.*))?$/.exec(line); if (m) res[m[2]] = { ok: m[1] === 'PASS', msg: m[4] || '' }; } return { res, error: r.error }; }

    // ═════════════════════════ 6. 仓库地图（Aider 式）：定义 / 引用 → 文件图 → 个性化 PageRank → 在 token 预算内渲染 ═════════════════════════
    const KW = new Set('if for while switch catch function return typeof new constructor static get set else const let var of in do try throw class super this module exports require'.split(' '));
    function tags(F) {
        const defs = [], refs = [];
        for (const [path, text] of F.files) {
            if (!path.endsWith('.js')) continue;
            let cls = null;
            text.split('\n').forEach((line, i) => {
                let m;
                if ((m = /^\s*class\s+(\w+)/.exec(line))) { cls = { name: m[1], line: i }; defs.push({ path, name: m[1], line: i, kind: 'class', text: line.trimEnd() }); }
                else if ((m = /^\s*(?:async\s+)?function\s+(\w+)\s*\(/.exec(line))) defs.push({ path, name: m[1], line: i, kind: 'function', text: line.trimEnd() });
                else if ((m = /^\s*(?:const|let)\s+(\w+)\s*=\s*(?:\([^)]*\)|\w+)\s*=>/.exec(line))) defs.push({ path, name: m[1], line: i, kind: 'function', text: line.trimEnd() });
                else if ((m = /^\s{4}(?:static\s+)?(\w+)\s*\([^)]*\)\s*\{/.exec(line)) && !KW.has(m[1])) defs.push({ path, name: m[1], line: i, kind: 'method', cls, text: line.trimEnd() });
                else if ((m = /^const\s+([A-Z_]{3,})\s*=/.exec(line))) defs.push({ path, name: m[1], line: i, kind: 'const', text: line.trimEnd() });
                for (const id of line.match(/\b[A-Za-z_]\w*\b/g) || []) if (!KW.has(id)) refs.push({ path, name: id });
            });
        }
        return { defs, refs };
    }
    const GLOSSARY = { 背包: 'inventory', 加: 'add', 物品: 'item', 道具: 'item', 堆叠: 'stack', 叠加: 'stack', 上限: 'max', 格子: 'slot', 掉落: 'loot', 伤害: 'damage', 护甲: 'armor', 暴击: 'crit', 存档: 'save', 读档: 'deserialize', 迁移: 'migrate', 任务: 'quest', 进度: 'progress', 随机: 'rng', 种子: 'seed', 数量: 'count', 移除: 'remove', 添加: 'add', 加入: 'add', 格式: 'format', 测试: 'test' };
    /** 需求里的关键词（中文按小词表翻成代码里常见的英文词——这一步真实系统里由模型完成）*/
    function taskTerms(task) { const t = String(task || ''), out = new Set(); for (const w of t.toLowerCase().match(/[a-z_][a-z0-9_]{2,}/g) || []) out.add(w); for (const [zh, en] of Object.entries(GLOSSARY)) if (t.includes(zh)) out.add(en); return [...out]; }
    function repoMap(F, task, budget, B) {
        const { defs, refs } = tags(F), terms = taskTerms(task), mentioned = (name) => terms.some((t) => name.toLowerCase().includes(t) || t.includes(name.toLowerCase()));
        const files = [...new Set(defs.map((d) => d.path).concat(refs.map((r) => r.path)))].sort(), fi = new Map(files.map((p, i) => [p, i])), n = files.length;
        const definers = new Map(); for (const d of defs) { if (!definers.has(d.name)) definers.set(d.name, new Set()); definers.get(d.name).add(d.path); }
        const cnt = new Map(); for (const r of refs) { const D = definers.get(r.name); if (!D) continue; for (const dp of D) if (dp !== r.path) { const k = `${r.path}\u0000${dp}\u0000${r.name}`; cnt.set(k, (cnt.get(k) || 0) + 1); } }
        // 边权的规则取自 Aider 的 repomap.py：长的驼峰 / 下划线名 ×10、需求里提到的 ×10、私有（_开头）×0.1、在 5 个以上文件里都有定义 ×0.1；再乘 √引用次数
        const longName = (name) => name.length >= 8 && ((/_/.test(name) && /[A-Za-z]/.test(name)) || (/[A-Z]/.test(name) && /[a-z]/.test(name)));
        const W = Array.from({ length: n }, () => new Map()), edgeIdent = [];
        for (const [k, c] of cnt) {
            const [u, v, name] = k.split('\u0000');
            let mul = 1; if (longName(name)) mul *= 10; if (mentioned(name)) mul *= 10; if (name.startsWith('_')) mul *= 0.1; if (definers.get(name).size > 5) mul *= 0.1;
            const w = mul * Math.sqrt(c);
            W[fi.get(u)].set(fi.get(v), (W[fi.get(u)].get(fi.get(v)) || 0) + w); edgeIdent.push([fi.get(u), fi.get(v), name, w]);
        }
        const pers = files.map((p) => { const text = F.files.get(p) || '', low = text.toLowerCase(); let s = 0.05; for (const t of terms) { if (p.toLowerCase().includes(t)) s += 5; if (low.includes(t)) s += 1; } return s; }), ps = pers.reduce((a, b) => a + b, 0), P = pers.map((v) => v / ps);
        let r = P.slice();
        for (let it = 0; it < 40; it++) { const nr = P.map((p) => 0.15 * p); let dang = 0; for (let u = 0; u < n; u++) { const out = [...W[u].values()].reduce((a, b) => a + b, 0); if (!out) { dang += r[u]; continue; } for (const [v, w] of W[u]) nr[v] += (0.85 * r[u] * w) / out; } for (let v = 0; v < n; v++) nr[v] += 0.85 * dang * P[v]; r = nr; }
        const outW = files.map((_, u) => [...W[u].values()].reduce((a, b) => a + b, 0)), defRank = new Map();
        for (const [u, v, name, w] of edgeIdent) { const k = `${files[v]}\u0000${name}`; defRank.set(k, (defRank.get(k) || 0) + (r[u] * w) / (outW[u] || 1)); }
        const ranked = defs.map((d) => Object.assign({}, d, { rank: (defRank.get(`${d.path}\u0000${d.name}`) || 0) + r[fi.get(d.path)] * 0.01 + (mentioned(d.name) ? r[fi.get(d.path)] * 0.05 : 0) })).sort((a, b) => b.rank - a.rank);
        const render = (m) => {
            const pick = ranked.slice(0, m), byFile = new Map(); for (const d of pick) { if (!byFile.has(d.path)) byFile.set(d.path, []); byFile.get(d.path).push(d); if (d.cls && !pick.some((q) => q.path === d.path && q.line === d.cls.line)) byFile.get(d.path).push({ path: d.path, line: d.cls.line, text: (F.files.get(d.path).split('\n')[d.cls.line] || '').trimEnd() }); }
            const order = [...byFile.keys()].sort((a, b) => r[fi.get(b)] - r[fi.get(a)]);
            return order.map((p) => { const ls = byFile.get(p).sort((a, b) => a.line - b.line).filter((d, i, a) => !i || a[i - 1].line !== d.line); let s = p + ':\n', last = -2; for (const d of ls) { if (d.line !== last + 1) s += '⋮\n'; s += '│' + d.text + '\n'; last = d.line; } return s + '⋮\n'; }).join('');
        };
        let lo = 1, hi = ranked.length, best = render(1);
        while (lo <= hi) { const mid = (lo + hi) >> 1, s = render(mid), t = TK.countTokens(B, s); if (t <= budget) { best = s; lo = mid + 1; } else hi = mid - 1; }
        return { map: best, tokens: TK.countTokens(B, best), terms, fileRank: files.map((p, i) => ({ path: p, rank: r[i] })).sort((a, b) => b.rank - a.rank), ranked, nDefs: defs.length, graph: { files, rank: r, edges: W.flatMap((m, u) => [...m].map(([v, w]) => [u, v, w])) } };
    }

    // ═════════════════════════ 7. 工具定义（交给模型的 JSON Schema，结构仿照真实编程 Agent 的工具集）═════════════════════════
    const TOOLS = [
        { name: 'Read', description: 'Read a file from the local filesystem. Returns lines with line numbers (cat -n format). Reads up to 2000 lines by default; use offset/limit for large files.', input_schema: { type: 'object', properties: { file_path: { type: 'string' }, offset: { type: 'integer' }, limit: { type: 'integer' } }, required: ['file_path'] } },
        { name: 'Write', description: 'Write a file (overwrites). Read the file first if it already exists.', input_schema: { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } }, required: ['file_path', 'content'] } },
        { name: 'Edit', description: 'Exact string replacement in a file. old_string must match exactly and be unique unless replace_all is set. The file must have been read first.', input_schema: { type: 'object', properties: { file_path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' }, replace_all: { type: 'boolean' } }, required: ['file_path', 'old_string', 'new_string'] } },
        { name: 'Glob', description: 'Find files by glob pattern, e.g. "src/**/*.js".', input_schema: { type: 'object', properties: { pattern: { type: 'string' }, path: { type: 'string' } }, required: ['pattern'] } },
        { name: 'Grep', description: 'Search file contents with a regular expression (ripgrep). output_mode: files_with_matches (default), content, count.', input_schema: { type: 'object', properties: { pattern: { type: 'string' }, path: { type: 'string' }, glob: { type: 'string' }, output_mode: { type: 'string', enum: ['files_with_matches', 'content', 'count'] }, '-n': { type: 'boolean' }, '-i': { type: 'boolean' }, head_limit: { type: 'integer' } }, required: ['pattern'] } },
        { name: 'Bash', description: 'Run a shell command. Output over 30000 characters is truncated. Commands that modify state need user approval.', input_schema: { type: 'object', properties: { command: { type: 'string' }, timeout: { type: 'integer' }, description: { type: 'string' } }, required: ['command'] } },
        { name: 'Task', description: 'Launch a sub-agent with its own context window for a self-contained search or analysis task; only its final report comes back.', input_schema: { type: 'object', properties: { description: { type: 'string' }, prompt: { type: 'string' } }, required: ['description', 'prompt'] } },
        { name: 'TodoWrite', description: 'Create and update a structured task list for the current session.', input_schema: { type: 'object', properties: { todos: { type: 'array', items: { type: 'object' } } }, required: ['todos'] } },
    ];
    const SYSTEM = 'You are a coding agent working in a git repository. Use Grep/Glob to find code before reading it, read only what you need, make minimal edits with Edit, and run the tests after every change. Never claim success without running the tests.';
    const TASK = '玩家反馈：往背包里加药水时，一格里出现了 100 瓶（上限是 99）。请修复，并保证测试通过。';

    // ═════════════════════════ 8. 脚本化的 Agent 轨迹：每一步都真的调用上面的工具 ═════════════════════════
    // 「模型的决定」是写好的（页面里跑不动大模型），但工具执行、输出、token 都是真的
    const PLANS = {
        smart: [
            { think: '先跑测试，看看现在哪里不对。', tool: 'Bash', input: { command: 'npm test', description: 'Run the test suite' } },
            { think: '失败的是「never puts more than maxStack in one slot」。搜一下 maxStack 在哪里用到。', tool: 'Grep', input: { pattern: 'maxStack', path: 'src', output_mode: 'content', '-n': true } },
            { think: 'addItem 在 src/inventory.js，读这一段（第 1–40 行）。', tool: 'Read', input: { file_path: 'src/inventory.js', offset: 1, limit: 40 } },
            { think: '第 16 行多了一个 + 1：已有 98 个时 room 算成 2，格子里就成了 100。改成 max - slot.count。', tool: 'Edit', input: { file_path: 'src/inventory.js', old_string: '            const room = max - slot.count + 1;', new_string: '            const room = max - slot.count;' } },
            { think: '再跑一遍测试。', tool: 'Bash', input: { command: 'npm test', description: 'Run the test suite again' } },
            { think: '全部通过。看一眼最终的改动。', tool: 'Bash', input: { command: 'git diff', description: 'Review the change' } },
        ],
        naive: [
            { think: '先把整个项目都读一遍。', tool: 'Bash', input: { command: 'cat package.json README.md CHANGELOG.md src/index.js src/items.js src/inventory.js src/rng.js src/loot.js src/damage.js src/save.js src/quests.js src/utils/clamp.js src/utils/format.js data/items.json tests/harness.js tests/run.js tests/inventory.test.js tests/loot.test.js tests/damage.test.js tests/save.test.js' } },
            { think: '找到了 addItem 里的问题，改掉。', tool: 'Edit', input: { file_path: 'src/inventory.js', old_string: '            const room = max - slot.count + 1;', new_string: '            const room = max - slot.count;' } },
            { think: '跑测试。', tool: 'Bash', input: { command: 'npm test' } },
        ],
    };
    function execTool(F, tool, input, B) {
        if (tool === 'Bash') return runShell(F, input.command, B);
        if (tool === 'Read') return toolRead(F, input.file_path, input.offset, input.limit);
        if (tool === 'Edit') return toolEdit(F, input.file_path, input.old_string, input.new_string, input.replace_all);
        if (tool === 'Write') return toolWrite(F, input.file_path, input.content);
        if (tool === 'Glob') return toolGlob(F, input.pattern);
        if (tool === 'Grep') return toolGrep(F, input.pattern, { path: input.path, glob: input.glob, output_mode: input.output_mode, n: input['-n'], i: input['-i'], head_limit: input.head_limit });
        return { error: `unknown tool ${tool}` };
    }
    /** 跑一条轨迹：返回每一步的调用、结果与 token；context = 系统提示 + 工具定义 + 用户任务 + 之前所有的调用与结果 */
    function runPlan(plan, B) {
        const F = newFS(), steps = [];
        const fixed = TK.countTokens(B, SYSTEM) + TK.countTokens(B, JSON.stringify(TOOLS)) + TK.countTokens(B, TASK);
        let ctx = fixed;
        for (const s of PLANS[plan]) {
            const call = { type: 'tool_use', name: s.tool, input: s.input }, r = execTool(F, s.tool, s.input, B), result = r.error ? `<tool_use_error>${r.error}</tool_use_error>` : r.out;
            const tCall = TK.countTokens(B, s.think + JSON.stringify(call)), tRes = TK.countTokens(B, result);
            ctx += tCall + tRes;
            steps.push({ think: s.think, call, result, error: !!r.error, perm: r.perm, tCall, tRes, ctx, code: r.code });
        }
        return { F, steps, fixed };
    }
    /** 子 Agent：在自己的上下文里搜索、阅读，只把结论交回来 */
    function runSubagent(B) {
        const F = newFS(), log = [], prompt = '在整个项目里找出所有直接修改 count 的地方，判断哪些可能让数量超过 maxStack 或变成负数，只回报结论。';
        let sub = TK.countTokens(B, SYSTEM) + TK.countTokens(B, JSON.stringify(TOOLS)) + TK.countTokens(B, prompt);
        const calls = [['Grep', { pattern: 'count\\s*(\\+=|-=|:)', path: 'src', output_mode: 'content', '-n': true }], ['Read', { file_path: 'src/inventory.js', offset: 12, limit: 18 }], ['Read', { file_path: 'src/save.js', offset: 10, limit: 11 }], ['Grep', { pattern: 'Math\\.(min|max)', path: 'src', output_mode: 'content', '-n': true }]];
        for (const [tool, input] of calls) { const r = execTool(F, tool, input, B), out = r.error || r.out, t = TK.countTokens(B, JSON.stringify(input)) + TK.countTokens(B, out); sub += t; log.push({ tool, input, out, t }); }
        const report = '结论：(1) src/inventory.js:16 addItem 里 room 多加了 1，已有的格子会被填到 maxStack + 1（就是用户看到的 100 瓶）；(2) removeItem 先检查总数再扣，不会变负；(3) src/save.js:14 迁移 v2 存档时把 gold 整个塞进一个 arrow 格子，gold 超过 999（arrow 的 maxStack）时同样会超上限——这是第二个同类问题，建议改成调用 addItem；(4) quests.js:14 用 Math.min 截断进度，安全。';
        const tReport = TK.countTokens(B, report), tCall = TK.countTokens(B, JSON.stringify({ name: 'Task', input: { description: 'Audit count updates', prompt } }));
        return { prompt, log, sub, report, tReport, tCall };
    }

    return { REPO, HIDDEN, FIXES, newFS, cloneFS, norm, listDir, isDir, globToRe, toolRead, toolEdit, toolWrite, toolGlob, toolGrep, tokenize, classify, policy, MODES, CLI, runShell, lineDiff, unifiedDiff, guardLoops, runTests, evaluate, tags, taskTerms, repoMap, TOOLS, SYSTEM, TASK, PLANS, execTool, runPlan, runSubagent, LIMITS };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = CA;
