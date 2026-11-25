// mcp_common.js — 第 21 篇（游戏引擎与商用软件）共用的「真东西」：一个虚拟引擎里的场景、一个按 MCP 规范工作的服务器（JSON-RPC 消息、
// tools/list、tools/call、annotations、isError）、一段在引擎里执行的脚本（迷你解释器）、Computer Use 的截图与点击代价、各软件的接入方式。
// 依赖：TK（BPE 计数）。浏览器与 Node 通用。
const MC = (() => {
    // ═════════════════════════ 1. 场景：一个关卡里的 actor（灯光、静态网格、后期、音频……）═════════════════════════
    function buildScene() {
        const actors = [];
        let k = 0;
        const add = (id, type, cls, props, parent) => { actors.push({ id, type, cls, props, parent: parent || null }); return actors[actors.length - 1]; };
        for (let i = 1; i <= 6; i++) add(`PointLight_${i}`, 'light', 'PointLightComponent', { intensity: 5000, color: '#fff2d8', mobility: 'stationary' }, 'Lighting');
        for (let i = 1; i <= 3; i++) add(`SpotLight_${i}`, 'light', 'SpotLightComponent', { intensity: 12000, color: '#ffd9a0', cone: 35 }, 'Lighting');
        add('SkyLight_0', 'light', 'SkyLightComponent', { intensity: 1.2, mobility: 'static' }, 'Lighting');
        add('Fog_0', 'post', 'ExponentialHeightFog', { density: 0.02, color: '#8fa6c0' }, 'Atmosphere');
        for (let i = 0; i < 18; i++) add(`SM_Rock_${String(i).padStart(3, '0')}`, 'mesh', 'StaticMeshComponent', { asset: '/Game/Env/Rock', lod: 1, castShadow: true }, 'Environment');
        for (let i = 0; i < 12; i++) add(`SM_Crate_${String(i).padStart(3, '0')}`, 'mesh', 'StaticMeshComponent', { asset: '/Game/Env/Crate', lod: 0, castShadow: true }, 'Props');
        for (let i = 0; i < 6; i++) add(`BP_Torch_${i}`, 'blueprint', 'Blueprint', { class: '/Game/Blueprints/BP_Torch', state: 'idle' }, 'Props');
        for (let i = 0; i < 4; i++) add(`Audio_Ambient_${i}`, 'audio', 'AudioComponent', { cue: 'Amb_Forest', volume: 0.6 }, 'Audio');
        // 上千个批量放置的植被：真实关卡里到处是这样生成的 actor
        for (let i = 0; i < 1600; i++) { const t = i % 5; add(`Foliage_${['Fern', 'Grass', 'Moss', 'Vine', 'Shrub'][t]}_${String(i).padStart(4, '0')}`, 'foliage', 'InstancedStaticMesh', { mesh: `/Game/Foliage/${['Fern', 'Grass', 'Moss', 'Vine', 'Shrub'][t]}`, density: 1 }, 'Foliage'); k++; }
        return { actors, k };
    }
    const SCHEMA = {
        list_actors: { type: 'object', properties: { type: { type: 'string', enum: ['light', 'mesh', 'foliage', 'blueprint', 'audio', 'post'] }, limit: { type: 'integer', minimum: 1, maximum: 50 }, offset: { type: 'integer' } }, required: [] },
        get_actor: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
        set_property: { type: 'object', properties: { id: { type: 'string' }, property: { type: 'string' }, value: { type: ['number', 'string', 'boolean'] } }, required: ['id', 'property', 'value'] },
        spawn_actor: { type: 'object', properties: { type: { type: 'string' }, class: { type: 'string' }, count: { type: 'integer' }, location: { type: 'array', items: { type: 'number' } } }, required: ['class'] },
        delete_actor: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
        import_asset: { type: 'object', properties: { path: { type: 'string' }, destination: { type: 'string' } }, required: ['path'] },
        save_level: { type: 'object', properties: { path: { type: 'string' } }, required: [] },
        run_script: { type: 'object', properties: { language: { type: 'string', enum: ['python', 'csharp'] }, source: { type: 'string' } }, required: ['source'] },
    };
    // ═════════════════════════ 2. MCP 服务器：8 个工具，带 annotations；调用真的改场景 ═════════════════════════
    function makeServer() {
        const st = buildScene();
        const byId = new Map(st.actors.map((a) => [a.id, a]));
        const TOOLS = [
            { name: 'list_actors', description: 'List actors in the open level, optionally filtered by type. Parameters are 1-based offset and limit (max 50).', annotations: { readOnlyHint: true, openWorldHint: false } },
            { name: 'get_actor', description: 'Get one actor: class, parent, and all properties.', annotations: { readOnlyHint: true } },
            { name: 'set_property', description: 'Set a property on one actor, e.g. intensity on a light. Persists only after save_level.', annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true } },
            { name: 'spawn_actor', description: 'Spawn one or more actors of a class into the level.', annotations: { readOnlyHint: false, destructiveHint: false } },
            { name: 'delete_actor', description: 'Delete an actor from the level. This cannot be undone from the tool.', annotations: { readOnlyHint: false, destructiveHint: true } },
            { name: 'import_asset', description: 'Import a file from disk into the project (FBX, PNG, WAV). Heavy: cooks on import.', annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: true } },
            { name: 'save_level', description: 'Save the current level to disk.', annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true } },
            { name: 'run_script', description: 'Run a short engine script (python) with the same API as these tools, in one call.', annotations: { readOnlyHint: false, destructiveHint: false } },
        ].map((t) => Object.assign({ inputSchema: SCHEMA[t.name] }, t));
        const names = (t) => st.actors.filter((a) => !t || a.type === t).map((a) => a.id);
        function call(name, args) {
            args = args || {};
            if (name === 'list_actors') {
                const list = names(args.type), off = Math.max(0, (args.offset || 1) - 1), lim = Math.min(50, args.limit || 20), page = list.slice(off, off + lim);
                return { structuredContent: { total: list.length, offset: off + 1, returned: page.length, actors: page } };
            }
            if (name === 'get_actor') { const a = byId.get(args.id); return a ? { structuredContent: { id: a.id, type: a.type, class: a.cls, parent: a.parent, properties: a.props } } : { isError: true, content: `No actor named "${args.id}"` }; }
            if (name === 'set_property') {
                const a = byId.get(args.id); if (!a) return { isError: true, content: `No actor named "${args.id}"` };
                if (!(args.property in a.props)) return { isError: true, content: `Actor ${a.id} (${a.cls}) has no property "${args.property}"` };
                const before = a.props[args.property]; a.props[args.property] = args.value;
                return { structuredContent: { id: a.id, property: args.property, before, after: args.value, saved: false } };
            }
            if (name === 'spawn_actor') {
                const n = Math.max(1, Math.min(10, args.count || 1)), made = [];
                for (let i = 0; i < n; i++) { const id = `${(args.class || 'Actor').split('/').pop()}_${st.actors.length + i}`; add(id, args.type || 'mesh', args.class || 'StaticMeshActor', { asset: args.class || '/Game/Props/Default' }, 'Spawned'); made.push(id); }
                return { structuredContent: { spawned: made } };
            }
            if (name === 'delete_actor') { const i = st.actors.findIndex((a) => a.id === args.id); if (i < 0) return { isError: true, content: `No actor named "${args.id}"` }; st.actors.splice(i, 1); return { structuredContent: { deleted: args.id } }; }
            if (name === 'import_asset') return { structuredContent: { imported: args.path, destination: args.destination || '/Game/Imported', took: '3.2 s' } };
            if (name === 'save_level') return { structuredContent: { saved: args.path || 'CurrentLevel', actors: st.actors.length } };
            if (name === 'run_script') { const r = runScript(st, args.source || ''); return r.error ? { isError: true, content: r.error } : { structuredContent: { calls: r.calls, log: r.log } }; }
            return { isError: true, content: `Unknown tool ${name}` };
        }
        return { st, byId, TOOLS, call, total: () => st.actors.length, reqId: 3 };
    }
    // ═════════════════════════ 3. 脚本：在引擎里执行的迷你解释器（dim / set / spawn / count / delete / print + for i in range(n)）═════════════════════════
    const SCRIPT_API = ['dim(type, factor)', 'set(id, property, value)', 'spawn(class, n)', 'count(type)', 'delete(id)', 'save()', 'print(x)'];
    function parseArgs(s) {
        const out = []; let cur = '', q = null;
        for (const ch of s) {
            if (q) { if (ch === q) q = null; else cur += ch; continue; }
            if (ch === '"' || ch === "'") { q = ch; continue; }
            if (ch === ',') { out.push(cur.trim()); cur = ''; continue; }
            cur += ch;
        }
        if (cur.trim() || s.trim()) out.push(cur.trim());
        return out.map((a) => (/^-?\d+(\.\d+)?$/.test(a) ? Number(a) : /^(true|false)$/.test(a) ? a === 'true' : a));
    }
    function runScript(st, src) {
        const log = [], byId = new Map(st.actors.map((a) => [a.id, a]));
        let calls = 0, error = null, guard = 0;
        const exec = (name, args, ind) => {
            calls++;
            if (name === 'dim') { let n = 0; for (const a of st.actors) if (a.type === args[0] && 'intensity' in a.props) { a.props.intensity = Math.round(a.props.intensity * args[1]); n++; } log.push(`${' '.repeat(ind)}dim(${args[0]}, ${args[1]}) → 改了 ${n} 个 actor`); }
            else if (name === 'set') { const a = byId.get(args[0]); if (!a) throw new Error(`no actor "${args[0]}"`); if (!(args[1] in a.props)) throw new Error(`${a.id} has no property "${args[1]}"`); a.props[args[1]] = args[2]; log.push(`${' '.repeat(ind)}set(${args[0]}, ${args[1]}, ${args[2]})`); }
            else if (name === 'spawn') { const n = Math.max(1, Math.min(10, args[1] || 1)); for (let i = 0; i < n; i++) { const id = `${String(args[0]).split('/').pop()}_${st.actors.length + i}`; st.actors.push({ id, type: 'mesh', cls: args[0], props: { asset: args[0] }, parent: 'Spawned' }); byId.set(id, st.actors[st.actors.length - 1]); } log.push(`${' '.repeat(ind)}spawn(${args[0]}, ${n})`); }
            else if (name === 'count') { const n = st.actors.filter((a) => !args[0] || a.type === args[0]).length; log.push(`${' '.repeat(ind)}count(${args[0]}) → ${n}`); return n; }
            else if (name === 'delete') { const i = st.actors.findIndex((a) => a.id === args[0]); if (i < 0) throw new Error(`no actor "${args[0]}"`); st.actors.splice(i, 1); byId.delete(args[0]); log.push(`${' '.repeat(ind)}delete(${args[0]})`); }
            else if (name === 'save') log.push(`${' '.repeat(ind)}save() → ${st.actors.length} 个 actor`);
            else if (name === 'print') log.push(`${' '.repeat(ind)}print → ${args[0]}`);
            else throw new Error(`unknown function ${name}（演示解释器支持 ${SCRIPT_API.join('、')}）`);
            return null;
        };
        const lines = String(src || '').split('\n');
        try {
            for (let i = 0; i < lines.length && !error; i++) {
                const raw = lines[i], line = raw.trim();
                if (!line || line.startsWith('#')) continue;
                const ind = raw.length - raw.trimStart().length;
                let m = /^for\s+\w+\s+in\s+range\((\d+)\)\s*:$/.exec(line);
                if (m) {
                    const n = +m[1], body = [];
                    let j = i + 1;
                    for (; j < lines.length; j++) { const t = lines[j]; if (!t.trim()) continue; if (t.length - t.trimStart().length <= ind) break; body.push(t); }
                    for (let k = 0; k < n; k++) for (const b of body) { if (++guard > 500) throw new Error('script ran more than 500 statements'); const mm = /^(\w+)\((.*)\)$/.exec(b.trim()); if (!mm) throw new Error(`cannot parse "${b.trim()}"`); exec(mm[1], parseArgs(mm[2]), ind + 4); }
                    i = j - 1; continue;
                }
                m = /^(\w+)\((.*)\)$/.exec(line);
                if (!m) throw new Error(`cannot parse "${line}"`);
                exec(m[1], parseArgs(m[2]), ind);
                if (++guard > 500) throw new Error('script ran more than 500 statements');
            }
        } catch (e) { error = `${e.name}: ${e.message}`; }
        return { log, calls, error };
    }
    // ═════════════════════════ 4. JSON-RPC 会话：按 MCP 规范拼消息，并数 token ═════════════════════════
    const PROTO = '2025-06-18';
    function rpc(id, method, params) { return Object.assign({ jsonrpc: '2.0', id }, method ? { method } : {}, params ? { params } : {}); }
    /** 一次 tools/call 的来回：请求 + 响应，token 分开数（请求是模型写的，响应是 Harness 读进去的） */
    function callOnce(S, B, name, args) {
        const res = S.call(name, args);
        const req = rpc(S.reqId++, 'tools/call', { name, arguments: args });
        const content = res.structuredContent ? [{ type: 'text', text: JSON.stringify(res.structuredContent) }] : [{ type: 'text', text: res.content || '' }];
        const resp = rpc(S.reqId - 1, null, null); delete resp.method;
        const result = Object.assign({ jsonrpc: '2.0', id: resp.id, result: { content, isError: !!res.isError } }, res.structuredContent ? { result: { content, structuredContent: res.structuredContent, isError: !!res.isError } } : {});
        const tReq = TK.countTokens(B, JSON.stringify(req)), tRes = TK.countTokens(B, JSON.stringify(result));
        return { req, result, res, tReq, tRes, tok: tReq + tRes };
    }
    function initializeSession(S, B) {
        const init = { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: PROTO, capabilities: {}, clientInfo: { name: 'level-editor-agent', version: '0.3' } } };
        const initRes = { jsonrpc: '2.0', id: 1, result: { protocolVersion: PROTO, capabilities: { tools: { listChanged: true } }, serverInfo: { name: 'unreal-editor', version: '5.6' } } };
        const list = { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} };
        const listRes = { jsonrpc: '2.0', id: 2, result: { tools: S.TOOLS.map((t) => ({ name: t.name, description: t.description, inputSchema: t.inputSchema, annotations: t.annotations })) } };
        S.reqId = 3;
        return [{ who: 'client', msg: init, note: '握手：协议版本、双方能力' }, { who: 'server', msg: initRes, note: '服务器报告自己有 tools 能力（工具可能随时变：listChanged）' }, { who: 'client', msg: list, note: '要工具清单（可带 cursor 分页）' }, { who: 'server', msg: listRes, note: `${S.TOOLS.length} 个工具的完整定义进了上下文` }].map((m) => Object.assign(m, { tok: TK.countTokens(B, JSON.stringify(m.msg)) }));
    }
    // ═════════════════════════ 5. 大结果：全部列表 / 过滤 / 在服务器里聚合 ═════════════════════════
    function bigResult(S, B, type) {
        const all = S.call('list_actors', { limit: 50 });
        const total = all.structuredContent.total;
        const fakeAll = { total, returned: total, actors: S.st.actors.map((a) => a.id) };
        const filtered = S.call('list_actors', { type: type || undefined, limit: 50 });
        const script = 'count("mesh")\ncount("foliage")';
        const agg = S.call('run_script', { source: script });
        const t = (o) => TK.countTokens(B, JSON.stringify(o));
        return { total, page: t(all), allIds: t(fakeAll), filtered: t(filtered), filteredN: filtered.structuredContent.total, filteredKind: type || '全部', agg: t(agg) + TK.countTokens(B, script), pageN: all.structuredContent.returned };
    }
    // ═════════════════════════ 6. Computer Use 的代价 ═════════════════════════
    const SHOT = { w: 1456, h: 819, tokens: 1560 };   // 一张 1456×819 的截图约 1560 token（Anthropic 的视觉 token 公式）
    function computerUse(S, B) {
        // 每个动作 = 1 张「看」+ 1 张「确认」；10 个动作的关卡任务
        const actions = ['点开 World Settings 面板', '找到 Lightmass 分类', '把 Static Lighting Level Scale 改成 2', '关闭面板', '在场景里选中 PointLight_3', '打开 Details 面板', '找到 Intensity 输入框', '改成 8000', '点 Save', '确认保存对话框'];
        const per = 2 * SHOT.tokens + 40;            // 两张截图（看 + 确认）+ 一句「点 (x, y)」的思考和坐标
        return { actions, perAction: per, total: per * actions.length, screenshots: 2 * actions.length, calls: actions.length };
    }
    // ═════════════════════════ 7. 接入方式一览（第 09 步的表）═════════════════════════
    const INTEGRATIONS = [
        { app: 'Unreal Engine', api: 'Python（编辑器内）· Remote Control HTTP API · UnrealEditor-Cmd -run=pythonscript', how: '插件把编辑器 API 包成 MCP 工具；无界面批处理走命令行', note: '蓝图、材质、关卡都能改；二进制 .uasset 只能通过引擎改' },
        { app: 'Unity', api: 'Editor C#（EditorWindow / SerializedObject）· -batchmode -executeMethod', how: '编辑器扩展包成 MCP 服务器；CI 里用 batchmode', note: 'Unity 6 起有官方 MCP 集成；Prefab 是 YAML，但改结构仍建议走编辑器' },
        { app: 'Godot', api: 'GDScript（EditorPlugin）· --headless --script', how: '插件 + 命令行', note: '场景是文本 .tscn，可以直接改文本' },
        { app: 'Blender', api: 'bpy（Python）· blender -b -P script.py', how: 'BlenderMCP 这类插件把 bpy 包成工具', note: '建模、材质、导出 glb 全都能脚本化' },
        { app: 'Maya / 3ds Max', api: 'Maya Python (maya.cmds) · MaxScript', how: 'MCP 服务器 + 编辑器里的命令端口', note: '批处理同样走命令行' },
        { app: 'Houdini', api: 'HOM（Python）· hython', how: '脚本接口 + 命令行', note: '程序化生成的场景最适合脚本' },
        { app: 'Photoshop', api: 'UXP 插件（JS）· ActionDescriptor', how: 'UXP 插件暴露工具；或批处理 Action', note: '抠图、批量导出图层' },
        { app: 'Figma', api: 'REST API（读）· Plugin API（改）', how: 'REST 做读；插件做写', note: '设计稿 → 切图 → 导入引擎' },
        { app: 'Office / 飞书 / 腾讯文档', api: 'Office JS · 开放平台 REST', how: '官方 MCP 或自建服务器', note: '表格、文档、PPT 的批量处理' },
        { app: 'Wwise / FMOD', api: 'WAAPI（WebSocket）· FMOD Studio 命令行', how: 'WAAPI 客户端包成工具', note: '事件、总线、随机容器' },
        { app: 'Perforce / Git', api: 'p4 命令行 · git 命令行', how: '直接走 Bash（第 20 篇）', note: '文件在 p4 edit 之前只读' },
        { app: '只有图形界面的软件', api: '无', how: 'Computer Use：截图 + 坐标点击', note: '最慢、最贵、最不稳，最后才用' },
    ];

    return { buildScene, makeServer, runScript, SCRIPT_API, rpc, callOnce, initializeSession, bigResult, computerUse, SHOT, INTEGRATIONS, PROTO, SCHEMA };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = MC;
