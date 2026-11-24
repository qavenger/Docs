// live_shots.js — 滚动到指定步骤、让页面里的实时训练真的跑一段时间，再给这一步截图（验证器只模拟约 20 帧，看不到训练后的状态）。
// 用法（在 pages/ 下）：node tools/live_shots.js <page.html> <outDir> <步骤序号:等待秒数> [...] [参数id=值 ...]   例如 node tools/live_shots.js 13_audio_generation.html tools/shots13/live 7:40 9:30
const { spawn } = require('child_process'), fs = require('fs'), os = require('os'), path = require('path'), { pathToFileURL } = require('url');
const [docArg, outArg, ...rest] = process.argv.slice(2), jobs = rest.filter((a) => !a.includes('=')), sets = rest.filter((a) => a.includes('='));
if (!docArg || !outArg || !jobs.length) { console.error('usage: node tools/live_shots.js <page.html> <outDir> <step:seconds> ...'); process.exit(2); }
const doc = path.resolve(docArg), outDir = path.resolve(outArg);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browsers = [process.env['PROGRAMFILES(X86)'] + '\\Microsoft\\Edge\\Application\\msedge.exe', process.env.PROGRAMFILES + '\\Microsoft\\Edge\\Application\\msedge.exe', process.env.PROGRAMFILES + '\\Google\\Chrome\\Application\\chrome.exe'];
const exe = browsers.find((p) => p && fs.existsSync(p));
if (!exe) { console.error('no Edge / Chrome found'); process.exit(2); }
(async () => {
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'live-shots-'));
    const proc = spawn(exe, ['--headless=new', '--no-first-run', '--disable-extensions', '--allow-file-access-from-files', '--hide-scrollbars', '--mute-audio', '--window-size=1440,900', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });
    let port = null;
    for (let i = 0; i < 200 && !port; i++) { const f = path.join(profile, 'DevToolsActivePort'); if (fs.existsSync(f)) port = +fs.readFileSync(f, 'utf8').split(/\r?\n/)[0]; if (!port) await sleep(100); }
    let page = null;
    for (let i = 0; i < 50 && !page; i++) { try { page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch { /* retry */ } if (!page) await sleep(100); }
    const ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise((r) => ws.addEventListener('open', r));
    let seq = 0; const pend = new Map();
    ws.addEventListener('message', (e) => { const m = JSON.parse(String(e.data)); if (m.id && pend.has(m.id)) { const p = pend.get(m.id); pend.delete(m.id); if (m.error) p.rej(new Error(m.error.message)); else p.res(m.result); } });
    const send = (method, params = {}) => new Promise((res, rej) => { const id = ++seq; pend.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
    const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception ? r.exceptionDetails.exception.description : r.exceptionDetails.text); return r.result.value; };
    await send('Page.enable'); await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: pathToFileURL(doc).href });
    await sleep(2500);
    for (const kv of sets) { const at = kv.indexOf('='); await ev(`PipelineLab.setParam(${JSON.stringify(kv.slice(0, at))}, ${JSON.stringify(kv.slice(at + 1))}), true`); }
    fs.mkdirSync(outDir, { recursive: true });
    const n = await ev(`document.querySelectorAll('.pl-step').length`);
    for (const job of jobs) {
        const [si, secs] = job.split(':').map(Number), idx = si - 1;
        if (!(idx >= 0 && idx < n)) { console.log('skip', job, '(page has', n, 'steps)'); continue; }
        // 每隔 0.5 s 在这一步里上下滚一次，让卡片里所有预览轮流进入视口（预览只在可见时重画，lazy stage 只在可见时推进）
        const t0 = Date.now();
        while (Date.now() - t0 < secs * 1000) {
            await ev(`(async () => { const el = document.querySelectorAll('.pl-step')[${idx}], r = el.getBoundingClientRect(), top = r.top + scrollY, raf = () => new Promise((q) => requestAnimationFrame(() => requestAnimationFrame(q)));
                for (let y = top + r.height - 500; y > top - 1; y -= 450) { window.scrollTo(0, y); await raf(); await new Promise((q) => setTimeout(q, 120)); }
                el.scrollIntoView({ block: 'start' }); await raf(); return true; })()`);
            await sleep(400);
        }
        await ev(`(async () => { const el = document.querySelectorAll('.pl-step')[${idx}], r = el.getBoundingClientRect(), top = r.top + scrollY, raf = () => new Promise((q) => requestAnimationFrame(() => requestAnimationFrame(q)));
            for (let y = top + r.height - 400; y > top - 1; y -= 300) { window.scrollTo(0, y); await raf(); await new Promise((q) => setTimeout(q, 150)); }
            el.scrollIntoView({ block: 'start' }); await raf(); return new Promise((q) => setTimeout(q, 300)); })()`);
        const rect = await ev(`(() => { const r = document.querySelectorAll('.pl-step')[${idx}].getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; })()`);
        const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: Math.max(0, rect.x - 4), y: Math.max(0, rect.y - 4), width: Math.min(rect.w + 8, 4000), height: Math.min(rect.h + 8, 6000), scale: 1 } });
        const file = path.join(outDir, `step-${String(si).padStart(2, '0')}-after-${secs}s.png`);
        fs.writeFileSync(file, Buffer.from(shot.data, 'base64'));
        console.log('shot', file);
    }
    try { await send('Browser.close'); } catch { /* ignore */ }
    ws.close(); await sleep(300); try { proc.kill(); } catch { /* ignore */ }
    try { fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch { /* ignore */ }
})().catch((e) => { console.error(e); process.exit(1); });
