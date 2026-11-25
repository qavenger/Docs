// p08_final_shots.js — 把页面里的训练一口气跑到最后一步，再给控制台和每一步截图（验证器只模拟约 20 帧，看不到最终状态）。
// 用法（在 pages/ 下）：node tools/p08_final_shots.js [outDir=tools/shots08/final] [id=value ...]   需要 Node 22+（全局 WebSocket）和 Edge / Chrome
const { spawn } = require('child_process'), fs = require('fs'), os = require('os'), path = require('path'), { pathToFileURL } = require('url');
const args = process.argv.slice(2), sets = args.filter((a) => a.includes('=')), outDir = path.resolve(args.find((a) => !a.includes('=')) || 'tools/shots08/final');
const doc = path.resolve(__dirname, '..', '08_pretraining.html');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browsers = [process.env['PROGRAMFILES(X86)'] + '\\Microsoft\\Edge\\Application\\msedge.exe', process.env.PROGRAMFILES + '\\Microsoft\\Edge\\Application\\msedge.exe', process.env.PROGRAMFILES + '\\Google\\Chrome\\Application\\chrome.exe'];
const exe = browsers.find((p) => p && fs.existsSync(p));
if (!exe) { console.error('no Edge / Chrome found'); process.exit(2); }
(async () => {
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'p05-shots-'));
    const proc = spawn(exe, ['--headless=new', '--no-first-run', '--disable-extensions', '--allow-file-access-from-files', '--hide-scrollbars', '--window-size=1440,900', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });
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
    for (const kv of sets) { const [k, v] = kv.split('='); await ev(`PipelineLab.setParam(${JSON.stringify(k)}, ${JSON.stringify(v)}), true`); }
    await sleep(500);
    const t0 = Date.now();
    const info = await ev(`(async () => {
        const S = PipelineLab.S; S.playing = false;
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const R = S.run; while (R.step < R.total) P05.step(R, S.D, S.probes, S.G.legit);
        PipelineLab.invalidate('frame');
        const ev = P05.lastEval(R);
        return { step: R.step, val: ev.val, facts: Array.from(ev.probe).filter((p) => p > 0.9).length, ms: R.ms };
    })()`);
    console.log('trained to', JSON.stringify(info), `in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    fs.mkdirSync(outDir, { recursive: true });
    for (const f of fs.readdirSync(outDir)) if (f.endsWith('.png')) fs.rmSync(path.join(outDir, f));
    const sels = await ev(`(() => { const out = []; document.querySelectorAll('.pl-section, .pl-step').forEach((el, i) => { el.setAttribute('data-p05', String(i)); out.push(el.id || ('el' + i)); }); return out; })()`);
    for (let i = 0; i < sels.length; i++) {
        // 先把卡片从下往上滚一遍：预览只在进入视口时重画（IntersectionObserver），长卡片的下半部分否则会停在旧状态
        await ev(`(async () => { const el = document.querySelector('[data-p05="${i}"]'), r = el.getBoundingClientRect(), top = r.top + scrollY, raf = () => new Promise((q) => requestAnimationFrame(() => requestAnimationFrame(q)));
            for (let y = top + r.height - 400; y > top - 1; y -= 400) { window.scrollTo(0, y); await raf(); await new Promise((q) => setTimeout(q, 60)); }
            el.scrollIntoView({ block: 'start' }); await raf(); return new Promise((q) => setTimeout(q, 250)); })()`);
        const rect = await ev(`(() => { const r = document.querySelector('[data-p05="${i}"]').getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; })()`);
        const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: Math.max(0, rect.x - 4), y: Math.max(0, rect.y - 4), width: Math.min(rect.w + 8, 4000), height: Math.min(rect.h + 8, 6000), scale: 1 } });
        fs.writeFileSync(path.join(outDir, `${String(i + 1).padStart(2, '0')}-${sels[i]}.png`), Buffer.from(shot.data, 'base64'));
    }
    console.log('screenshots:', outDir, sels.length);
    try { await send('Browser.close'); } catch { /* ignore */ }
    ws.close(); await sleep(300); try { proc.kill(); } catch { /* ignore */ }
    try { fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch { /* ignore */ }
})().catch((e) => { console.error(e); process.exit(1); });
