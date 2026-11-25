// 一次性：修 App 叫法、换说法模板（有 / 没有应用名两种写法）、加 SaaS 长尾（约 60 个常用商业软件与游戏服务）
const fs = require('fs');
const path = require('path');
const p = path.join(__dirname, '..', 'tools_catalog.js');
let s = fs.readFileSync(p, 'utf8');
function rep(a, b) { const n = s.split(a).length - 1; if (n !== 1) throw new Error(`expected 1 × «${a.slice(0, 70)}», found ${n}`); s = s.replace(a, () => b); }
// 1) 应用叫法里不要混进对象名
rep("A('excel', 'Excel', 'office', 'Excel,表格,电子表格,xlsx',", "A('excel', 'Excel', 'office', 'Excel,电子表格,xlsx',");
rep("A('mail', '邮件（Outlook / Gmail）', 'office', '邮件,邮箱,Outlook,Gmail,email',", "A('mail', '邮件（Outlook / Gmail）', 'office', '邮箱,Outlook,Gmail',");
rep("A('calendar', '日历（Outlook / Google Calendar）', 'office', '日历,日程,会议,行程',", "A('calendar', '日历（Outlook / Google Calendar）', 'office', '日历,Outlook 日历,Google 日历',");
rep("A('feishu', '飞书（Lark）', 'office', '飞书,Lark,飞书文档,多维表格',", "A('feishu', '飞书（Lark）', 'office', '飞书,Lark',");
rep("A('filesystem', '文件系统', 'dev', '文件,目录,本地文件,磁盘',", "A('filesystem', '文件系统', 'dev', '本地,本地磁盘,文件系统',");
rep("A('git', 'Git', 'dev', 'git,Git,本地仓库,版本',", "A('git', 'Git', 'dev', 'git,本地仓库',");
rep("A('github', 'GitHub', 'dev', 'GitHub,PR,合并请求,issue',", "A('github', 'GitHub', 'dev', 'GitHub,GitHub 上',");
rep("A('browser', '浏览器自动化（Playwright）', 'dev', '浏览器,网页,Playwright,网页自动化',", "A('browser', '浏览器自动化（Playwright）', 'dev', '浏览器,Playwright',");
rep("A('web', '联网搜索与抓取', 'dev', '搜索,上网查,百度,谷歌,网页内容',", "A('web', '联网搜索与抓取', 'dev', '网上,百度,谷歌',");
// 2) 说法模板：每条 [有应用名, 没有应用名]
rep(`    const ZH_T = ['{app}里{v}{o}', '帮我在{app}{v}{o}', '{v}一下{o}', '{app}：{v}{o}', '把{o}{v}', '请{v}{o}{x}', '能不能{v}{app}的{o}', '{v}{o}{x}', '在{app}中{v}这个{o}', '我想{v}{o}', '{o}{v}一下', '给我{v}{o}{x}'];
    const ZH_HOLD = ['麻烦{v}一下{app}的{o}', '{app}那边{v}{o}{x}', '要{v}{o}'];
    const EN_T = ['{v} the {o} in {app}', '{app}: {v} {o}', 'please {v} {o}{x}', 'can you {v} the {o}', '{v} {o}{x} in {app}', 'I need to {v} a {o}', '{v} {o}'];
    const EN_HOLD = ['could you {v} {o} for me in {app}', 'go ahead and {v} the {o}{x}'];`,
`    const ZH_T = [['{app}里{v}{o}', '{v}{o}'], ['帮我在{app}里{v}{o}', '帮我{v}{o}'], ['把{app}里的{o}{v}', '把{o}{v}'], ['{app}：{v}{o}{x}', '{v}{o}{x}'], ['请在{app}{v}{o}{x}', '请{v}{o}{x}'], ['我要在{app}{v}{o}', '我要{v}{o}'], ['{app}的{o}{v}', '{o}{v}'], ['给我{v}{o}{x}', '给我{v}{o}{x}'], ['能不能帮我在{app}里{v}{o}', '能不能帮我{v}{o}'], ['用{app}{v}{o}', '{v}{o}']];
    const ZH_HOLD = [['麻烦在{app}{v}{o}', '麻烦{v}{o}'], ['{app}那边{v}{o}{x}', '{v}{o}{x}'], ['想在{app}里{v}{o}', '想{v}{o}']];
    const EN_T = [['{v} the {o} in {app}', '{v} the {o}'], ['{app}: {v} {o}{x}', '{v} {o}{x}'], ['please {v} the {o}{x}', 'please {v} the {o}{x}'], ['can you {v} a {o} in {app}', 'can you {v} a {o}'], ['I need to {v} the {o}', 'I need to {v} the {o}'], ['{v} {o} in {app}', '{v} {o}']];
    const EN_HOLD = [['could you {v} the {o} for me in {app}', 'could you {v} the {o} for me'], ['go ahead and {v} the {o}{x}', 'go ahead and {v} the {o}{x}']];`);
rep(`    function fill(tpl, app, v, o, x, zh) {
        const a = app ? (zh ? app : app) : '';
        let s = tpl.replace('{app}', a).replace('{v}', v).replace('{o}', o).replace('{x}', x);
        if (!app) s = s.replace(/^(帮我在|在|能不能)?：?/, (m) => (m === '帮我在' ? '帮我' : m === '能不能' ? '能不能' : '')).replace(/里|中(?=\\S)/, '').replace(/ in $/, '').replace(/^: /, '').replace(/ in (?=\\S)/, ' ');
        return s.replace(/\\s+/g, ' ').replace(/的的/g, '的').trim();
    }`,
`    function fill(tpl, app, v, o, x) {
        const s = (app ? tpl[0] : tpl[1]).replace('{app}', app).replace('{v}', v).replace('{o}', o).replace('{x}', x);
        return s.replace(/\\s+/g, ' ').replace(/ the\\s*$| a\\s*$/, '').replace(/的的/g, '的').trim();
    }`);
rep("            let s = fill(pick(tpls), appName, v, o, x, zh);\n            if (!o) s = s.replace(/这个\\s*$/, '').replace(/ the\\s*$/, '').replace(/ a\\s*$/, '');", "            const s = fill(pick(tpls), appName, v, o, x);");
// 3) SaaS 长尾：大多是对业务对象的增删改查（真实的 Zapier / Composio 式目录就是这样到上千个动作的）
rep("    // —— 没有接口时：操作屏幕 ——", fs.readFileSync(path.join(__dirname, 'tkcat_saas.txt'), 'utf8') + "\n    // —— 没有接口时：操作屏幕 ——");
// 4) 新的通用动词
rep("        apply: ['应用,套用', 'apply', 'Apply a {o}.', 'name:str', 'w'],\n    };", `        apply: ['应用,套用', 'apply', 'Apply a {o}.', 'name:str', 'w'],
        cancel: ['取消', 'cancel', 'Cancel a {o}.', 'id:str reason:str?', 'd'],
        refund: ['退款', 'refund', 'Refund a {o}.', 'id:str amount:num?', 'd'],
        fulfill: ['发货', 'fulfill,ship', 'Mark a {o} as fulfilled / shipped.', 'id:str tracking_number:str?', 'o'],
        archive: ['归档', 'archive', 'Archive a {o}.', 'id:str', 'w'],
        complete: ['完成,标记完成,打勾', 'complete,mark done', 'Mark a {o} as complete.', 'id:str', 'w'],
        pause: ['暂停', 'pause', 'Pause a {o}.', 'id:str', 'w'],
        resume: ['恢复,继续', 'resume', 'Resume a paused {o}.', 'id:str', 'w'],
        publish: ['发布', 'publish', 'Publish a {o}.', 'id:str', 'o'],
        adjust: ['调整,修改数量', 'adjust', 'Adjust a {o}.', 'id:str delta:int', 'w'],
        tag: ['打标签,加标签', 'tag,label', 'Add tags to a {o}.', 'id:str tags:str[]', 'w'],
        void: ['作废', 'void', 'Void a {o}.', 'id:str', 'd'],
    };`);
fs.writeFileSync(p, s);
console.log('ok');
