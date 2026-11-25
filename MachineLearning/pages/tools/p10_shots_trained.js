// 运行 validate_doc --eval p10_eval_trained.js，把返回的 dataURL 存成 shots10/trained/*.png
const { execFileSync } = require('child_process'), fs = require('fs'), path = require('path');
const SK = 'C:/Users/User/.claude/skills/pipeline-explainer/scripts/validate_doc.mjs', page = path.join(__dirname, '..', '10_multimodal.html');
const args = [SK, page, '--no-shots', '--eval', path.join(__dirname, 'p10_eval_trained.js')].concat(process.argv.slice(2));
const txt = execFileSync('node', args, { maxBuffer: 1 << 28, encoding: 'utf8' });
const j = JSON.parse(txt.slice(txt.indexOf('eval →') + 6, txt.indexOf('\nPipeline Lab validation')));
const dir = path.join(__dirname, 'shots10', 'trained'); fs.mkdirSync(dir, { recursive: true });
for (const [k, v] of Object.entries(j)) { if (typeof v === 'string' && v.startsWith('data:image/png')) fs.writeFileSync(path.join(dir, k + '.png'), Buffer.from(v.split(',')[1], 'base64')); else console.log(k, v); }
console.log('saved to', dir);
