#!/usr/bin/env node
// 一键重建（并可选验证）整个系列。在任意目录执行：
//   node F:/Docs/MachineLearning/pages/tools/build_all.mjs            # 只构建
//   node F:/Docs/MachineLearning/pages/tools/build_all.mjs --validate # 构建 + 逐篇验证（截图放 tools/shotsNN）
//   node .../build_all.mjs 04 06                                       # 只处理指定的篇（入口页用 index）
// 修改讲义时只改 .lab.js（或共享的 llm_common.js），然后重新运行本脚本；不要直接改生成的 HTML。
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const pages = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skill = process.env.PIPELINE_EXPLAINER || 'C:/Users/User/.claude/skills/pipeline-explainer';
const build = path.join(skill, 'scripts', 'build_doc.mjs'), validate = path.join(skill, 'scripts', 'validate_doc.mjs');

// 每篇的构建输入（共享文件在前）；需要基座模型权重的篇加上 llm_weights.js；第 16–18 篇和入口页再加上 agent_common.js + agent_cpp.js + agent_weights.js（会用工具的迷你 GPT）
const DOCS = [
    { id: 'index', src: 'index.lab.js', out: '../index.html', weights: true, agent: true },
    { id: '01', src: '01_linear_regression.lab.js', out: '01_linear_regression.html' },
    { id: '02', src: '02_mlp_backprop.lab.js', out: '02_mlp_backprop.html' },
    { id: '03', src: '03_cnn_unet.lab.js', out: '03_cnn_unet.html' },
    { id: '04', src: '04_latent_vae.lab.js', out: '04_latent_vae.html' },
    { id: '05', src: '05_reinforcement_learning.lab.js', out: '05_reinforcement_learning.html' },
    { id: '06', src: '06_tokenizer_embedding.lab.js', out: '06_tokenizer_embedding.html' },
    { id: '07', src: '07_transformer.lab.js', out: '07_transformer.html' },
    { id: '08', src: '08_pretraining.lab.js', out: '08_pretraining.html' },
    { id: '09', src: '09_alignment.lab.js', out: '09_alignment.html' },
    { id: '10', src: '10_multimodal.lab.js', out: '10_multimodal.html' },
    { id: '11', src: '11_diffusion.lab.js', out: '11_diffusion.html' },
    { id: '12', src: '12_video_generation.lab.js', out: '12_video_generation.html' },
    { id: '13', src: '13_audio_generation.lab.js', out: '13_audio_generation.html' },
    { id: '14', src: '14_world_models.lab.js', out: '14_world_models.html' },
    { id: '15', src: '15_nerf_gaussian_splatting.lab.js', out: '15_nerf_gaussian_splatting.html' },
    { id: '16', src: '16_tool_calling.lab.js', out: '16_tool_calling.html', agent: true },
    { id: '17', src: '17_agent_loop.lab.js', out: '17_agent_loop.html', agent: true },
    { id: '18', src: '18_agent_harness.lab.js', out: '18_agent_harness.html', agent: true },
    { id: '19', src: '19_tool_retrieval.lab.js', out: '19_tool_retrieval.html', tools: true },
    { id: '20', src: '20_coding_agent.lab.js', out: '20_coding_agent.html', coding: true },
    { id: '21', src: '21_engine_mcp.lab.js', out: '21_engine_mcp.html', mcp: true },
];
const usesWeights = (file) => /LM_WEIGHTS/.test(fs.readFileSync(path.join(pages, file), 'utf8'));

const args = process.argv.slice(2), doValidate = args.includes('--validate'), only = args.filter((a) => !a.startsWith('--'));
let failed = 0;
for (const d of DOCS) {
    if (only.length && !only.includes(d.id)) continue;
    if (!fs.existsSync(path.join(pages, d.src))) { console.log(`skip ${d.id}: ${d.src} not found`); continue; }
    const inputs = ['llm_common.js'].concat(d.weights || usesWeights(d.src) ? ['llm_weights.js'] : [], d.agent ? ['agent_common.js', 'agent_cpp.js', 'agent_weights.js'] : [], d.tools ? ['tk_bpe.js', 'tools_catalog.js', 'tools_common.js', 'tools_weights.js', 'tools_cpp.js'] : d.coding ? ['tk_bpe.js', 'tools_catalog.js', 'tools_common.js', 'tools_cpp.js', 'coding_common.js', 'coding_cpp.js'] : [], d.mcp ? ['tk_bpe.js', 'tools_catalog.js', 'tools_common.js', 'mcp_common.js', 'mcp_cpp.js'] : [], [d.src]);
    try {
        execFileSync(process.execPath, [build, ...inputs, '-o', d.out], { cwd: pages, stdio: 'inherit' });
        if (doValidate) {
            const shots = path.join('tools', 'shots' + (d.id === 'index' ? 'Idx' : d.id));
            execFileSync(process.execPath, [validate, d.out, '--shots', shots], { cwd: pages, stdio: 'inherit' });
        }
    } catch (e) { failed++; console.error(`✖ ${d.id} failed (exit ${e.status})`); }
}
process.exit(failed ? 1 : 0);
