# 系列设计说明：从机器学习 Hello World 到多模态大模型

> **更新**：用户要求全部代码改为 C++（C++17 + 标准库 + 本系列自带的 pages/cpp/ 库）。本文件里关于 Python / NumPy / PyTorch 代码块的说法已作废，以 `DESIGN3.md` 为准。


这份文件是写作各篇讲义时共用的设计说明：系列结构、共享代码、约定、每篇提纲和验收标准都在这里。

## 0. 目录结构（GitHub Pages 可直接发布）

```
F:\Docs\MachineLearning\
  index.html                      ← 入口（由 pages/index.lab.js 构建，-o ../index.html）
  pages\
    llm_common.js                 ← 共享代码（每篇构建时第一个输入）——各篇作者不要修改
    llm_weights.js                ← 离线训练好的迷你 GPT「基座模型」权重（04、06 篇需要；05、07 可选）——不要修改
    01_linear_regression.lab.js   ← 已完成，作为风格与结构的参考实现
    02_mlp_backprop.lab.js        → 02_mlp_backprop.html
    06_tokenizer_embedding.lab.js → 06_tokenizer_embedding.html
    07_transformer.lab.js         → 07_transformer.html
    08_pretraining.lab.js         → 08_pretraining.html
    09_alignment.lab.js           → 09_alignment.html
    10_multimodal.lab.js          → 10_multimodal.html
    index.lab.js                  → ../index.html
    tools\                        ← Node 脚本（测试引擎、离线训练权重）、各篇的截图目录 shotsNN\
```

## 1. 构建与验证（在 `F:\Docs\MachineLearning\pages` 目录下执行）

```bash
SK="C:/Users/User/.claude/skills/pipeline-explainer/scripts"
# 不需要基座权重的篇（02、03）：
node "$SK/build_doc.mjs" llm_common.js 02_mlp_backprop.lab.js -o 02_mlp_backprop.html
# 需要基座权重的篇（04、06，05/07 如用到也加上）：
node "$SK/build_doc.mjs" llm_common.js llm_weights.js 07_transformer.lab.js -o 07_transformer.html
# 验证（截图放到各自的目录）：
node "$SK/validate_doc.mjs" 07_transformer.html --shots tools/shots07
# 参数和预览定稿后生成「参数 → 步骤」映射，然后重新构建：
node "$SK/validate_doc.mjs" 07_transformer.html --write-deps 07_transformer.lab.js --shots tools/shots07
```

- 指向**其他尚未完成的篇**的 `broken link` 错误是预期的（导航条链接全部 7 篇 + 入口），可以忽略；其他 error 必须清零。
- 验证器**不检查 Python 代码里的未定义函数**（它只对 HLSL/GLSL/WGSL/C++/JS 做这个检查）。所以必须人工保证：代码块里调用的每个**非内置、非模块属性**（`np.` / `torch.` / `F.` / `nn.` / `math.` 之外）的函数，都在本篇某个代码块（步骤代码、`graph.common`、`library`）里有定义。
- 每帧 stage 开销（验证器打印的 `per-frame stages`）在默认参数下 ≲ 15 ms。
- 一定要打开截图看：文字是否重叠 / 溢出、量程是否合适、每个预览是否真的看得出这一步的作用。

## 2. 共享代码 `llm_common.js`（全局 `LM`，只读）

读源码即可了解细节，这里只列要点：

- `LM.C`（语义色：`train` 蓝、`val` 橙、`target` 绿、`wrong` 红、`yellow` 高亮…）、`LM.SERIES`、`LM.FONT(px, bold)`、`LM.rgbStr`、`LM.tokColor(id)`
- 数学：`LM.rng(seed)`、`LM.randn(rand)`、`LM.shuffle`、`LM.softmax(src, out, temp, off, n)`、`LM.argmax`、`LM.topk`、`LM.ema`、`LM.fmt`、`LM.pca2(X, n, d)`
- **自动微分 `LM.AG`**：张量一律是二维行主序 `[rows × cols]`，batch 序列展平为 `[B·T × d]`。
  `AG.param(name, r, c, init)`、`AG.tensor(r, c, data)`、`AG.track(out, inputs, back)`（扩展自定义算子）、`AG.noGrad(fn)`、`AG.backward(loss)`、`AG.clearTape()`；
  算子 `matmul, matmulBT, linear, addRow, add, sub, mul, scale, relu, tanh, gelu, sigmoid, logsigmoid, mean, sum, transpose, layernorm, gather, embed, concat, meanRows, l2norm, attention(Q,K,V,B,T,H,causal) → .probs, crossEntropy(logits, targets, {weights, denom, keepProbs}) → .rowLoss, seqLogProb(logits, targets, rowSeq, nSeq), mse`。全部已通过有限差分梯度校验（`tools/test_engine.js`）。
  注意：每次前向都会往 tape 里压闭包；只前向不反向时用 `AG.noGrad(...)`，或者结束后 `AG.clearTape()`，否则 tape 会无限增长。
- 优化器：`new LM.AdamW(params, {lr, b1, b2, eps, wd})`（`p.decay` 为 true 的才衰减；`step(lr)` 后 `opt.updRatio[k] = ‖Δp‖/‖p‖`）、`new LM.SGD(params, {lr, momentum})`、`LM.clipGradNorm(params, max)`、`LM.gradNorm(p)`、`LM.cosineLR(it, {maxLR, minLR, warmup, total})`
- **迷你 GPT `LM.gpt`**（与 nanoGPT 同构：可学习位置 embedding、Pre-LN、GELU(tanh)、MLP 4×、权重共享 LM head、GPT-2 初始化）：
  `create(cfg, seed)`（`cfg = {V, T, d, L, H, ff, std}`）、`numParams`、`forward(m, ids, B, T, {targets, weights, trace, inputEmb, keepProbs})`（`trace.layers[l] = {xin, h1, q, k, v, att(probs), a, ao, x1, h2, f1, f2, x2}`、`trace.tok/x0/hf`）、`trainStep(m, opt, x, y, B, T, lr, {clip})`、`batch(data, B, T, rand)`、`nextLogits(m, ids)`、`decodeDist(logits, {temperature, topK, topP})`、`generate(m, prompt, maxNew, {temperature, topK, topP, rand, greedy, stop, keepDists, onStep})`（KV cache，已验证与完整前向逐位相等）、`exportWeights / loadWeights / clone`
- 基座权重：`llm_weights.js` 定义全局 `LM_WEIGHTS.base = { cfg, steps, batch, seeds, maxLR, minLR, warmup, valLoss, curve:[[step, train, val]…], samples, b64 }`；
  加载：`const base = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64)`。
  它是 `cfg = {V:76, T:32, d:32, L:2, H:4, ff:128}`（28 928 个参数），在玩具世界语料上训练了 4000 步（batch 16），val loss 0.507。
- **玩具世界语料 `LM.world`**：8 种动物 × 3 个属性（爱吃 / 住处 / 颜色）= 24 条事实；10 个句式（含需要「抄写」前文的句式：`…所以{名字}每天都吃{食物}`、`…，它住在{住处}`）；`world.sentence(rand, k)`、`world.corpus(rand, n)`、`world.questions`（问答模板，SFT 用）。
- **字符级词表 `LM.vocab`**：76 个 token = 4 个特殊 token（`<pad>`=0、`<eos>`=1、`<user>`=2、`<bot>`=3）+ 72 个汉字/标点。`encode(str)`（特殊 token 写作 `<eos>` 等）、`decode(ids)`、`show(id)`（特殊 token 显示成 `⟨eos⟩`）。基座模型预训练时**从没把**特殊 token 当作输入或目标。注意：它们的 embedding **不是**初始化值——由于权重共享（wte 同时是 LM head），softmax 的梯度每一步都在推所有行，9 个从未出现的 token（4 个特殊 token + 什么？哪颜）在 Adam 下被推成了**同一个向量**（两两余弦 1.000，范数 3.34，而初始化时约 0.11）。所以 SFT 开始时 `<user>`、`<bot>`、`<eos>` 作为输入完全无法区分（第 06 篇实测，第 09 篇可以讲这一点）。`LM.makeVocab(specials, text)` 构造其他字符级词表（多模态 caption 用）。
- 绘图（深色主题、设计坐标）：`LM.draw.heat(canvas, data, R, C, {rowLabels, colLabels, map, range, symmetric, digits, hiRow, hiCol, hiCells, title, legend, square, cellMax, colorFor})`（NaN 画成暗色斜纹，适合因果掩码）+ `heatHit(canvas, info)`；`LM.draw.bars(canvas, vals, labels, {horizontal, colors, hi, range, digits, title, showValues})`；`LM.draw.tokens(canvas, toks, {px, showIds, title})` + `tokenHit`；`LM.draw.text(canvas, lines, {px, title})`（行可以是带颜色的片段数组）；`LM.draw.scatter(canvas, pts, {x, y, paths, lines, bg, xLabel, yLabel, equal, title})`（返回 `{X, Y, inv}`，probe/pick 时用 `inv(info.dx, info.dy)` 反算数据坐标——把返回值存到 `S` 里再在 probe 里用，probe 拿不到 canvas）；`LM.draw.curves(canvas, series, plotOpts)`；`LM.draw.imageGrid(canvas, imgs, W, H, {cols, labels, symmetric, map, outline})` + `gridHit`。
  `PL.draw.plot` 的坑：给了 `o.x` 时，没有自带 `x` 数组的 series 会被**拉伸铺满整个 x 范围**——画「前 k 步」这种部分曲线时必须给 series 传 `x: [0, 1, …, k]`。
- Python 源码字符串：`LM.PY.gpt`（PyTorch 版 `GPTConfig / CausalSelfAttention / MLP / Block / GPT / init_weights`，与 JS 一一对应）、`LM.PY.gptInit`（`configure_init`）、`LM.PY.train`（`get_lr / get_batch / configure_optimizer`）。04–07 篇把需要的放进 `graph.common` 的 blocks，代码里调用 `GPT(...)`、`get_lr(...)` 等就能悬停跳转。
- 导航：`LM.navHtml('0X')` 放在 `intro.html` 最前面；`css: LM.CSS`（还提供 `.lm-tbl` 表格、`.lm-kpi` 小标签样式）。

需要的辅助函数如果共享库没有，就在**自己的** `.lab.js` 里写，放进一个以页号命名的命名空间常量（如 `const P04 = (() => { … })();`），顶层名字不要和 `LM`、`LM_WEIGHTS` 或其他页冲突。确实应该进共享库的东西，在最后的报告里提出来，不要自己改 `llm_common.js`。

可以在 Node 里原型验证（`const LM = require('../llm_common.js'); const W = require('../llm_weights.js');`），例如先确认 SFT 能不能泛化、某个注意力头在干什么，再写进页面。脚本放 `pages/tools/`，命名带页号前缀。

## 3. 全系列约定

- **读者**：懂编程、懂一点线性代数的工程师（很多是游戏 / 图形方向），不一定做过机器学习。
- **语言**：中文正文 + 英文术语（第一次出现时给英文，如「注意力（Attention）」）。
- **代码**：Python。02 篇用 NumPy 从零实现（手写反向传播）；03–07 篇以 PyTorch 为主（需要展示底层时可以用 NumPy）。代码块必须和页面里 JS 参考实现做的是同一件事、同一套约定（形状、归一化、初始化、超参数）。
- **每一步**：`desc` 讲为什么 / 直觉 / 不做会怎样；`formula` 给公式；`code` 给核心代码；`previews` 至少一个、主预览 `large`；`notes` 写「拖哪个参数会看到什么」；`warn` 写坑和与论文不同的约定。6–12 步。最后一步「教学版 → 生产版」。
- **真数据**：预览只画 stage 真实算出的数据。玩具规模不够时如实说明「这是 ×× 的缩小版」，绝不用示意图冒充中间结果。介绍真实大模型的数字（参数量、词表大小、训练 token 数等）只写有把握的；拿不准的宁可不写。
- **训练步数不要按 `S.dt` 缩放**：验证器的 `--deps` 扫描用 dt = 0 推进帧，按 dt 缩放的训练在扫描里一步都不走，参数会被误判为「无影响」。每帧固定步数即可。
- **生命周期**：数据集 / 权重加载 = `resident`；随超参数变 = `param`；训练循环 = `history`（每帧 1 步或若干步，`S.epoch` 变化时重置）；只依赖时间的 = `frame`。只有每帧真的在变的预览才标 `live`。训练类页面默认 `autoplay`，要有「训练到第几步」的状态显示，并设置上限步数（到了就停在最终状态）。
- **参数**：每个都要有 `desc`（一两句：控制什么、拖到哪里看什么）。header 参数总数控制在 ~10 个以内。
- **节点图**：节点 = 一次真实的计算（GEMM、attention kernel、softmax、优化器 step…），pin = 张量名 · 形状（如 `X · [B·T, d] fp32`），同名资源自动连线；`category` 用 `bake`（数据 / 权重）、`user`（超参数）、`function`、`output`；训练循环里的参数是 `history`（`history: true` 的回边）。`graph.common` 放共用 Python 函数。图标题全大写显示，别用小写希腊字母。
- **参考资料**：只列真实存在的论文 / 书 / 课程；没把握的 URL 不写。
- **画布文字** ≥ 11 px（设计坐标）；canvas 上不能写 HTML，用 Unicode 上下标。
- 页面标题：`'0X · <标题>'`（与 `LM.PAGES` 一致）；`lang: 'zh-CN'`、`codeLang: 'py'`、`css: LM.CSS`、`seed` 固定。

## 4. 各篇提纲（建议，作者可以调整，但核心内容要覆盖）

### 02 · 手写数字：从 Softmax 回归到 MLP 与反向传播（NumPy）
数据：在页面里**程序化生成**的 16×16 灰度手写数字 0–9（笔画字形 + 随机仿射 / 粗细 / 抖动 / 噪声），约 1000 训练 + 300 测试，resident stage 生成。intro 里说明这是 MNIST（28×28、6 万张）的缩小替身。
1. 数据集与向量化（样本网格；一张图的像素值热图 → 256 维向量）
2. Softmax 回归：z = Wx + b → softmax → 概率（选中样本的 10 类概率柱状图；可在网格上 pick 样本）
3. 交叉熵损失：−log p_y；为什么不用 MSE（梯度饱和对比）
4. 训练 Softmax 回归（live）：W 的 10 行画成 10 张 16×16「模板」，随训练演化；训练 / 测试准确率曲线
5. 隐藏层与非线性：为什么线性层叠加还是线性；激活函数（sigmoid / tanh / ReLU / GELU）及导数
6. 前向传播：选中样本在每层的真实激活
7. 反向传播：δ₂ = p − y；dW₂ = h·δ₂ᵀ；δ₁ = (W₂ᵀδ₂) ⊙ act'(z₁)；计算图；**三方梯度校验**：手写反传 vs 有限差分 vs `LM.AG`
8. 训练循环（live）：MLP 的 loss / accuracy（train vs test）、第一层权重图像、混淆矩阵
9. 优化器：SGD / Momentum / Adam（偏差修正）——在一个病态二维函数上的轨迹 + 在 MLP 上的 loss 曲线对比
10. 初始化与梯度流：深层 MLP 各层激活 / 梯度的标准差（init 太小 / Xavier / He × sigmoid / ReLU），梯度消失与爆炸
11. 过拟合与正则化（权重衰减 / dropout / early stopping）——可与 8 合并
12. 教学版 → 生产版：autograd、GPU 批量化、CNN（真实 MNIST 上的量级）、MLP 就是 Transformer 里的 FFN（链到 04）、残差与归一化让深网络可训

### 03 · 文本变向量：Tokenizer、Embedding 与位置编码（PyTorch / Python）
1. 字符与字节：中英混排句子的 Unicode 码位与 UTF-8 字节（汉字 3 字节）；byte-level 没有 OOV
2. BPE 训练（字节级）：在玩具世界语料（可再加少量英文）上统计相邻对频率、合并最频繁的一对；参数 = 合并次数；前几次合并先把 3 个字节拼回汉字，然后拼出「小猫」「喜欢」这类词——真实计算
3. 压缩率 vs 词表大小（真实曲线）；词表大小的代价（embedding V×d、softmax 计算）
4. 编码 / 解码：用学到的合并规则切分新句子（token 边界 + id），无损还原
5. Token id → Embedding：one-hot × E = 查表；用**基座模型训练好的 wte**（字符级）画所选 token 的向量与余弦相似度（如果结构不明显就如实说明）
6. Embedding 是学出来的：在玩具语料上（按词切分）**实时训练 word2vec skip-gram + 负采样**，2D PCA 投影随训练演化：动物 / 食物 / 住处 / 颜色聚成簇
7. 相似度与类比：余弦相似度热图、最近邻；类比测试结果如实展示
8. 为什么需要位置信息：注意力对顺序不敏感（打乱 token → 输出集合不变，真实演示）；正弦位置编码热图与 PE(i)·PE(j) 随距离的变化
9. RoPE：成对维度按 pos·θᵢ 旋转；q·k 只依赖相对位置 m−n（不同绝对位置的打分曲线完全重合，真实计算）
10. 输入层合起来：X = E[ids] + P → [T × d]，交给第 07 篇
11. 教学版 → 生产版：tiktoken / SentencePiece、byte fallback、真实词表大小（只写确定的：GPT-2 50257、LLaMA/LLaMA-2 32000、LLaMA-3 128256…）、中文 token 效率、特殊 token 与 chat template、权重共享、长上下文的 RoPE 外推（NTK / YaRN）

### 04 · Transformer：逐层拆解一次前向传播（PyTorch；构建时加 llm_weights.js）
用 `LM_WEIGHTS.base` 做一次完整前向，每一步都画真实中间张量；参数里可切换「训练后 / 随机初始化」权重做对比、选输入句子（几句玩具世界句子，≤ 32 字，挑能体现「抄写」的，如「熊猫喜欢吃竹子，所以熊猫每天都吃」）、选层 / 头 / 查询位置。
1. 输入：token id → wte + wpe → X₀ [T × d]
2. LayerNorm（Pre-LN）：每个 token 归一化前后的均值 / 方差
3. Q / K / V 投影（选中层、头）
4. 打分 QKᵀ/√d_h（T×T 带 token 标签的热图）；为什么除 √d_h（随机向量点积方差随维度增长——真实采样）
5. 因果掩码 + softmax：注意力权重（4 个头并排 + 选中头放大）；pick 一个查询位置看它看向谁；描述训练后模型真实出现的模式（如「抄写头」看向前文同名词之后的那个字）——先在 Node 里分析清楚
6. 加权求和、多头拼接、W_O
7. 残差流：各子层前后每个 token 的范数
8. FFN：GELU 激活 [T × 4d] 热图、稀疏度
9. 输出：ln_f → 权重共享的 LM head → logits → softmax，所选位置的 top-k 下一个 token（训练后 vs 随机）
10. 逐位置损失：整句每个位置的 −log p（哪里容易、哪里难）
11. 自回归生成与 KV cache：续写；KV cache 的代价分析；cache 与完整前向逐位相等的校验（真实）
12. 教学版 → 生产版：FlashAttention（分块 + online softmax）、MQA/GQA、RoPE、RMSNorm、SwiGLU、MoE、长上下文
节点图：Embed → [LN1 → QKV GEMM → Attention → O GEMM → 残差 → LN2 → FC1 → GELU → FC2 → 残差] × L → ln_f → LM head → softmax。

### 05 · 预训练：在浏览器里从零训练一个迷你 GPT（PyTorch；可加 llm_weights.js）
history stage 从随机初始化开始**实时训练**（`LM.gpt.create` + `trainStep`，默认 B=4、T=32，每帧 1 步——在 Node 里约 12 ms；参数「每帧步数」让快的机器加速；cosine 调度总步数如 3000，到了就停）。超参数（lr、batch、warmup…）一改就重置训练。
1. 语料：玩具世界（样本文本、字符数、76 词表、24 条事实）
2. 训练样本：id 流 → 随机窗口 → x / y 右移一位（当前 batch 的 token 条对齐显示，live）
3. 前向与损失：初始 loss ≈ ln 76 = 4.33（为什么）；当前 batch 每个位置的 −log p 热图（live）
4. 反向：每个参数张量的梯度范数（live 柱状图）、全局梯度范数曲线、裁剪到 1.0
5. AdamW 更新：公式（偏差修正、解耦权重衰减）；每个张量的 ‖Δp‖/‖p‖（live，经验值 ~1e-3）
6. 学习率调度：warmup + cosine 曲线 + 当前步标记
7. 训练曲线（live）：train / val loss；水平参考线：语料的 unigram 熵、bigram 条件熵（真实统计）——模型依次越过这些「阶段」
8. 模型在学什么（live）：当前步的生成样本；各 checkpoint（第 0/100/300/1000… 步）的样本快照；事实探针：每种动物 P(正确食物首字 | 「X喜欢吃」) 随训练变化
9. 采样与解码：temperature / top-k / top-p 怎样改变下一个 token 分布（可用基座权重，保证效果稳定），并实际生成
10. 规模化：Chinchilla 参数化损失 L(N, D) = E + A/N^α + B/D^β（E=1.69, A=406.4, B=410.7, α=0.34, β=0.28），C ≈ 6ND、compute-optimal ≈ 20 token/参数；本页模型已用的 FLOPs 计数（真实）
11. 教学版 → 生产版：数据清洗 / 去重 / 配比、bf16 混合精度、DP / FSDP(ZeRO) / TP / PP、activation checkpointing、loss spike 与回滚、μP、MFU
节点图：get_batch → embed → blocks → lm_head → cross_entropy → backward → clip → AdamW → Params（history 回边）。

### 06 · 后训练与对齐：SFT、奖励模型、RLHF、DPO、GRPO（PyTorch；构建时加 llm_weights.js）
1. 基座模型不会「对话」：把 `<user>熊猫喜欢吃什么？<bot>` 喂给基座模型，看它真实生成什么
2. 对话模板与 loss mask：`<user> 问题 <bot> 回答 <eos>`；问题部分 targets = −1；token 条可视化
3. SFT（live）：在部分动物的问答上微调基座模型，留出几种动物不训练；loss 曲线；留出动物上的回答是否正确——**先在 Node 里验证能否泛化**，如实呈现结果
4. 偏好数据：同一问题的若干候选回答（正确完整 / 事实错误 / 啰嗦 / 不结束…），用隐藏的「真实奖励」+ Bradley-Terry 模拟标注者
5. 奖励模型（live）：Bradley-Terry 损失 −log σ(r_w − r_l)；可以用冻结 SFT 模型的隐状态做特征；准确率、各候选的奖励
6. RLHF 目标：max E[r] − β·KL(π‖π_ref) 的解析最优解 π* ∝ π_ref·exp(r/β)（随 β 变化，精确计算）
7. PPO（live）：在候选回答上的策略（初始化为 SFT 模型给出的真实序列概率）用裁剪目标 + KL 惩罚训练，收敛到 π*；裁剪目标函数图
8. 奖励过度优化：RM 不完美时，随 KL 增大，代理奖励一直涨、真实奖励先涨后跌（Goodhart；Gao et al. 2022）
9. DPO（live）：同一批偏好对直接优化，收敛到同一个 π*；如可行，也在真实迷你 LM 上用 `AG.seqLogProb` 做 DPO
10. GRPO / RLVR：每个问题采样 G 个回答，advantage = (r − mean)/std，无价值网络；可验证奖励（答案里有没有正确的食物）——能在真实迷你 LM 上跑就跑（KV cache 采样 + 加权交叉熵更新），否则在候选集上
11. 教学版 → 生产版：InstructGPT 三阶段、PPO 需要的 4 个模型、DPO 变体、RLAIF、推理 RL（DeepSeek-R1）、常见坑（reward hacking、长度偏好、模式坍缩）

### 07 · 多模态：ViT、CLIP 对比学习与迷你 LLaVA（PyTorch）
数据：程序化生成的小图（如 24×24 RGB，1–2 个不同颜色 / 形状 / 位置的图形）+ 中文 caption（如「红色圆形在左边」），字符级 caption 词表用 `LM.makeVocab`。
1. 图像也是 token：切 P×P patch → 展平 → 线性投影 → patch token（+ [CLS] + 位置）
2. ViT 编码器：双向注意力（`AG.attention(..., causal=false)`），[CLS] 对各 patch 的注意力图
3. CLIP 双塔：图像编码器 + 文本编码器 → L2 归一化嵌入
4. InfoNCE 对比损失（live）：batch 内 N×N 相似度矩阵逐渐变成对角；对称交叉熵、温度 τ
5. 共享嵌入空间（live）：图文嵌入的 PCA，配对逐渐靠拢
6. Zero-shot 分类：测试图与「一个{颜色}的{形状}」等提示的相似度柱状图、测试集准确率
7. LLaVA 架构：冻结的视觉编码器 → 投影层（MLP）→ 视觉 token 拼在文本 token 前 → 解码器 LM（`LM.gpt.forward` 的 `inputEmb`）；序列布局与 loss mask
8. 训练（live）：投影层（+ 解码器）在 caption 上训练，测试图的生成 caption 随训练变好；讲清 LLaVA 真实的两阶段（特征对齐 → 指令微调），数字只写有把握的
9. 教学版 → 生产版：Flamingo（Perceiver resampler + gated cross-attention）、BLIP-2 Q-Former、LLaVA-1.5、动态分辨率（如 Qwen2-VL）、原生多模态 / early fusion、图像生成（扩散 / 离散图像 token）、语音

## 5. 交付时的报告（每篇作者）

最终回复里写：文件路径；步骤清单；验证器的最终输出（error / warning / 每帧耗时）；做了哪些简化、与原论文不同的约定；在 Node 里验证过的事实（例如 SFT 是否泛化、注意力头的模式）；还没解决的问题；希望共享库增加的东西。
