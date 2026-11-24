# 系列扩展设计说明（第 08–15 篇）：从共享基础到分化领域

> **更新**：用户要求全部代码改为 C++（C++17 + 标准库 + 本系列自带的 pages/cpp/ 库）。本文件里关于 Python / NumPy / PyTorch 代码块的说法已作废，以 `DESIGN3.md` 为准。


先读 `DESIGN.md`（目录结构、构建 / 验证命令、共享库 API、全系列约定都在那里，全部继续有效）。本文件只写扩展部分：新的系列结构、新增的共享 API、第一批作者踩过的坑、每篇提纲。

## 1. 新的系列结构

篇号是写作顺序（正文里用「第 NN 篇」互相引用，不改）；`part` 是知识结构里的位置。入口页和每篇的导航条按 part 分组展示（`LM.PARTS` / `LM.PAGES`，已经更新好）：

| part | 篇 |
|---|---|
| 共享基础 base | 01 线性回归 · 02 MLP / 反向传播 · **08 CNN / U-Net** · **09 潜空间：AE / VAE / VQ-VAE** · **10 强化学习基础** |
| 通用架构 arch | 03 Tokenizer / Embedding · 04 Transformer |
| 大语言模型 llm | 05 预训练 · 06 对齐（SFT / RLHF / DPO / GRPO） |
| 分化领域 domain | 07 多模态理解（ViT / CLIP / LLaVA）· **11 扩散与流匹配** · **12 视频生成** · **13 音频与语音** · **14 世界模型** · **15 3D 重建：NeRF / 3DGS** |

新篇的文件名（全小写，已写进 `LM.PAGES`）：`08_cnn_unet` · `09_latent_vae` · `10_reinforcement_learning` · `11_diffusion` · `12_video_generation` · `13_audio_generation` · `14_world_models` · `15_nerf_gaussian_splatting`（`.lab.js` → `.html`）。

依赖关系（写正文时可以引用，读者可能跳着读，所以引用时带上一句话的回顾）：
- 08 用到 02（反向传播、He 初始化、优化器）；为 11（U-Net 去噪器）、07（ViT 对比）、12、14 做准备。
- 09 用到 02；为 11（潜空间扩散）、12（视频 VAE）、13（音频 codec 的 RVQ）、14（世界模型的编码器 V）、07（图像 token）做准备。
- 10 用到 01 / 02；为 06（PPO / GRPO 是这里的策略梯度）、14（模型式 RL、规划）做准备。
- 11 用到 02、08、09；12、13 在它之上。
- 12 用到 11、09、04（时空注意力）、07（patch）。
- 13 用到 09（VQ）、04 / 05（token 语言模型）、11（扩散）。
- 14 用到 10、09、12（动作条件视频预测）。
- 15 用到 02、08（位置编码 ↔ 第 03 篇的正弦编码）、01（梯度下降拟合）。

## 2. 共享库新增内容（`llm_common.js`，只读；都已通过 `tools/test_engine2.js` 的有限差分梯度校验）

- 逐元素：`AG.exp`、`AG.log`、`AG.softplus`、`AG.square`、`AG.silu`、`AG.addConst(A, c)`
- 列操作：`AG.concatCols(A, B)`（按列拼接；**特征图的通道拼接也是它**，因为一张图 = 一行）、`AG.sliceCols(A, c0, c1)`
- 卷积族（特征图约定：一张图 = 张量的一行，按通道优先展平 `[C × H × W]`，与 PyTorch NCHW 一致；形状由调用方用 `{ C, H, W }` 传入，返回张量带 `.shape`）：
  `AG.conv2d(X, Wt, b, { C, H, W, K | KH, KW, stride, pad | padH, padW })`（Wt: `[Cout × C·KH·KW]`；一维卷积 = `H: 1, KH: 1`）、`AG.avgPool2`、`AG.maxPool2`、`AG.upsample2`（最近邻）、`AG.addChannel(X, B[N×C], {C})`（扩散 U-Net 加时间 embedding）、`AG.mulChannel`（FiLM 调制）。
  速度参考（Node）：1→8 通道 3×3、16×16、batch 32 的前向 + 反向约 3.3 ms；**浏览器里还要慢约 1.7 倍**。
- 初始化：`LM.init.he(rand, fanIn)`、`LM.init.xavier(rand, fanIn, fanOut)`、`LM.init.normal(rand, std)`、`LM.init.uniform(rand, lim)`（返回给 `AG.param(name, r, c, init)` 用的填充函数）
- 数据：`LM.digits = { IMG: 16, D: 256, K: 10, render(d, rand, out, {clean, dx, dy}), makeSet(n, rand, {clean}) }`——与第 02 篇同一个程序化手写数字生成器（逐像素一致，已验证），第 08、09 篇复用。
- 绘图：`LM.draw.imageGrid(..., { rgb: true })` 画交错 RGB（W·H·3，0..1）小图网格；所有带 `title` 的绘图函数支持 `titleX`（live 预览左上角有 LIVE 徽标，传 `titleX: 46` 避开）；`LM.draw.heat` 支持 `labelEvery`；`LM.draw.text` 按词换行、组合附加符（r̂）正确显示；图例 / 刻度 / id 文字统一 ≥ 11 px。
- 导航：`LM.PARTS`、`LM.PAGES[i].part`；`LM.navHtml('NN')` 自动分组。

需要而共享库没有的东西（GRU 单元、时间步 embedding、VQ 的直通估计、k-means、STFT、体渲染、高斯投影…）写在自己的 `.lab.js` 里、放进页号命名空间（`const P11 = (() => {...})();`）。自定义可微算子用 `AG.track(out, inputs, back)`，写完一定在 Node 里做有限差分梯度校验。

## 3. 第一批作者踩过的坑（务必看）

1. **训练步数不要按 `S.dt` 或墙钟时间缩放**：`--deps` 扫描用 dt = 0 推进帧，按 dt 缩放的训练一步都不走；每帧固定步数。
2. **浏览器比 Node 慢约 1.7 倍**（自动微分大量分配数组 + GC）。在 Node 里量好每步耗时后，按 1.7 倍估浏览器开销；最终以验证器打印的 per-frame 为准（默认 ≲ 15 ms）。本轮有 8 个作者同时在这台机器上跑训练和浏览器，计时会偏高、有抖动——多跑几次取较低值，并在报告里注明。
3. 验证器只模拟约 20 帧：截图看到的是训练初期。长期行为在 Node 里用页面自己的代码验证（参考 `tools/p02_proto.js` 的写法：`new Function('LM', head + '\nreturn P02;')(LM)` 直接加载 `.lab.js` 里 `PipelineLab.mount(` 之前的部分），需要时另写脚本截取训练完成后的预览（参考 `tools/p05_final_shots.js`、`tools/p07_shots_trained.js`）。
4. 只在评估点（每 N 步）才更新的预览如果标了 `live`，验证器会警告「live 但没变化」。要么让它每帧都有可见变化（例如画出「当前步」游标），要么不标 `live` 而在数据变化时由 stage 写入新数据（它们仍会被重画）。
5. `PL.draw.plot`：给了 `o.x` 时，没有自带 `x` 数组的 series 会被拉伸铺满整个 x 范围——画「前 k 步」的部分曲线必须传 `x`。
6. probe / pick 拿不到 canvas：把绘图函数返回的布局（`LM.draw.heat` / `scatter` 的返回值）存进 `S`，在 probe 里用 `info.dx / info.dy` 反算。
7. V8：不要让 `Float64Array` 和 `Float32Array` 走同一个热函数（内联缓存变成多态，训练慢 1.7 倍；第 02 篇的做法是把核函数源码编译两份）。
8. 改任何参数都会让 `S.epoch` 自增、所有 history stage 重置；如果某个参数不该重置主训练，自己用参数 key 判断（第 02 篇的做法），但要保证 `--deps` 扫描结果可复现。`--deps` 看不到「只影响长期训练结果」的参数时，手写 `param.steps`。
9. 嵌入离线训练好的权重 / checkpoint 时，生成它的 Node 脚本放 `tools/pNN_*.js`，用页面里同一份代码，页面里写明「离线训练 N 步，脚本 xxx」。
10. **只构建、验证你自己的页面**，不要运行 `tools/build_all.mjs`（它会重建所有人的页面）。指向尚未完成的新篇的 broken link 是预期的。
11. 第一批的经验：真数据比完美叙事重要。实验结果不如预期时（例如第 06 篇真实模型上的 DPO 变差了），如实呈现并解释，再给出最接近的忠实版本。

## 4. 各篇提纲（建议；核心内容要覆盖，具体步骤作者可调整，6–12 步）

### 08 · 卷积神经网络：从卷积核到 U-Net（PyTorch）
数据：`LM.digits`（与第 02 篇同一批数字，便于和 MLP 对比）。
1. 卷积 = 共享权重的局部线性层：卷积核在图上滑动；Sobel / 拉普拉斯核作用在数字上的真实特征图
2. 为什么用卷积：参数量对比（MLP 256→64 vs 3×3 卷积）、平移等变（平移输入 → 特征图跟着平移；MLP 不行——真实演示）
3. 多通道、步长、padding、池化；感受野随层数增长（用反向传播求一个输出单元对输入的梯度 = 有效感受野，真实计算）
4. 训练一个小 CNN（live）：与第 02 篇 MLP 比准确率 / 参数量；第一层学到的卷积核可视化
5. 逐层特征图：选中样本在每层的激活
6. 平移鲁棒性 / 数据增强：平移后的测试集上 CNN vs MLP（真实）
7. 编码器-解码器与 U-Net：上采样、跳连；训练一个小 U-Net 做去噪（或分割数字笔画），与无跳连版本对比（为第 11 篇的去噪网络铺垫）
8. 残差连接与归一化：深 CNN 可训练性（可与 7 合并）
9. 教学版 → 生产版：LeNet → AlexNet → VGG → ResNet、检测（YOLO 思路）/ 分割（U-Net、SAM）、CNN vs ViT（链到第 07 篇）、扩散里的 U-Net

### 09 · 潜空间：自编码器、VAE 与 VQ-VAE（PyTorch）
数据：`LM.digits`。
1. 压缩的直觉：PCA = 线性自编码器，k 个主成分的重建（真实）
2. 自编码器（live）：编码 → 2 维瓶颈 → 解码；潜空间散点按数字上色；重建
3. 自编码器潜空间的「洞」：在潜空间网格上解码，簇之间是乱码（真实）
4. VAE（live）：μ、logσ²、重参数化、ELBO = 重建 + KL；潜空间变连续，网格解码平滑，从先验采样
5. β 的权衡：重建 vs KL（率-失真）；后验坍缩
6. 插值：AE vs VAE 在两个数字之间插值
7. VQ-VAE（live）：码本、最近邻量化、直通估计、commitment loss；码本使用率、死码；每张图变成一串离散 token（链到第 07、13 篇）
8. 潜空间里的生成：在 VAE 潜空间 / VQ token 上拟合一个先验并采样新数字（链到第 11 篇的潜空间扩散）
9. 教学版 → 生产版：Stable Diffusion 的 VAE（8× 下采样）、VQGAN（感知 + 对抗损失）、RVQ（SoundStream / EnCodec）、FSQ、视频 VAE——数字只写有把握的

### 10 · 强化学习：从价值迭代到策略梯度（Python / NumPy，深度部分用 PyTorch）
环境：网格世界（墙、目标、陷阱 / 悬崖，可带随机滑动）；可选一个连续小环境。
1. MDP：状态、动作、转移、奖励、折扣 γ；采样轨迹可视化
2. 回报与价值：随机策略的策略评估（迭代求 V^π，热图）
3. 贝尔曼最优方程与价值迭代（live 收敛）、贪心策略箭头；γ 的影响
4. 策略迭代（可合并）
5. 无模型：Q-learning（live，ε-greedy）；悬崖行走上 SARSA vs Q-learning 的经典路线差异（真实）
6. 探索与利用：多臂赌博机上 ε-greedy vs UCB（真实）
7. 函数逼近 / DQN 思路：小网络 Q 值、经验回放、目标网络（规模允许就真实训练）
8. 策略梯度 REINFORCE：方差、基线（真实）
9. Actor-Critic、优势函数、PPO 的裁剪目标（链到第 06 篇的 RLHF / GRPO）
10. 教学版 → 生产版：DQN Atari、AlphaGo / AlphaZero（MCTS + 价值 / 策略网络）、RLHF、离线 RL、模型式 RL（链到第 14 篇）

### 11 · 扩散模型与流匹配：从噪声里生成数据（PyTorch）
核心用二维点云（看得清向量场和轨迹），可再加一个小图像扩散（如 8×8 / 16×16 的数字或图形）。
1. 生成模型要做什么：从样本学 p(x)、为什么难
2. 前向加噪 q(x_t | x_0)：噪声日程（linear vs cosine）、ᾱ_t、SNR；点云一步步变成高斯（真实）
3. 训练目标：预测噪声 ε 的 MSE、时间步 embedding；实时训练一个小 MLP 去噪器；各 t 的损失
4. 学到了什么：不同 t 的去噪方向 / score 场（箭头），Tweedie 公式给出的 x̂₀
5. DDPM 祖先采样：大量粒子的反向轨迹（随机性）
6. DDIM / 概率流 ODE：确定性轨迹；采样步数（10 / 50 / 1000）与质量（用真实的分布距离度量，如 MMD / 最近邻距离）
7. 条件生成与 Classifier-Free Guidance：条件丢弃训练、引导强度 w：更像但更单一（真实）
8. 流匹配 / Rectified Flow：直线插值路径、预测速度场；轨迹更直、少步数（与 DDPM 轨迹对比，真实）
9. 图像上的扩散（小规模真实训练，或接第 09 篇潜空间做「潜空间扩散」）
10. 教学版 → 生产版：U-Net vs DiT、潜空间扩散（Stable Diffusion）、文本条件（交叉注意力）、采样器（DPM-Solver 等）、少步蒸馏（consistency models）、流匹配在新一代模型中的使用——只写有把握的

### 12 · 视频生成：时空扩散与时间一致性（PyTorch）
数据：程序生成的小视频（如 8 帧 × 12×12 或 16×16，小球匀速运动 / 弹墙，可多个球、可带方向类别）。
1. 视频 = 时空张量：帧条带、x-y-t 时空体（`mesh3d` 画运动轨迹「管子」很合适）
2. 压缩与 token：逐帧 patch vs 时空 patch（tubelet）的 token 数；潜空间视频（时间方向也压缩）
3. 逐帧独立生成的问题：每帧单独采样拼起来 → 闪烁、不连贯（真实，给一个时间一致性指标）
4. 时空联合去噪（live）：去噪网络同时看整段视频（如空间 + 时间分解的小网络），生成连贯运动（真实训练）
5. 物理一致性：生成视频里速度是否恒定、反弹是否正确（真实度量）；失败案例
6. 条件生成：图生视频（固定首帧）、类别 / 方向条件 + CFG（真实）
7. 长视频：自回归分块续写，误差累积 / 漂移（真实）
8. 规模账：token 数 = T·H·W / patch³、注意力的平方代价，为什么要潜空间 + 时空 patch（真实计算表）
9. 教学版 → 生产版：Sora 技术报告（时空 patch、DiT、「世界模拟器」的说法与局限）、Stable Video Diffusion、因果 3D VAE、时间注意力、流匹配、评估（FVD）——只写有把握的

### 13 · 音频与语音：频谱、音频 token 与生成（PyTorch / NumPy）
数据：页面里合成的声音（音阶旋律、啁啾、共振峰合成的元音、噪声…）。可以用 WebAudio 加「▶ 播放」按钮（必须用户点击触发；放在 section 或 html 块里；验证器无声运行，不能依赖播放）。
1. 声音 = 波形：采样率、奈奎斯特、混叠（真实演示）、量化
2. STFT 与频谱图：窗长 / 跳长、时频分辨率权衡（参数可调，真实）
3. Mel 频谱：滤波器组、对数压缩（真实）
4. 从频谱回到波形：Griffin-Lim 相位重建（真实迭代）vs 神经声码器
5. 神经音频编解码器：帧特征 → 残差向量量化 RVQ（k-means 码本，实时训练）→ 码率 vs 量化层数 → 重建音频（可播放、真实）
6. 音频 token 语言模型：用 `LM.gpt` 在 codec token / 音符 token 上训练（live），采样 → 解码 → 播放；多码本的 delay pattern（MusicGen 思路）
7. 扩散生成音频：在 mel 频谱上做扩散（AudioLDM 思路，规模允许就真实）
8. 语音合成（TTS）流水线：文本 → 音素 → 时长 / 声学特征 → 声码器（用共振峰合成做真实的小版本）
9. 语音识别（ASR）：log-mel → 编码器（Whisper 思路）；小 CNN 识别合成元音（真实）
10. 教学版 → 生产版：SoundStream / EnCodec、AudioLM、MusicGen、VALL-E、Whisper、流式低延迟——只写有把握的

### 14 · 世界模型：学会模拟环境，在想象中规划（PyTorch）
环境：一个简单的、渲染成小图（如 12×12 / 16×16）的环境：网格或带惯性的小球、墙、目标；离散或连续动作。
1. 什么是世界模型：p(o_{t+1} | o_≤t, a_t)；无模型 vs 有模型（链到第 10 篇）
2. 收集经验：随机策略的轨迹数据集
3. 编码器 V：把观测压进潜空间（链到第 09 篇，真实训练）
4. 动力学模型 M：z_{t+1} = f(z_t, a_t)（MLP / GRU；确定性 + 随机部分，RSSM 思路），实时训练，一步预测误差
5. 想象：开环 rollout vs 真实帧并排；误差随预测步数复合增长（真实曲线）；多步训练 vs teacher forcing
6. 在想象中规划：潜空间里用 CEM / random shooting 做 MPC，执行到真实环境，成功率（真实）
7. 在想象中学策略（Dreamer 思路，简化版）
8. 不确定性与模型被利用：规划器钻模型漏洞（真实演示）、集成 / 随机潜变量
9. 可交互的世界模型：动作条件的帧预测——**让读者用方向键在模型的「想象」里玩，旁边是真实环境**（Genie / GameNGen 的思路，强烈建议做）
10. 教学版 → 生产版：World Models（Ha & Schmidhuber 2018）、PlaNet / Dreamer（RSSM）、MuZero、Genie、GameNGen、「视频模型即世界模拟器」的争论——只写有把握的

### 15 · 3D 重建：NeRF 与 3D Gaussian Splatting（PyTorch；读者多是图形 / 游戏工程师，可以讲得更硬核）
1. 问题设定：多视角图像 → 3D 表示；针孔相机与光线（`mesh3d` 画相机、视锥、光线）
2. 体渲染方程：σ、c、透射率 T、离散化的 alpha 合成（对一个已知解析体积真实光线步进）
3. 谱偏差与位置编码 / Fourier features：MLP 拟合一张 2D 图，有无位置编码对比（live，经典演示；链到第 03 篇的正弦编码）
4. 一个小 NeRF：在低分辨率的合成多视角图上实时训练或离线训练 + 页面续训（预算紧就降分辨率、射线数、采样点数并写明）
5. 3D Gaussian Splatting：高斯参数（μ、Σ = R S Sᵀ Rᵀ、颜色、不透明度）、投影到屏幕（EWA、投影雅可比）、排序与 alpha 混合、可微光栅化
6. 拟合：2D 高斯拟合一张图（live），以及少量 3D 高斯拟合多视角（规模允许）；`mesh3d` 把高斯画成椭球
7. 自适应密度控制：克隆 / 分裂 / 剪枝
8. NeRF vs 3DGS：速度、质量、显存、编辑性（真实测量本页两种实现）
9. 教学版 → 生产版：Instant-NGP 哈希网格、Mip-NeRF、3DGS 实时渲染、球谐颜色、动态场景、游戏引擎里的使用——只写有把握的

## 5. 交付报告（每篇作者）

同 `DESIGN.md` §5：文件路径；步骤清单；验证器最终输出（error / warning / 每帧耗时，注明机器负载下的抖动）；简化与约定差异；在 Node 里验证过的数字；未解决的问题；希望共享库增加的东西。
