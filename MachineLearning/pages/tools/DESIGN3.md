# 代码改用 C++：规则、共享库与工具（全系列适用）

用户要求：**文档里所有代码块都写成 C++**（C++17 + 标准库），尽量不用第三方库（不用 PyTorch / NumPy / Eigen…）；可以自己写公用库代码，太长的部分可以用「函数原型 + 注释说明用途」的伪代码代替。

## 1. 共享 C++ 库：`pages/cpp/minml_*.h`（只读，真实可编译）

| 文件 | 内容 |
|---|---|
| `minml_core.h` | `Rng`（mulberry32，与页面 JS 的 `LM.rng` 逐位相同）`makeRng(seed)` `uniform(r)` `randn(r)` `randint(r, n)`；`Tensor`（二维行主序 `rows/cols/v`，`at(i, j)`）`zeros` `full` `randn(rows, cols, std, r)` `fromVector`；`floats(n, x)` `ints(n, x)`；`softmaxInPlace` `argmaxIndex` `sampleIndex` `permutation` |
| `minml_autograd.h` | 录带式自动微分（与 `LM.AG` 一一对应）：`Node` / `Var`（`x->val`、`x->grad`）、`param(t)` `constant(t)` `backward(loss)` `clearTape()` `NoGrad`（RAII）`gradOf(x)` `record(...)`（写自定义算子用）；算子 `matmul` `matmulBT` `addRow` `linear(X, W, b)` `add` `sub` `mul` `scale` `addConst` `relu` `tanhAct` `sigmoid` `gelu` `silu` `expAct` `logAct` `softplus` `logSigmoid` `square` `activation(A, Act)` `meanAll` `sumAll` `transpose` `gatherRows` `embedding(ids, E)` `concatRows` `concatCols` `sliceCols` `meanRows` `layerNorm` `l2Normalize` `attention(Q, K, V, B, T, H, causal, probsOut)` `crossEntropy(L, targets, weights, denom, rowLossOut)` `mse` `seqLogProb`；卷积族 `ConvSpec` `conv(C, H, W, K, stride, pad)` `convOutH/W` `conv2d(X, W, b, spec)` `avgPool2(X, C, H, W)` `maxPool2` `upsample2` `addChannel(X, B, C)` `mulChannel` |
| `minml_nn.h` | `heInit` `xavierInit`；`Linear` `makeLinear(in, out, std, r)` `forward(lin, x)`；`zeroGrad` `clipGradNorm`；`SGD` `makeSGD` `sgdStep`；`AdamW` `makeAdamW(params, decay, lr, wd, b1, b2, eps)` `adamwStep(opt, lr)`；`cosineLR(it, maxLR, minLR, warmup, total)` |
| `minml_gpt.h` | 迷你 GPT（与 `LM.gpt` 逐参数对应）：`GPTConfig` `Block` `GPT` `makeGPT(cfg, seed)` `numParams` `blockForward` `gptForward(m, ids, B, T, inputEmb)` `Batch` `getBatch` `trainStep(m, opt, batch, B, T, lr)` `decodeDist(logits, temperature, topK, topP)` `KVCache` `gptStep` `generate(m, prompt, maxNew, temperature, topK, topP, r, greedy, stopId)`；`Vocab` `utf8Chars` `encode` `decode` |

- 已验证：`cpp/minml_test.cpp`（`clang++ -std=c++17 -O2 -Wall minml_test.cpp`）对每个算子做有限差分梯度校验；同一个 seed 下迷你 GPT 的初始权重、4 步 AdamW 后的 loss（小数点后 6 位）和采样结果与页面 JS 完全一致（`tools/cpp_crosscheck.js`）。
- 页面里用：`llm_common.js` 的 `LM.CPP.core / autograd / nn / gpt` 是这些头文件的原文（由 `tools/gen_cpp.js` 生成，**不要手改**）。放进 `graph.common`：
  ```js
  common: { title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['autograd', 'nn']).concat([{ label: '━ 本页公用 ━', code: `...` }]) },
  ```
  `LM.CPP.blocks([...])` 自动补齐依赖（`gpt` → `nn` → `autograd` → `core`）。只用到普通数组循环的页面（第 01、02 篇）用 `['core']` 就够。
- 库里缺你需要的东西：写在本页的代码块 / `graph.common` 的「本页公用」块里（自由函数 + 结构体）。确实通用的，报告里提出来，不要改 `cpp/*.h`。

## 2. 写法规则（为了悬停跳转、未定义函数检查、能编译）

1. `codeLang: 'cpp'`；代码块 `lang` 不写（默认跟随）或写 `'cpp'`。HLSL / CUDA 片段照常写 `lang: 'hlsl'` 或 `'cuda'`（不参与 C++ 编译检查）。
2. 只有**自由函数调用** `foo(x)` 会被链接到定义（悬停看实现、点击跳转）并做「未定义 / 参数个数」检查；`obj.method()`、`ns::f()`、`std::sort()` 不链接也不检查。读者需要能查到的东西写成自由函数。
3. 全部定义写在全局作用域：**不用 namespace、不用 template**（框架的符号索引认不出模板函数 / 命名空间里的函数）；返回值按值返回（`Tensor f()`，不要 `const Tensor& f()`）；返回类型不要用嵌套模板（`std::vector<std::vector<float>>` 认不出 → 包成 struct）。 返回类型也不要用两个词的类型名（`long long`、`unsigned int` 认不出 → 用 `int64_t`、`uint32_t`）。另外注意：`GPT c = m;` 只复制了句柄，c 和 m 共享同一组参数；要独立副本（例如做消融）需要深拷贝每个 `Var` 的 `val`。
4. **不要用直接初始化** `std::vector<float> x(n);` —— 框架会把 `x(n)` 当成函数调用报「未定义函数 x」。写 `std::vector<float> x = floats(n);`、`Tensor t = zeros(r, c);`、`std::vector<int> v = ints(n, -1);`。同理，构造函数初始化列表 `: a(x)` 也别用，在函数体里赋值。
5. 不要按名字调用 lambda / 函数指针变量（`auto f = [](...){}; f(1);` 会被报未定义）；需要复用就写成自由函数。直接传给 `std::sort` 之类的 lambda 没问题。
6. 每个代码块都写成**完整的顶层定义**（函数 / 结构体 / 常量），不要写游离的语句片段；一段「主循环」也包进一个函数（如 `void trainLoop(...)`）。
7. **先定义后使用**：步骤代码里调用的辅助函数，要么出现在更早的代码块，要么放进 `graph.common` / `library`。
8. 太长、和主题无关的部分（读文件、画图、多线程…）写函数原型 + 一行注释说明用途，例如 `Image loadPng(const char* path);   // 读取 PNG 图片（实现略）`——这算定义，不会报未定义。引擎 / 平台提供的函数也可以列进 `externs`。
9. 风格：注释用中文；变量名和公式里的符号对应（x、W、b、logits、loss…）；单精度 `float`，需要累加精度时用 `double`；数学函数用 `std::exp` / `std::sqrt`。
10. 正文里提到「Python / NumPy / PyTorch 版」的地方改成「C++ 版」；介绍真实世界做法时可以说「在 PyTorch 里对应 `F.cross_entropy`」，但代码块里只放 C++。

## 3. 工具（在 `F:\Docs\MachineLearning\pages` 下）

- **编译检查**：`node tools/check_cpp.js <page>.lab.js` —— 把本页全部 C++ 代码块（graph.common 里除库以外的块 → library → 各步骤 → 节点详情，按这个顺序）拼成一个翻译单元，`#include` 全部 `cpp/minml_*.h`，用 VS 自带的 clang++ 做 `-fsyntax-only`。报错会标出是哪个代码块的第几行。故意写成片段的块在第一行加 `// @nocompile` 跳过（尽量少用）。**交付前必须通过。**
- 验证器：未定义函数（C++ 是真检查）必须为 0；`--write-deps` 不受影响（代码块不改变预览）。
- 只构建、验证你自己的页面；不要运行 `tools/build_all.mjs`，不要改 `llm_common.js`、`cpp/*.h`、DESIGN 文件和别人的页面。

## 4. PyTorch / NumPy → 本库 速查

| PyTorch / NumPy | minml（C++） |
|---|---|
| `torch.randn(r, c) * s` | `randn(r, c, s, rng)` |
| `torch.zeros(r, c)` / `np.zeros` | `zeros(r, c)`（`Tensor`）；一维用 `floats(n)` |
| `nn.Parameter(t)` / `requires_grad=True` | `Var p = param(t);` |
| `nn.Linear(i, o)` | `Linear l = makeLinear(i, o, std, rng);` `forward(l, x)` 或 `linear(x, W, b)` |
| `F.relu / gelu / silu / torch.tanh / sigmoid` | `relu / gelu / silu / tanhAct / sigmoid` |
| `x @ W`、`a @ b.T` | `matmul(x, W)`、`matmulBT(a, b)` |
| `nn.LayerNorm` | `layerNorm(x, gamma, beta)` |
| `nn.Embedding` | `embedding(ids, E)` |
| `F.cross_entropy(logits, y, ignore_index=-1)` | `crossEntropy(logits, y)`（y 里 −1 = 忽略） |
| `F.mse_loss` | `mse(a, b)` |
| `F.conv2d(x, w, b, stride, padding)` | `conv2d(x, w, b, conv(C, H, W, K, stride, pad))` |
| `F.avg_pool2d(x, 2)` / `max_pool2d` / `F.interpolate(scale_factor=2)` | `avgPool2(x, C, H, W)` / `maxPool2` / `upsample2` |
| `loss.backward()` | `backward(loss);` |
| `opt.zero_grad()` / `opt.step()` | `zeroGrad(params);` / `adamwStep(opt, lr);` 或 `sgdStep(opt, lr);` |
| `torch.nn.utils.clip_grad_norm_(params, 1.0)` | `clipGradNorm(params, 1.0f);` |
| `with torch.no_grad():` | `{ NoGrad ng; ... }` |
| `torch.softmax(z / T, -1)`（不求梯度） | `softmaxInPlace(z, T)` |
| `torch.multinomial(p, 1)` | `sampleIndex(p, rng)` |
| `np.random.permutation(n)` | `permutation(n, rng)` |
