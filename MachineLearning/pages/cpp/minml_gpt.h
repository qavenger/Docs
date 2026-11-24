// minml_gpt.h — 极简 C++ 机器学习库 · 第 4 部分：迷你 GPT（与页面 JS 的 LM.gpt 逐个参数、逐个算子对应）
// 结构同 nanoGPT / GPT-2：可学习位置 embedding、Pre-LN、多头因果注意力、GELU(tanh) MLP（4×）、权重共享的 LM head。
#pragma once
#include "minml_nn.h"

struct GPTConfig {
    int V = 76;        // 词表大小（4 个特殊 token + 72 个汉字 / 标点）
    int T = 32;        // 上下文长度
    int d = 32;        // 残差流宽度
    int L = 2;         // Block 层数
    int H = 4;         // 注意力头数（每头 d / H = 8 维）
    int ff = 128;      // MLP 隐藏层宽度（4d）
    float std = 0.02f; // 初始化标准差（GPT-2）
};

// 一个 Transformer Block 的全部参数（W 都存成 [in × out]）
struct Block { Var ln1g, ln1b, Wq, bq, Wk, bk, Wv, bv, Wo, bo, ln2g, ln2b, W1, b1, W2, b2; };

struct GPT {
    GPTConfig cfg;
    Var wte, wpe;                                                // token / 位置 embedding：[V × d]、[T × d]
    std::vector<Block> blocks;
    Var lnfg, lnfb;                                              // 最后的 LayerNorm
    std::vector<Var> params;                                     // 全部参数（顺序与 JS 版相同）
    std::vector<bool> decay;                                     // 哪些参数做权重衰减（二维矩阵 / embedding）
};

// 新建一个参数并登记（顺序决定了随机数的消耗顺序：同一个 seed 得到与 JS 版逐位相同的初始权重）
inline Var addParam(GPT& m, Tensor init, bool decay) { Var p = param(std::move(init)); m.params.push_back(p); m.decay.push_back(decay); return p; }

// GPT-2 初始化：N(0, 0.02)；残差分支的输出投影（Wo、W2）再除以 √(2L)；LayerNorm γ = 1、β = 0；偏置 0
inline GPT makeGPT(const GPTConfig& c, uint32_t seed) {
    GPT m; m.cfg = c;
    Rng r = makeRng(seed);
    float s = c.std, rs = c.std / std::sqrt(2.f * c.L);
    m.wte = addParam(m, randn(c.V, c.d, s, r), true);
    m.wpe = addParam(m, randn(c.T, c.d, s, r), true);
    for (int l = 0; l < c.L; l++) {
        Block b;
        b.ln1g = addParam(m, full(1, c.d, 1.f), false); b.ln1b = addParam(m, zeros(1, c.d), false);
        b.Wq = addParam(m, randn(c.d, c.d, s, r), true); b.bq = addParam(m, zeros(1, c.d), false);
        b.Wk = addParam(m, randn(c.d, c.d, s, r), true); b.bk = addParam(m, zeros(1, c.d), false);
        b.Wv = addParam(m, randn(c.d, c.d, s, r), true); b.bv = addParam(m, zeros(1, c.d), false);
        b.Wo = addParam(m, randn(c.d, c.d, rs, r), true); b.bo = addParam(m, zeros(1, c.d), false);
        b.ln2g = addParam(m, full(1, c.d, 1.f), false); b.ln2b = addParam(m, zeros(1, c.d), false);
        b.W1 = addParam(m, randn(c.d, c.ff, s, r), true); b.b1 = addParam(m, zeros(1, c.ff), false);
        b.W2 = addParam(m, randn(c.ff, c.d, rs, r), true); b.b2 = addParam(m, zeros(1, c.d), false);
        m.blocks.push_back(b);
    }
    m.lnfg = addParam(m, full(1, c.d, 1.f), false); m.lnfb = addParam(m, zeros(1, c.d), false);
    return m;
}

inline int numParams(const GPT& m) { int n = 0; for (const Var& p : m.params) n += p->val.size(); return n; }

// 一个 Pre-LN Block：x ← x + Attn(LN₁(x))；x ← x + MLP(LN₂(x))
inline Var blockForward(const Block& b, const Var& x, int B, int T, int H) {
    Var h1 = layerNorm(x, b.ln1g, b.ln1b);
    Var q = linear(h1, b.Wq, b.bq), k = linear(h1, b.Wk, b.bk), v = linear(h1, b.Wv, b.bv);
    Var x1 = add(x, linear(attention(q, k, v, B, T, H, true), b.Wo, b.bo));
    Var h2 = layerNorm(x1, b.ln2g, b.ln2b);
    return add(x1, linear(gelu(linear(h2, b.W1, b.b1)), b.W2, b.b2));
}

// 前向：ids = B 条长度 T 的序列（展平）→ logits [B·T × V]。inputEmb 非空时替代 token embedding（多模态前缀）
inline Var gptForward(GPT& m, const std::vector<int>& ids, int B, int T, const Var* inputEmb = nullptr) {
    std::vector<int> pos = ints(B * T);
    for (int i = 0; i < B * T; i++) pos[i] = i % T;
    Var x = add(inputEmb ? *inputEmb : embedding(ids, m.wte), gatherRows(m.wpe, pos));
    for (const Block& b : m.blocks) x = blockForward(b, x, B, T, m.cfg.H);
    return matmulBT(layerNorm(x, m.lnfg, m.lnfb), m.wte);   // 权重共享：LM head 就是 wteᵀ
}

// ── 训练数据：随机取 B 个长度 T+1 的窗口，y 是 x 右移一位（每个位置都在预测下一个 token） ──
struct Batch { std::vector<int> x, y; };

inline Batch getBatch(const std::vector<int>& data, int B, int T, Rng& r) {
    Batch b; b.x.resize(B * T); b.y.resize(B * T);
    for (int i = 0; i < B; i++) {
        int s = (int)(uniform(r) * (data.size() - T - 1));
        for (int t = 0; t < T; t++) { b.x[i * T + t] = data[s + t]; b.y[i * T + t] = data[s + t + 1]; }
    }
    return b;
}

// 一步训练：前向 → 交叉熵 → 反向 → 梯度裁剪到 1.0 → AdamW。返回这一步的 loss
inline float trainStep(GPT& m, AdamW& opt, const Batch& b, int B, int T, float lr) {
    zeroGrad(m.params);
    Var loss = crossEntropy(gptForward(m, b.x, B, T), b.y);
    backward(loss);
    clipGradNorm(m.params, 1.0f);
    adamwStep(opt, lr);
    return loss->val.v[0];
}

// ── 解码：温度 → top-k → top-p（nucleus），返回重新归一化后的分布（没选中的为 0） ──
inline std::vector<float> decodeDist(std::vector<float> logits, float temperature, int topK, float topP) {
    int V = (int)logits.size();
    softmaxInPlace(logits, std::max(temperature, 1e-4f));
    std::vector<int> order = ints(V);
    for (int i = 0; i < V; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&logits](int a, int b) { return logits[a] > logits[b]; });
    std::vector<float> p = floats(V);
    double cum = 0, s = 0;
    for (int r = 0; r < V; r++) {
        if (topK > 0 && r >= topK) break;
        int i = order[r];
        p[i] = logits[i]; cum += logits[i];
        if (topP > 0 && topP < 1 && cum >= topP) break;
    }
    for (float x : p) s += x;
    for (float& x : p) x = (float)(x / s);
    return p;
}

// ── 推理用的逐 token 前向（不建计算图）+ KV cache：每个新 token 只算自己的 q/k/v，复用之前所有位置的 k/v ──
struct KVCache { std::vector<std::vector<float>> k, v; };       // 每层一份 [T × d]

// 一行的 LayerNorm（推理用）
inline void layerNormRow(const float* x, const Var& g, const Var& b, int d, float* out) {
    float mu = 0, var = 0;
    for (int j = 0; j < d; j++) mu += x[j];
    mu /= d;
    for (int j = 0; j < d; j++) var += (x[j] - mu) * (x[j] - mu);
    float rs = 1.f / std::sqrt(var / d + 1e-5f);
    for (int j = 0; j < d; j++) out[j] = (x[j] - mu) * rs * g->val.v[j] + b->val.v[j];
}

// 一行的线性层：out = x·W + b（推理用）
inline void linearRow(const float* x, const Var& W, const Var& b, int in, int outN, float* out) {
    for (int j = 0; j < outN; j++) out[j] = b->val.v[j];
    for (int p = 0; p < in; p++) { float xv = x[p]; if (xv == 0) continue; for (int j = 0; j < outN; j++) out[j] += xv * W->val.v[p * outN + j]; }
}

// 把位置 pos 的 token 喂进模型，返回下一个 token 的 logits（并把这一位置的 k / v 写进 cache）
inline std::vector<float> gptStep(const GPT& m, KVCache& c, int tok, int pos) {
    const GPTConfig& g = m.cfg;
    int d = g.d, dh = d / g.H;
    std::vector<float> x = floats(d), h = floats(d), q = floats(d), k = floats(d), v = floats(d), a = floats(d);
    std::vector<float> t = floats(std::max(d, g.ff)), f = floats(g.ff), att = floats(g.T), logits = floats(g.V);
    for (int j = 0; j < d; j++) x[j] = m.wte->val.v[tok * d + j] + m.wpe->val.v[pos * d + j];
    for (int l = 0; l < g.L; l++) {
        const Block& b = m.blocks[l];
        layerNormRow(x.data(), b.ln1g, b.ln1b, d, h.data());
        linearRow(h.data(), b.Wq, b.bq, d, d, q.data()); linearRow(h.data(), b.Wk, b.bk, d, d, k.data()); linearRow(h.data(), b.Wv, b.bv, d, d, v.data());
        std::copy(k.begin(), k.end(), c.k[l].begin() + pos * d);
        std::copy(v.begin(), v.end(), c.v[l].begin() + pos * d);
        std::fill(a.begin(), a.end(), 0.f);
        for (int hh = 0; hh < g.H; hh++) {
            float mx = -1e30f, z = 0;
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int e = 0; e < dh; e++) s += q[hh * dh + e] * c.k[l][j * d + hh * dh + e];
                att[j] = s / std::sqrt((float)dh); mx = std::max(mx, att[j]);
            }
            for (int j = 0; j <= pos; j++) { att[j] = std::exp(att[j] - mx); z += att[j]; }
            for (int j = 0; j <= pos; j++) for (int e = 0; e < dh; e++) a[hh * dh + e] += att[j] / z * c.v[l][j * d + hh * dh + e];
        }
        linearRow(a.data(), b.Wo, b.bo, d, d, t.data());
        for (int j = 0; j < d; j++) x[j] += t[j];
        layerNormRow(x.data(), b.ln2g, b.ln2b, d, h.data());
        linearRow(h.data(), b.W1, b.b1, d, g.ff, f.data());
        for (float& z : f) z = actF(ACT_GELU, z);
        linearRow(f.data(), b.W2, b.b2, g.ff, d, t.data());
        for (int j = 0; j < d; j++) x[j] += t[j];
    }
    layerNormRow(x.data(), m.lnfg, m.lnfb, d, h.data());
    for (int vv = 0; vv < g.V; vv++) { float s = 0; for (int j = 0; j < d; j++) s += h[j] * m.wte->val.v[vv * d + j]; logits[vv] = s; }
    return logits;
}

// 自回归生成：先把 prompt 喂进去，然后每步从 decodeDist 采样一个 token（greedy = 取最大），遇到 stopId 或上下文满了就停
inline std::vector<int> generate(const GPT& m, std::vector<int> prompt, int maxNew, float temperature, int topK, float topP, Rng& r, bool greedy = false, int stopId = -1) {
    const GPTConfig& g = m.cfg;
    KVCache c;
    c.k.assign(g.L, std::vector<float>(g.T * g.d, 0.f));
    c.v.assign(g.L, std::vector<float>(g.T * g.d, 0.f));
    std::vector<float> logits;
    for (int t = 0; t < (int)prompt.size() && t < g.T; t++) logits = gptStep(m, c, prompt[t], t);
    std::vector<int> out = prompt;
    for (int n = 0; n < maxNew && (int)out.size() < g.T; n++) {
        int id = greedy ? argmaxIndex(logits) : sampleIndex(decodeDist(logits, temperature, topK, topP), r);
        out.push_back(id);
        if (id == stopId) break;
        if ((int)out.size() < g.T) logits = gptStep(m, c, id, (int)out.size() - 1);
    }
    return out;
}

// ── 字符级词表（UTF-8）：一个汉字 / 标点 = 一个 token；特殊 token 写成 <eos> 这样的整体 ──
struct Vocab { std::vector<std::string> itos; };

// 把 UTF-8 字符串切成一个个字符（汉字占 3 个字节）
inline std::vector<std::string> utf8Chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        out.push_back(s.substr(i, n)); i += n;
    }
    return out;
}

inline int tokenId(const Vocab& v, const std::string& tok) {
    for (int i = 0; i < (int)v.itos.size(); i++) if (v.itos[i] == tok) return i;
    return -1;
}

inline std::vector<int> encode(const Vocab& v, const std::string& text) {
    std::vector<int> ids;
    std::vector<std::string> chars = utf8Chars(text);
    for (size_t i = 0; i < chars.size(); i++) {
        if (chars[i] == "<") {                                   // 特殊 token：<eos>、<user>、<bot>…
            std::string tok = "<";
            size_t j = i + 1;
            while (j < chars.size() && chars[j] != ">") tok += chars[j++];
            int id = tokenId(v, tok + ">");
            if (id >= 0) { ids.push_back(id); i = j; continue; }
        }
        ids.push_back(tokenId(v, chars[i]));
    }
    return ids;
}

inline std::string decode(const Vocab& v, const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) s += v.itos[id];
    return s;
}
