// minml_nn.h — 极简 C++ 机器学习库 · 第 3 部分：初始化、线性层、优化器、学习率调度（与页面 JS 的 LM.AdamW / LM.SGD / LM.cosineLR 相同）
#pragma once
#include "minml_autograd.h"

// ── 初始化 ──
// He / Kaiming：std = √(2 / fanIn)，适合 ReLU 一类激活（第 02 篇）
inline Tensor heInit(int rows, int cols, int fanIn, Rng& r) { return randn(rows, cols, std::sqrt(2.f / fanIn), r); }

// Xavier / Glorot：std = √(2 / (fanIn + fanOut))，适合 tanh / sigmoid
inline Tensor xavierInit(int rows, int cols, int fanIn, int fanOut, Rng& r) { return randn(rows, cols, std::sqrt(2.f / (fanIn + fanOut)), r); }

// ── 线性层：Y = X·W + b，W 存成 [in × out]（PyTorch 的 nn.Linear 存 [out × in]，算的是同一个东西） ──
struct Linear { Var W, b; };

inline Linear makeLinear(int in, int out, float std, Rng& r) { Linear l; l.W = param(randn(in, out, std, r)); l.b = param(zeros(1, out)); return l; }

inline Var forward(const Linear& l, const Var& x) { return linear(x, l.W, l.b); }

// ── 梯度工具 ──
inline void zeroGrad(std::vector<Var>& params) { for (Var& p : params) std::fill(p->grad.v.begin(), p->grad.v.end(), 0.f); }

// 所有梯度的全局 L2 范数；超过 maxNorm 就整体缩小到 maxNorm。返回裁剪前的范数
inline float clipGradNorm(std::vector<Var>& params, float maxNorm) {
    double s = 0;
    for (Var& p : params) for (float g : p->grad.v) s += (double)g * g;
    float norm = (float)std::sqrt(s);
    if (maxNorm > 0 && norm > maxNorm) {
        float k = maxNorm / (norm + 1e-6f);
        for (Var& p : params) for (float& g : p->grad.v) g *= k;
    }
    return norm;
}

// ── SGD（可带动量）：v ← μ·v + g；p ← p − lr·v ──
struct SGD { std::vector<Var> params; std::vector<Tensor> vel; float lr = 0.1f, momentum = 0.f; };

inline SGD makeSGD(std::vector<Var> params, float lr, float momentum = 0.f) {
    SGD o; o.params = params; o.lr = lr; o.momentum = momentum;
    for (Var& p : params) o.vel.push_back(zeros(p->val.rows, p->val.cols));
    return o;
}

inline void sgdStep(SGD& o, float lr) {
    for (size_t k = 0; k < o.params.size(); k++) {
        Var& p = o.params[k];
        if (p->grad.v.empty()) continue;
        for (int i = 0; i < p->val.size(); i++) { o.vel[k].v[i] = o.momentum * o.vel[k].v[i] + p->grad.v[i]; p->val.v[i] -= lr * o.vel[k].v[i]; }
    }
}

// ── AdamW（Loshchilov & Hutter：权重衰减与梯度的自适应缩放解耦）。decay[k] = 第 k 个参数要不要做权重衰减 ──
struct AdamW {
    std::vector<Var> params;
    std::vector<bool> decay;
    std::vector<Tensor> m, v;                                    // 一阶 / 二阶矩
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.1f;
    int t = 0;
};

inline AdamW makeAdamW(std::vector<Var> params, std::vector<bool> decay, float lr, float wd, float b1 = 0.9f, float b2 = 0.95f, float eps = 1e-8f) {
    AdamW o; o.params = params; o.decay = decay; o.lr = lr; o.wd = wd; o.b1 = b1; o.b2 = b2; o.eps = eps;
    for (Var& p : params) { o.m.push_back(zeros(p->val.rows, p->val.cols)); o.v.push_back(zeros(p->val.rows, p->val.cols)); }
    return o;
}

// 一步更新：m ← β₁m + (1−β₁)g；v ← β₂v + (1−β₂)g²；p ← p − lr·( m̂/(√v̂ + ε) + wd·p )，m̂、v̂ 是偏差修正后的值
inline void adamwStep(AdamW& o, float lr) {
    o.t++;
    double bc1 = 1 - std::pow((double)o.b1, o.t), bc2 = 1 - std::pow((double)o.b2, o.t);
    for (size_t k = 0; k < o.params.size(); k++) {
        Var& p = o.params[k];
        if (p->grad.v.empty()) continue;
        float wd = o.decay[k] ? o.wd : 0.f;
        for (int i = 0; i < p->val.size(); i++) {
            float g = p->grad.v[i];
            o.m[k].v[i] = o.b1 * o.m[k].v[i] + (1 - o.b1) * g;
            o.v[k].v[i] = o.b2 * o.v[k].v[i] + (1 - o.b2) * g * g;
            double mh = o.m[k].v[i] / bc1, vh = o.v[k].v[i] / bc2;
            p->val.v[i] -= (float)(lr * (mh / (std::sqrt(vh) + o.eps) + wd * p->val.v[i]));
        }
    }
}

// ── 学习率：线性 warmup → cosine 衰减到 minLR（nanoGPT 的 get_lr 同款） ──
inline float cosineLR(int it, float maxLR, float minLR, int warmup, int total) {
    if (it < warmup) return maxLR * (it + 1) / warmup;
    if (it >= total) return minLR;
    double r = (double)(it - warmup) / std::max(1, total - warmup);
    return (float)(minLR + 0.5 * (1 + std::cos(3.141592653589793 * r)) * (maxLR - minLR));
}
