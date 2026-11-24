// minml_core.h — 本系列自带的极简 C++ 机器学习库 · 第 1 部分：随机数与二维张量
// 只用 C++17 标准库，单线程，不追求速度，只求每一行都看得懂。约定与页面里的 JS 参考实现（llm_common.js 的 LM.AG）完全相同：
//   · 张量一律是二维行主序 [rows × cols]；一个 batch 的序列展平成 [B·T × d]：第 b 条序列第 t 个位置 = 第 b·T + t 行
//   · 一张图 = 张量的一行，按通道优先展平 [C × H × W]（与 PyTorch 的 NCHW 相同）
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ── 随机数：mulberry32，与页面里 JS 的 LM.rng / PL.rng 产生逐位相同的序列 ──
struct Rng { uint32_t a = 1; };

inline Rng makeRng(uint32_t seed) { Rng r; r.a = seed ? seed : 1; return r; }

// 返回 [0, 1) 上的均匀随机数
inline double uniform(Rng& r) {
    r.a += 0x6D2B79F5u;
    uint32_t t = (r.a ^ (r.a >> 15)) * (1u | r.a);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

// 标准正态 N(0, 1)：Box-Muller（与 LM.randn 相同：消耗两个均匀数，只用 cos 那一半）
inline float randn(Rng& r) {
    double u1 = std::max(uniform(r), 1e-12), u2 = uniform(r);
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}

// [0, n) 上的随机整数
inline int randint(Rng& r, int n) { return (int)(uniform(r) * n); }

// ── 二维张量 ──
struct Tensor {
    int rows = 0, cols = 0;
    std::vector<float> v;                                        // rows × cols 个数，行主序
    float& at(int i, int j) { return v[(size_t)i * cols + j]; }
    float at(int i, int j) const { return v[(size_t)i * cols + j]; }
    int size() const { return rows * cols; }
};

inline Tensor zeros(int rows, int cols) {
    Tensor t; t.rows = rows; t.cols = cols; t.v.assign((size_t)rows * cols, 0.f);
    return t;
}

inline Tensor full(int rows, int cols, float x) {
    Tensor t = zeros(rows, cols);
    std::fill(t.v.begin(), t.v.end(), x);
    return t;
}

// 每个元素独立地 ~ N(0, std²)
inline Tensor randn(int rows, int cols, float std, Rng& r) {
    Tensor t = zeros(rows, cols);
    for (float& x : t.v) x = std * randn(r);
    return t;
}

inline Tensor fromVector(int rows, int cols, const std::vector<float>& data) {
    Tensor t = zeros(rows, cols);
    for (int i = 0; i < rows * cols; i++) t.v[i] = data[i];
    return t;
}

// n 个 float / int 组成的数组（统一用 std::vector<float> x = floats(n); 的写法，比 x(n) 更不容易和函数调用混淆）
inline std::vector<float> floats(int n, float x = 0.f) { return std::vector<float>((size_t)n, x); }
inline std::vector<int> ints(int n, int x = 0) { return std::vector<int>((size_t)n, x); }

// ── 不需要梯度的小工具（采样、评估用） ──
// x ← softmax(x / temperature)，原地；减去最大值防止 exp 溢出
inline void softmaxInPlace(std::vector<float>& x, float temperature) {
    float m = x[0];
    for (float a : x) m = std::max(m, a);
    double s = 0;
    for (float& a : x) { a = std::exp((a - m) / temperature); s += a; }
    for (float& a : x) a = (float)(a / s);
}

// 最大元素的下标
inline int argmaxIndex(const std::vector<float>& x) {
    int b = 0;
    for (int i = 1; i < (int)x.size(); i++) if (x[i] > x[b]) b = i;
    return b;
}

// 按概率 p 抽一个下标（p 之和为 1）
inline int sampleIndex(const std::vector<float>& p, Rng& r) {
    double u = uniform(r);
    for (int i = 0; i + 1 < (int)p.size(); i++) { u -= p[i]; if (u <= 0) return i; }
    return (int)p.size() - 1;
}

// 打乱 [0, n) 的一个排列（Fisher–Yates）
inline std::vector<int> permutation(int n, Rng& r) {
    std::vector<int> p = ints(n);
    for (int i = 0; i < n; i++) p[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = (int)(uniform(r) * (i + 1)); std::swap(p[i], p[j]); }
    return p;
}
