// minml_test.cpp — 本系列 C++ 库的自检：① 每个可微算子的有限差分梯度校验；② 迷你 GPT 与页面 JS 实现的逐步对照。
// 编译：clang++ -std=c++17 -O2 -Wall minml_test.cpp -o minml_test   （MSVC：cl /std:c++17 /O2 /EHsc minml_test.cpp）
#include "minml_gpt.h"

static int g_fail = 0;

// 用一个固定的随机投影把任意张量变成标量：L = Σ Y ⊙ R
static Var project(const Var& Y) {
    Rng r = makeRng(9);
    Tensor R = zeros(Y->val.rows, Y->val.cols);
    for (float& x : R.v) x = (float)(uniform(r) - 0.5);
    return sumAll(mul(Y, constant(R)));
}

// 对 inputs 里每个张量的若干元素做中心差分，和反向传播的结果比较
static void gradCheck(const char* name, std::function<Var()> build, std::vector<Var> inputs, float eps = 1e-2f) {
    clearTape();
    for (Var& p : inputs) p->grad = Tensor();
    Var loss = build();
    backward(loss);
    double worst = 0;
    for (Var& p : inputs) {
        Tensor ana = p->grad;
        int n = p->val.size();
        for (int c = 0; c < std::min(n, 40); c++) {
            int k = (c * 7919) % n;
            float old = p->val.v[k];
            float lp, lm;
            { NoGrad ng; p->val.v[k] = old + eps; lp = build()->val.v[0]; p->val.v[k] = old - eps; lm = build()->val.v[0]; }
            p->val.v[k] = old;
            double num = (lp - lm) / (2.0 * eps), a = ana.v.empty() ? 0.0 : ana.v[k];
            double rel = std::fabs(num - a) / std::max(1e-3, std::fabs(num) + std::fabs(a));
            worst = std::max(worst, rel);
        }
    }
    bool ok = worst < 0.05;
    if (!ok) g_fail++;
    std::printf("%s %-22s worst rel err %.2e\n", ok ? "OK  " : "FAIL", name, worst);
}

static Var P(int r, int c, float s, Rng& rng, float off = 0.f) {
    Tensor t = randn(r, c, s, rng);
    for (float& x : t.v) x += off;
    return param(t);
}

int main() {
    Rng rng = makeRng(42);
    { Var A = P(3, 4, 1, rng), B = P(4, 5, 1, rng); gradCheck("matmul", [&] { return project(matmul(A, B)); }, {A, B}); }
    { Var A = P(3, 4, 1, rng), B = P(5, 4, 1, rng); gradCheck("matmulBT", [&] { return project(matmulBT(A, B)); }, {A, B}); }
    { Var X = P(3, 4, 1, rng), W = P(4, 2, 1, rng), b = P(1, 2, 1, rng); gradCheck("linear", [&] { return project(linear(X, W, b)); }, {X, W, b}); }
    { Var A = P(3, 4, 1, rng), B = P(3, 4, 1, rng); gradCheck("add/sub/mul", [&] { return project(mul(add(A, B), sub(A, B))); }, {A, B}); }
    { Var A = P(3, 4, 1.5f, rng); gradCheck("relu", [&] { return project(relu(A)); }, {A}, 1e-3f); }
    { Var A = P(3, 4, 1, rng); gradCheck("tanh/sigmoid", [&] { return project(add(tanhAct(A), sigmoid(A))); }, {A}); }
    { Var A = P(3, 4, 1, rng); gradCheck("gelu/silu", [&] { return project(add(gelu(A), silu(A))); }, {A}); }
    { Var A = P(3, 4, 0.5f, rng, 2.f); gradCheck("exp/log", [&] { return project(add(expAct(A), logAct(A))); }, {A}); }
    { Var A = P(3, 4, 3, rng); gradCheck("softplus/logsig", [&] { return project(add(softplus(A), logSigmoid(A))); }, {A}); }
    { Var A = P(3, 4, 1, rng); gradCheck("square/scale/addC", [&] { return project(addConst(scale(square(A), 0.7f), 0.3f)); }, {A}); }
    { Var A = P(3, 4, 1, rng); gradCheck("mean/transpose", [&] { return add(meanAll(A), project(transpose(A))); }, {A}); }
    { Var E = P(5, 3, 1, rng); std::vector<int> idx = {1, 3, 1, 0}; gradCheck("gatherRows", [&] { return project(gatherRows(E, idx)); }, {E}); }
    { Var A = P(2, 3, 1, rng), B = P(3, 3, 1, rng); gradCheck("concatRows", [&] { return project(concatRows(A, B)); }, {A, B}); }
    { Var A = P(3, 4, 1, rng), B = P(3, 2, 1, rng); gradCheck("concat/sliceCols", [&] { return project(sliceCols(concatCols(A, B), 1, 5)); }, {A, B}); }
    { Var X = P(6, 3, 1, rng); gradCheck("meanRows", [&] { return project(meanRows(X, 3)); }, {X}); }
    { Var X = P(3, 6, 1, rng), g = P(1, 6, 1, rng), b = P(1, 6, 1, rng); gradCheck("layerNorm", [&] { return project(layerNorm(X, g, b)); }, {X, g, b}); }
    { Var X = P(3, 4, 1, rng); gradCheck("l2Normalize", [&] { return project(l2Normalize(X)); }, {X}); }
    { Var Q = P(8, 8, 1, rng), K = P(8, 8, 1, rng), V = P(8, 8, 1, rng);
      gradCheck("attention causal", [&] { return project(attention(Q, K, V, 2, 4, 2, true)); }, {Q, K, V});
      gradCheck("attention full", [&] { return project(attention(Q, K, V, 2, 4, 2, false)); }, {Q, K, V}); }
    { Var L = P(4, 6, 1, rng); std::vector<int> t = {1, -1, 5, 0}; std::vector<float> w = {1, 1, -0.5f, 2};
      gradCheck("crossEntropy", [&] { return crossEntropy(L, t); }, {L});
      gradCheck("CE weighted", [&] { return crossEntropy(L, t, &w, 3.f); }, {L}); }
    { Var L = P(5, 6, 1, rng); std::vector<int> t = {1, 2, 5, 0, 3}, rs = {0, 0, 1, -1, 1}; gradCheck("seqLogProb", [&] { return project(seqLogProb(L, t, rs, 2)); }, {L}); }
    { Var A = P(3, 4, 1, rng), B = P(3, 4, 1, rng); gradCheck("mse", [&] { return mse(A, B); }, {A, B}); }
    { Var X = P(2, 2 * 6 * 5, 1, rng), W = P(3, 2 * 9, 0.5f, rng), b = P(1, 3, 1, rng);
      gradCheck("conv2d pad1", [&] { return project(conv2d(X, W, b, conv(2, 6, 5, 3, 1, 1))); }, {X, W, b});
      gradCheck("conv2d stride2", [&] { return project(conv2d(X, W, b, conv(2, 6, 5, 3, 2, 1))); }, {X, W, b}); }
    { Var X = P(2, 2 * 4 * 6, 1, rng);
      gradCheck("avgPool2", [&] { return project(avgPool2(X, 2, 4, 6)); }, {X});
      gradCheck("maxPool2", [&] { return project(maxPool2(X, 2, 4, 6)); }, {X}, 1e-3f);
      gradCheck("upsample2", [&] { return project(upsample2(X, 2, 4, 6)); }, {X}); }
    { Var X = P(2, 3 * 5, 1, rng), Bc = P(2, 3, 1, rng), Sc = P(2, 3, 1, rng);
      gradCheck("add/mulChannel", [&] { return project(mulChannel(addChannel(X, Bc, 3), Sc, 3)); }, {X, Bc, Sc}); }
    { GPTConfig c; c.V = 11; c.T = 5; c.d = 8; c.L = 2; c.H = 2; c.ff = 16; c.std = 0.3f;
      GPT m = makeGPT(c, 3);
      std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, y = {2, 3, 4, 5, 6, 7, 8, 9, 10, 0};
      std::vector<Var> some; for (size_t i = 0; i < m.params.size(); i += 3) some.push_back(m.params[i]);
      gradCheck("gpt full", [&] { return crossEntropy(gptForward(m, x, 2, 5), y); }, some); }

    // ── 与 JS 的逐步对照（tools/cpp_crosscheck.js 用同样的设置跑一遍 llm_common.js） ──
    {
        Rng r = makeRng(42);
        double u0 = uniform(r), u1 = uniform(r), u2 = uniform(r);   // 分开写：函数参数的求值顺序在 C++ 里是不确定的
        std::printf("rng(42) first uniforms: %.9f %.9f %.9f\n", u0, u1, u2);
        GPTConfig c;                                             // V 76, T 32, d 32, L 2, H 4, ff 128
        GPT m = makeGPT(c, 1);
        int B = 2, T = 32;
        Batch b; b.x.resize(B * T); b.y.resize(B * T);
        for (int i = 0; i < B * T; i++) { b.x[i] = (i * 7) % 76; b.y[i] = (i * 7 + 3) % 76; }
        AdamW opt = makeAdamW(m.params, m.decay, 3e-3f, 0.1f);
        std::printf("params %d\n", numParams(m));
        for (int it = 0; it < 4; it++) std::printf("step %d loss %.6f\n", it, trainStep(m, opt, b, B, T, 3e-3f));
        Rng gr = makeRng(5);
        std::vector<int> g = generate(m, {4, 5, 6}, 8, 1.0f, 0, 1.0f, gr);
        std::printf("generate:");
        for (int id : g) std::printf(" %d", id);
        std::printf("\n");
        // KV cache 对照完整前向：最后一个位置的 logits 应该一致
        std::vector<int> ids = {4, 5, 6, 7, 8};
        KVCache kc; kc.k.assign(c.L, std::vector<float>(c.T * c.d, 0.f)); kc.v.assign(c.L, std::vector<float>(c.T * c.d, 0.f));
        std::vector<float> lg;
        for (int t = 0; t < 5; t++) lg = gptStep(m, kc, ids[t], t);
        Var full;
        { NoGrad ng; full = gptForward(m, ids, 1, 5); }
        float md = 0;
        for (int vv = 0; vv < c.V; vv++) md = std::max(md, std::fabs(lg[vv] - full->val.v[4 * c.V + vv]));
        std::printf("kv-cache vs full forward max |dlogit| = %.2e\n", md);
        if (md > 1e-4f) g_fail++;
    }
    std::printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
