// minml_autograd.h — 极简 C++ 机器学习库 · 第 2 部分：自动微分（与页面 JS 的 LM.AG 一一对应）
// 思路：每个算子算出前向结果，同时把「怎样把输出的梯度传回输入」写成一个闭包，压进全局 tape；
// backward(loss) 从 loss 出发逆序执行这些闭包（链式法则），每个参数的 grad 里就累加出了 ∂loss/∂参数。
#pragma once
#include "minml_core.h"

// 计算图里的一个节点：前向值 + 损失对它的梯度（同形状）+ 反向传播闭包
struct Node {
    Tensor val, grad;
    bool requiresGrad = false;
    std::function<void(Node&)> backFn;                           // 由创建它的算子填写
};

// 节点的句柄（共享所有权）：Var x;  x->val 是数值，x->grad 是梯度
struct Var {
    std::shared_ptr<Node> n;
    Node* operator->() const { return n.get(); }
};

// 全局 tape：按前向顺序记录所有需要反向的节点
struct Tape { std::vector<Var> nodes; bool enabled = true; };

inline Tape& tape() { static Tape t; return t; }

inline Var makeVar(Tensor val) { Var x; x.n = std::make_shared<Node>(); x->val = std::move(val); return x; }

// 可训练参数（requiresGrad = true，梯度会累加进 ->grad）
inline Var param(Tensor init) { Var p = makeVar(std::move(init)); p->requiresGrad = true; return p; }

// 常量输入（数据、标签），不求梯度
inline Var constant(Tensor t) { return makeVar(std::move(t)); }

// 取梯度缓冲区（第一次用时分配成全 0）
inline Tensor& gradOf(const Var& x) {
    if ((int)x->grad.v.size() != x->val.size()) x->grad = zeros(x->val.rows, x->val.cols);
    return x->grad;
}

// 生成算子的输出节点：只要有一个输入需要梯度、且没在 NoGrad 里，就记进 tape
inline Var record(Tensor val, std::vector<Var> inputs, std::function<void(Node&)> back) {
    Var out = makeVar(std::move(val));
    bool need = false;
    for (const Var& x : inputs) need = need || x->requiresGrad;
    if (need && tape().enabled) { out->requiresGrad = true; out->backFn = back; tape().nodes.push_back(out); }
    return out;
}

// 反向传播：∂loss/∂loss = 1，然后逆序执行 tape（loss 必须是 1×1）
inline void backward(const Var& loss) {
    Tensor& g = gradOf(loss);
    std::fill(g.v.begin(), g.v.end(), 0.f);
    g.v[0] = 1.f;
    for (int i = (int)tape().nodes.size() - 1; i >= 0; --i) {
        Node& nd = *tape().nodes[i].n;
        if (!nd.grad.v.empty()) nd.backFn(nd);
    }
    tape().nodes.clear();
}

// 丢弃 tape（只做前向、不反向时调用，否则 tape 会一直变长）
inline void clearTape() { tape().nodes.clear(); }

// RAII：作用域内不记录 tape —— 推理 / 评估时用（相当于 PyTorch 的 torch.no_grad()）
struct NoGrad {
    bool prev = true;
    NoGrad() { prev = tape().enabled; tape().enabled = false; }
    ~NoGrad() { tape().enabled = prev; }
};

// ═══════════════════════════ 矩阵乘与线性层 ═══════════════════════════
// C = A·B    A[n×k]  B[k×m]  →  C[n×m]
inline Var matmul(const Var& A, const Var& B) {
    const Tensor &a = A->val, &b = B->val;
    int n = a.rows, k = a.cols, m = b.cols;
    Tensor c = zeros(n, m);
    for (int i = 0; i < n; i++)
        for (int p = 0; p < k; p++) {
            float av = a.v[i * k + p];
            if (av == 0) continue;
            for (int j = 0; j < m; j++) c.v[i * m + j] += av * b.v[p * m + j];
        }
    return record(std::move(c), {A, B}, [A, B](Node& o) {
        const Tensor &a = A->val, &b = B->val;
        int n = a.rows, k = a.cols, m = b.cols;
        if (A->requiresGrad) {                                   // dA = dC · Bᵀ
            Tensor& ga = gradOf(A);
            for (int i = 0; i < n; i++)
                for (int p = 0; p < k; p++) {
                    float s = 0;
                    for (int j = 0; j < m; j++) s += o.grad.v[i * m + j] * b.v[p * m + j];
                    ga.v[i * k + p] += s;
                }
        }
        if (B->requiresGrad) {                                   // dB = Aᵀ · dC
            Tensor& gb = gradOf(B);
            for (int i = 0; i < n; i++)
                for (int p = 0; p < k; p++) {
                    float av = a.v[i * k + p];
                    if (av == 0) continue;
                    for (int j = 0; j < m; j++) gb.v[p * m + j] += av * o.grad.v[i * m + j];
                }
        }
    });
}

// C = A·Bᵀ    A[n×k]  B[m×k]  →  C[n×m]（权重共享的 LM head、CLIP 的相似度矩阵）
inline Var matmulBT(const Var& A, const Var& B) {
    const Tensor &a = A->val, &b = B->val;
    int n = a.rows, k = a.cols, m = b.rows;
    Tensor c = zeros(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += a.v[i * k + p] * b.v[j * k + p];
            c.v[i * m + j] = s;
        }
    return record(std::move(c), {A, B}, [A, B](Node& o) {
        const Tensor &a = A->val, &b = B->val;
        int n = a.rows, k = a.cols, m = b.rows;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                float g = o.grad.v[i * m + j];
                if (g == 0) continue;
                if (A->requiresGrad) { Tensor& ga = gradOf(A); for (int p = 0; p < k; p++) ga.v[i * k + p] += g * b.v[j * k + p]; }
                if (B->requiresGrad) { Tensor& gb = gradOf(B); for (int p = 0; p < k; p++) gb.v[j * k + p] += g * a.v[i * k + p]; }
            }
    });
}

// Y = X + b：b 是一行 [1×m]，加到 X 的每一行上（偏置）
inline Var addRow(const Var& X, const Var& b) {
    Tensor y = X->val;
    int n = y.rows, m = y.cols;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[i * m + j] += b->val.v[j];
    return record(std::move(y), {X, b}, [X, b](Node& o) {
        int n = o.val.rows, m = o.val.cols;
        if (X->requiresGrad) { Tensor& gx = gradOf(X); for (int i = 0; i < n * m; i++) gx.v[i] += o.grad.v[i]; }
        if (b->requiresGrad) { Tensor& gb = gradOf(b); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) gb.v[j] += o.grad.v[i * m + j]; }
    });
}

// 线性层 Y = X·W + b    X[n×in]  W[in×out]  b[1×out]
inline Var linear(const Var& X, const Var& W, const Var& b) { return addRow(matmul(X, W), b); }

// ═══════════════════════════ 逐元素运算 ═══════════════════════════
// 逐元素加 / 减 / 乘（形状相同）
inline Var add(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] += B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i];
            if (B->requiresGrad) gradOf(B).v[i] += o.grad.v[i];
        }
    });
}

inline Var sub(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] -= B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i];
            if (B->requiresGrad) gradOf(B).v[i] -= o.grad.v[i];
        }
    });
}

inline Var mul(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] *= B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i] * B->val.v[i];   // ∂(ab)/∂a = b
            if (B->requiresGrad) gradOf(B).v[i] += o.grad.v[i] * A->val.v[i];
        }
    });
}

// 逐元素一元运算：激活函数等。actF = 前向 y = f(x)，actD = 导数 dy/dx（可以用 x，也可以用已算好的 y）
enum Act { ACT_RELU, ACT_TANH, ACT_SIGMOID, ACT_GELU, ACT_SILU, ACT_EXP, ACT_LOG, ACT_SOFTPLUS, ACT_LOGSIGMOID, ACT_SQUARE };

inline float actF(Act a, float x) {
    const float k = 0.7978845608f;                                // √(2/π)，GELU 的 tanh 近似（GPT-2 同款）
    switch (a) {
        case ACT_RELU: return x > 0 ? x : 0.f;
        case ACT_TANH: return std::tanh(x);
        case ACT_SIGMOID: return 1.f / (1.f + std::exp(-x));
        case ACT_GELU: return 0.5f * x * (1 + std::tanh(k * (x + 0.044715f * x * x * x)));
        case ACT_SILU: return x / (1.f + std::exp(-x));
        case ACT_EXP: return std::exp(x);
        case ACT_LOG: return std::log(x);
        case ACT_SOFTPLUS: return x > 20 ? x : x < -20 ? std::exp(x) : std::log1p(std::exp(x));
        case ACT_LOGSIGMOID: return x >= 0 ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));   // log σ(x)，数值稳定
        case ACT_SQUARE: return x * x;
    }
    return x;
}

inline float actD(Act a, float x, float y) {
    const float k = 0.7978845608f;
    switch (a) {
        case ACT_RELU: return x > 0 ? 1.f : 0.f;
        case ACT_TANH: return 1 - y * y;
        case ACT_SIGMOID: return y * (1 - y);
        case ACT_GELU: { float t = std::tanh(k * (x + 0.044715f * x * x * x)); return 0.5f * (1 + t) + 0.5f * x * (1 - t * t) * k * (1 + 3 * 0.044715f * x * x); }
        case ACT_SILU: { float s = 1.f / (1.f + std::exp(-x)); return s * (1 + x * (1 - s)); }
        case ACT_EXP: return y;
        case ACT_LOG: return 1.f / x;
        case ACT_SOFTPLUS: return 1.f / (1.f + std::exp(-x));
        case ACT_LOGSIGMOID: return 1.f / (1.f + std::exp(x));
        case ACT_SQUARE: return 2 * x;
    }
    return 1.f;
}

inline Var activation(const Var& A, Act act) {
    Tensor y = A->val;
    for (float& x : y.v) x = actF(act, x);
    return record(std::move(y), {A}, [A, act](Node& o) {
        Tensor& ga = gradOf(A);
        for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i] * actD(act, A->val.v[i], o.val.v[i]);
    });
}

inline Var relu(const Var& A) { return activation(A, ACT_RELU); }
inline Var tanhAct(const Var& A) { return activation(A, ACT_TANH); }
inline Var sigmoid(const Var& A) { return activation(A, ACT_SIGMOID); }
inline Var gelu(const Var& A) { return activation(A, ACT_GELU); }
inline Var silu(const Var& A) { return activation(A, ACT_SILU); }
inline Var expAct(const Var& A) { return activation(A, ACT_EXP); }
inline Var logAct(const Var& A) { return activation(A, ACT_LOG); }
inline Var softplus(const Var& A) { return activation(A, ACT_SOFTPLUS); }
inline Var logSigmoid(const Var& A) { return activation(A, ACT_LOGSIGMOID); }
inline Var square(const Var& A) { return activation(A, ACT_SQUARE); }

// Y = s·X（常数缩放）
inline Var scale(const Var& A, float s) {
    Tensor y = A->val;
    for (float& x : y.v) x *= s;
    return record(std::move(y), {A}, [A, s](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i] * s; });
}

// Y = X + c（常数）
inline Var addConst(const Var& A, float c) {
    Tensor y = A->val;
    for (float& x : y.v) x += c;
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i]; });
}

// ═══════════════════════════ 归约与形状 ═══════════════════════════
// 所有元素的平均 → 1×1
inline Var meanAll(const Var& A) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (float x : A->val.v) s += x;
    y.v[0] = (float)(s / A->val.size());
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); float g = o.grad.v[0] / A->val.size(); for (float& x : ga.v) x += g; });
}

// 所有元素的和 → 1×1
inline Var sumAll(const Var& A) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (float x : A->val.v) s += x;
    y.v[0] = (float)s;
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); for (float& x : ga.v) x += o.grad.v[0]; });
}

// 转置 [n×m] → [m×n]
inline Var transpose(const Var& A) {
    int n = A->val.rows, m = A->val.cols;
    Tensor y = zeros(m, n);
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[j * n + i] = A->val.v[i * m + j];
    return record(std::move(y), {A}, [A, n, m](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) ga.v[i * m + j] += o.grad.v[j * n + i]; });
}

// 取行：Y[i] = X[idx[i]]（embedding 查表就是它）；反向 = 按行累加回去
inline Var gatherRows(const Var& X, const std::vector<int>& idx) {
    int n = (int)idx.size(), d = X->val.cols;
    Tensor y = zeros(n, d);
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) y.v[i * d + j] = X->val.v[idx[i] * d + j];
    return record(std::move(y), {X}, [X, idx, n, d](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) gx.v[idx[i] * d + j] += o.grad.v[i * d + j]; });
}

inline Var embedding(const std::vector<int>& ids, const Var& E) { return gatherRows(E, ids); }

// 按行拼接 [A; B]（列数相同）
inline Var concatRows(const Var& A, const Var& B) {
    Tensor y = zeros(A->val.rows + B->val.rows, A->val.cols);
    std::copy(A->val.v.begin(), A->val.v.end(), y.v.begin());
    std::copy(B->val.v.begin(), B->val.v.end(), y.v.begin() + A->val.size());
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        int na = A->val.size();
        if (A->requiresGrad) { Tensor& ga = gradOf(A); for (int i = 0; i < na; i++) ga.v[i] += o.grad.v[i]; }
        if (B->requiresGrad) { Tensor& gb = gradOf(B); for (int i = 0; i < B->val.size(); i++) gb.v[i] += o.grad.v[na + i]; }
    });
}

// 按列拼接 [A | B]（行数相同）：U-Net 跳连 = 特征图的通道拼接，因为一张图就是一行
inline Var concatCols(const Var& A, const Var& B) {
    int n = A->val.rows, ca = A->val.cols, cb = B->val.cols;
    Tensor y = zeros(n, ca + cb);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < ca; j++) y.v[i * (ca + cb) + j] = A->val.v[i * ca + j];
        for (int j = 0; j < cb; j++) y.v[i * (ca + cb) + ca + j] = B->val.v[i * cb + j];
    }
    return record(std::move(y), {A, B}, [A, B, n, ca, cb](Node& o) {
        for (int i = 0; i < n; i++) {
            if (A->requiresGrad) for (int j = 0; j < ca; j++) gradOf(A).v[i * ca + j] += o.grad.v[i * (ca + cb) + j];
            if (B->requiresGrad) for (int j = 0; j < cb; j++) gradOf(B).v[i * cb + j] += o.grad.v[i * (ca + cb) + ca + j];
        }
    });
}

// 取第 [c0, c1) 列（把一个线性层的输出拆成 μ / logσ²、GRU 的几个门…）
inline Var sliceCols(const Var& A, int c0, int c1) {
    int n = A->val.rows, c = A->val.cols, m = c1 - c0;
    Tensor y = zeros(n, m);
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[i * m + j] = A->val.v[i * c + c0 + j];
    return record(std::move(y), {A}, [A, n, c, m, c0](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) ga.v[i * c + c0 + j] += o.grad.v[i * m + j]; });
}

// 每 k 行取平均：[(n·k) × d] → [n × d]（把一张图的所有 patch token 平均成一个向量）
inline Var meanRows(const Var& X, int k) {
    int n = X->val.rows / k, d = X->val.cols;
    Tensor y = zeros(n, d);
    for (int i = 0; i < n; i++) for (int r = 0; r < k; r++) for (int j = 0; j < d; j++) y.v[i * d + j] += X->val.v[(i * k + r) * d + j] / k;
    return record(std::move(y), {X}, [X, n, k, d](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < n; i++) for (int r = 0; r < k; r++) for (int j = 0; j < d; j++) gx.v[(i * k + r) * d + j] += o.grad.v[i * d + j] / k; });
}

// ═══════════════════════════ 归一化 ═══════════════════════════
// LayerNorm（对每一行）：y = γ·(x − μ)/√(σ² + ε) + β
inline Var layerNorm(const Var& X, const Var& gamma, const Var& beta, float eps = 1e-5f) {
    int n = X->val.rows, d = X->val.cols;
    Tensor y = zeros(n, d), xhat = zeros(n, d);
    std::vector<float> rstd = floats(n);
    for (int i = 0; i < n; i++) {
        const float* x = &X->val.v[i * d];
        float mu = 0; for (int j = 0; j < d; j++) mu += x[j]; mu /= d;
        float var = 0; for (int j = 0; j < d; j++) var += (x[j] - mu) * (x[j] - mu); var /= d;
        rstd[i] = 1.f / std::sqrt(var + eps);
        for (int j = 0; j < d; j++) { xhat.v[i * d + j] = (x[j] - mu) * rstd[i]; y.v[i * d + j] = xhat.v[i * d + j] * gamma->val.v[j] + beta->val.v[j]; }
    }
    return record(std::move(y), {X, gamma, beta}, [X, gamma, beta, xhat, rstd, n, d](Node& o) {
        for (int i = 0; i < n; i++) {
            float s1 = 0, s2 = 0;
            for (int j = 0; j < d; j++) {
                float g = o.grad.v[i * d + j], dh = g * gamma->val.v[j];
                s1 += dh; s2 += dh * xhat.v[i * d + j];
                if (gamma->requiresGrad) gradOf(gamma).v[j] += g * xhat.v[i * d + j];
                if (beta->requiresGrad) gradOf(beta).v[j] += g;
            }
            if (X->requiresGrad) {
                s1 /= d; s2 /= d;
                Tensor& gx = gradOf(X);
                for (int j = 0; j < d; j++) gx.v[i * d + j] += rstd[i] * (o.grad.v[i * d + j] * gamma->val.v[j] - s1 - xhat.v[i * d + j] * s2);
            }
        }
    });
}

// 每一行除以自己的 L2 范数（CLIP 的嵌入）
inline Var l2Normalize(const Var& X) {
    int n = X->val.rows, d = X->val.cols;
    Tensor y = zeros(n, d);
    std::vector<float> inv = floats(n);
    for (int i = 0; i < n; i++) {
        double s = 0; for (int j = 0; j < d; j++) s += X->val.v[i * d + j] * X->val.v[i * d + j];
        inv[i] = (float)(1.0 / std::sqrt(s + 1e-12));
        for (int j = 0; j < d; j++) y.v[i * d + j] = X->val.v[i * d + j] * inv[i];
    }
    return record(std::move(y), {X}, [X, inv, n, d](Node& o) {
        Tensor& gx = gradOf(X);
        for (int i = 0; i < n; i++) {
            float dot = 0; for (int j = 0; j < d; j++) dot += o.grad.v[i * d + j] * o.val.v[i * d + j];
            for (int j = 0; j < d; j++) gx.v[i * d + j] += inv[i] * (o.grad.v[i * d + j] - o.val.v[i * d + j] * dot);
        }
    });
}

// ═══════════════════════════ 注意力 ═══════════════════════════
// 多头缩放点积注意力。Q/K/V: [B·T × D]，H 个头，每头 dh = D/H；causal = true 时位置 i 只看 j ≤ i。
// probsOut（可选）收集注意力权重：下标 ((b·H + h)·T + i)·T + j
inline Var attention(const Var& Q, const Var& K, const Var& Vv, int B, int T, int H, bool causal, std::vector<float>* probsOut = nullptr) {
    int D = Q->val.cols, dh = D / H;
    float sc = 1.f / std::sqrt((float)dh);
    Tensor out = zeros(B * T, D);
    std::vector<float> P = floats(B * H * T * T), row = floats(T);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < T; i++) {
                int jmax = causal ? i : T - 1;
                const float* q = &Q->val.v[(b * T + i) * D + h * dh];
                float m = -1e30f;
                for (int j = 0; j <= jmax; j++) {
                    const float* k = &K->val.v[(b * T + j) * D + h * dh];
                    float s = 0; for (int c = 0; c < dh; c++) s += q[c] * k[c];
                    row[j] = s * sc; m = std::max(m, row[j]);
                }
                float z = 0; for (int j = 0; j <= jmax; j++) { row[j] = std::exp(row[j] - m); z += row[j]; }
                float* o = &out.v[(b * T + i) * D + h * dh];
                for (int j = 0; j <= jmax; j++) {
                    float p = row[j] / z;
                    P[((size_t)(b * H + h) * T + i) * T + j] = p;
                    const float* v = &Vv->val.v[(b * T + j) * D + h * dh];
                    for (int c = 0; c < dh; c++) o[c] += p * v[c];
                }
            }
    if (probsOut) *probsOut = P;
    return record(std::move(out), {Q, K, Vv}, [Q, K, Vv, P, B, T, H, causal, dh, sc](Node& o) {
        int D = H * dh;
        std::vector<float> dP = floats(T);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < T; i++) {
                    int jmax = causal ? i : T - 1;
                    const float* g = &o.grad.v[(b * T + i) * D + h * dh];
                    const float* pr = &P[((size_t)(b * H + h) * T + i) * T];
                    float dot = 0;
                    for (int j = 0; j <= jmax; j++) {                // dV += pᵢⱼ · dO；dP = dO · V
                        const float* v = &Vv->val.v[(b * T + j) * D + h * dh];
                        float s = 0;
                        for (int c = 0; c < dh; c++) { s += g[c] * v[c]; if (Vv->requiresGrad) gradOf(Vv).v[(b * T + j) * D + h * dh + c] += pr[j] * g[c]; }
                        dP[j] = s; dot += pr[j] * s;
                    }
                    for (int j = 0; j <= jmax; j++) {                // softmax 反向：dS = P ⊙ (dP − Σ P·dP)
                        float ds = pr[j] * (dP[j] - dot) * sc;
                        if (ds == 0) continue;
                        for (int c = 0; c < dh; c++) {
                            if (Q->requiresGrad) gradOf(Q).v[(b * T + i) * D + h * dh + c] += ds * K->val.v[(b * T + j) * D + h * dh + c];
                            if (K->requiresGrad) gradOf(K).v[(b * T + j) * D + h * dh + c] += ds * Q->val.v[(b * T + i) * D + h * dh + c];
                        }
                    }
                }
    });
}

// ═══════════════════════════ 损失函数 ═══════════════════════════
// 交叉熵（自带 softmax）。logits[n×V]，targets[n]（−1 = 忽略该行）。weights（可空）= 每行权重，可为负（策略梯度的 advantage）；
// denom ≤ 0 时自动取有效行数（或 Σ|w|）。返回 1×1。rowLossOut（可选）收集每行的 −log p。
inline Var crossEntropy(const Var& L, const std::vector<int>& targets, const std::vector<float>* weights = nullptr, float denom = 0.f, std::vector<float>* rowLossOut = nullptr) {
    int n = L->val.rows, V = L->val.cols;
    Tensor P = zeros(n, V);
    double tot = 0, cnt = 0;
    if (rowLossOut) rowLossOut->assign(n, NAN);
    for (int i = 0; i < n; i++) {
        const float* z = &L->val.v[i * V];
        float m = z[0]; for (int j = 1; j < V; j++) m = std::max(m, z[j]);
        double s = 0; for (int j = 0; j < V; j++) { P.v[i * V + j] = std::exp(z[j] - m); s += P.v[i * V + j]; }
        for (int j = 0; j < V; j++) P.v[i * V + j] = (float)(P.v[i * V + j] / s);
        int t = targets[i];
        if (t < 0) continue;
        double li = -(z[t] - m - std::log(s)), w = weights ? (*weights)[i] : 1.0;
        if (rowLossOut) (*rowLossOut)[i] = (float)li;
        tot += w * li; cnt += weights ? std::fabs(w) : 1.0;
    }
    float den = denom > 0 ? denom : (float)std::max(cnt, 1.0);
    Tensor y = zeros(1, 1); y.v[0] = (float)(tot / den);
    std::vector<float> w = weights ? *weights : std::vector<float>();   // 复制一份：反向时调用方的数组可能已经不在了
    return record(std::move(y), {L}, [L, P, targets, w, den, n, V](Node& o) {
        Tensor& gl = gradOf(L);
        for (int i = 0; i < n; i++) {
            int t = targets[i];
            if (t < 0) continue;
            float k = o.grad.v[0] * (w.empty() ? 1.f : w[i]) / den;
            for (int j = 0; j < V; j++) gl.v[i * V + j] += k * P.v[i * V + j];   // ∂/∂z = (softmax − onehot) · w / denom
            gl.v[i * V + t] -= k;
        }
    });
}

// 均方误差 mean((A − B)²)
inline Var mse(const Var& A, const Var& B) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (int i = 0; i < A->val.size(); i++) s += (A->val.v[i] - B->val.v[i]) * (A->val.v[i] - B->val.v[i]);
    int n = A->val.size();
    y.v[0] = (float)(s / n);
    return record(std::move(y), {A, B}, [A, B, n](Node& o) {
        float k = 2 * o.grad.v[0] / n;
        for (int i = 0; i < n; i++) {
            float d = A->val.v[i] - B->val.v[i];
            if (A->requiresGrad) gradOf(A).v[i] += k * d;
            if (B->requiresGrad) gradOf(B).v[i] -= k * d;
        }
    });
}

// 每条序列的 Σ log p(目标)：rowSeq[i] = 第 i 行属于第几条序列（−1 = 不计）。返回 [nSeq × 1]（DPO 用）
inline Var seqLogProb(const Var& L, const std::vector<int>& targets, const std::vector<int>& rowSeq, int nSeq) {
    int n = L->val.rows, V = L->val.cols;
    Tensor P = zeros(n, V), y = zeros(nSeq, 1);
    for (int i = 0; i < n; i++) {
        int s = rowSeq[i], t = targets[i];
        if (s < 0 || t < 0) continue;
        const float* z = &L->val.v[i * V];
        float m = z[0]; for (int j = 1; j < V; j++) m = std::max(m, z[j]);
        double sum = 0; for (int j = 0; j < V; j++) { P.v[i * V + j] = std::exp(z[j] - m); sum += P.v[i * V + j]; }
        for (int j = 0; j < V; j++) P.v[i * V + j] = (float)(P.v[i * V + j] / sum);
        y.v[s] += (float)(z[t] - m - std::log(sum));
    }
    return record(std::move(y), {L}, [L, P, targets, rowSeq, n, V](Node& o) {
        Tensor& gl = gradOf(L);
        for (int i = 0; i < n; i++) {
            int s = rowSeq[i], t = targets[i];
            if (s < 0 || t < 0) continue;
            float k = o.grad.v[s];
            for (int j = 0; j < V; j++) gl.v[i * V + j] -= k * P.v[i * V + j];
            gl.v[i * V + t] += k;
        }
    });
}

// ═══════════════════════════ 卷积族（一张图 = 一行，通道优先 C×H×W） ═══════════════════════════
struct ConvSpec { int C = 1, H = 1, W = 1, KH = 3, KW = 3, stride = 1, padH = 0, padW = 0; };

// 方形卷积核的常用写法：conv(C, H, W, K, stride, pad)
inline ConvSpec conv(int C, int H, int W, int K, int stride = 1, int pad = 0) {
    ConvSpec s; s.C = C; s.H = H; s.W = W; s.KH = K; s.KW = K; s.stride = stride; s.padH = pad; s.padW = pad;
    return s;
}

inline int convOutH(const ConvSpec& s) { return (s.H + 2 * s.padH - s.KH) / s.stride + 1; }
inline int convOutW(const ConvSpec& s) { return (s.W + 2 * s.padW - s.KW) / s.stride + 1; }

// 二维卷积（互相关，与 PyTorch Conv2d 相同）。X[N × C·H·W]，Wt[Cout × C·KH·KW]，b[1 × Cout]
inline Var conv2d(const Var& X, const Var& Wt, const Var& b, const ConvSpec& s) {
    int N = X->val.rows, Co = Wt->val.rows, Ho = convOutH(s), Wo = convOutW(s);
    int inS = s.C * s.H * s.W, outS = Co * Ho * Wo, kS = s.C * s.KH * s.KW;
    Tensor y = zeros(N, outS);
    for (int n = 0; n < N; n++)
        for (int co = 0; co < Co; co++)
            for (int oy = 0; oy < Ho; oy++)
                for (int ox = 0; ox < Wo; ox++) {
                    float acc = b->val.v[co];
                    for (int ci = 0; ci < s.C; ci++)
                        for (int ky = 0; ky < s.KH; ky++) {
                            int iy = oy * s.stride - s.padH + ky;
                            if (iy < 0 || iy >= s.H) continue;
                            for (int kx = 0; kx < s.KW; kx++) {
                                int ix = ox * s.stride - s.padW + kx;
                                if (ix < 0 || ix >= s.W) continue;
                                acc += X->val.v[n * inS + (ci * s.H + iy) * s.W + ix] * Wt->val.v[co * kS + (ci * s.KH + ky) * s.KW + kx];
                            }
                        }
                    y.v[n * outS + (co * Ho + oy) * Wo + ox] = acc;
                }
    return record(std::move(y), {X, Wt, b}, [X, Wt, b, s, N, Co, Ho, Wo, inS, outS, kS](Node& o) {
        for (int n = 0; n < N; n++)
            for (int co = 0; co < Co; co++)
                for (int oy = 0; oy < Ho; oy++)
                    for (int ox = 0; ox < Wo; ox++) {
                        float g = o.grad.v[n * outS + (co * Ho + oy) * Wo + ox];
                        if (g == 0) continue;
                        if (b->requiresGrad) gradOf(b).v[co] += g;
                        for (int ci = 0; ci < s.C; ci++)
                            for (int ky = 0; ky < s.KH; ky++) {
                                int iy = oy * s.stride - s.padH + ky;
                                if (iy < 0 || iy >= s.H) continue;
                                for (int kx = 0; kx < s.KW; kx++) {
                                    int ix = ox * s.stride - s.padW + kx;
                                    if (ix < 0 || ix >= s.W) continue;
                                    int xi = n * inS + (ci * s.H + iy) * s.W + ix, wi = co * kS + (ci * s.KH + ky) * s.KW + kx;
                                    if (Wt->requiresGrad) gradOf(Wt).v[wi] += g * X->val.v[xi];
                                    if (X->requiresGrad) gradOf(X).v[xi] += g * Wt->val.v[wi];
                                }
                            }
                    }
    });
}

// 2×2 平均池化：[N × C·H·W] → [N × C·(H/2)·(W/2)]
inline Var avgPool2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, h = H / 2, w = W / 2;
    Tensor y = zeros(N, C * h * w);
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
        int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j;
        y.v[n * C * h * w + (c * h + i) * w + j] = 0.25f * (X->val.v[base] + X->val.v[base + 1] + X->val.v[base + W] + X->val.v[base + W + 1]);
    }
    return record(std::move(y), {X}, [X, N, C, H, W, h, w](Node& o) {
        Tensor& gx = gradOf(X);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
            int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j;
            float g = 0.25f * o.grad.v[n * C * h * w + (c * h + i) * w + j];
            gx.v[base] += g; gx.v[base + 1] += g; gx.v[base + W] += g; gx.v[base + W + 1] += g;
        }
    });
}

// 2×2 最大池化：梯度只回到每个窗口里最大的那个位置
inline Var maxPool2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, h = H / 2, w = W / 2;
    Tensor y = zeros(N, C * h * w);
    std::vector<int> arg = ints(y.size());
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
        int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j, cand[4] = {base, base + 1, base + W, base + W + 1}, best = cand[0];
        for (int k = 1; k < 4; k++) if (X->val.v[cand[k]] > X->val.v[best]) best = cand[k];
        int oi = n * C * h * w + (c * h + i) * w + j;
        y.v[oi] = X->val.v[best]; arg[oi] = best;
    }
    return record(std::move(y), {X}, [X, arg](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < o.val.size(); i++) gx.v[arg[i]] += o.grad.v[i]; });
}

// 最近邻 2× 上采样：[N × C·H·W] → [N × C·2H·2W]（U-Net 解码器）
inline Var upsample2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, H2 = 2 * H, W2 = 2 * W;
    Tensor y = zeros(N, C * H2 * W2);
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < H2; i++) for (int j = 0; j < W2; j++)
        y.v[n * C * H2 * W2 + (c * H2 + i) * W2 + j] = X->val.v[n * C * H * W + (c * H + i / 2) * W + j / 2];
    return record(std::move(y), {X}, [X, N, C, H, W, H2, W2](Node& o) {
        Tensor& gx = gradOf(X);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < H2; i++) for (int j = 0; j < W2; j++)
            gx.v[n * C * H * W + (c * H + i / 2) * W + j / 2] += o.grad.v[n * C * H2 * W2 + (c * H2 + i) * W2 + j];
    });
}

// 按通道广播加：X[N × C·HW] + Bc[N × C]（扩散 U-Net 里把时间步 embedding 加到每个通道上）
inline Var addChannel(const Var& X, const Var& Bc, int C) {
    int N = X->val.rows, HW = X->val.cols / C;
    Tensor y = X->val;
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int k = 0; k < HW; k++) y.v[n * C * HW + c * HW + k] += Bc->val.v[n * C + c];
    return record(std::move(y), {X, Bc}, [X, Bc, N, C, HW](Node& o) {
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
            float s = 0;
            for (int k = 0; k < HW; k++) { float g = o.grad.v[n * C * HW + c * HW + k]; s += g; if (X->requiresGrad) gradOf(X).v[n * C * HW + c * HW + k] += g; }
            if (Bc->requiresGrad) gradOf(Bc).v[n * C + c] += s;
        }
    });
}

// 按通道广播乘：X[N × C·HW] ⊙ Sc[N × C]（FiLM 式条件调制）
inline Var mulChannel(const Var& X, const Var& Sc, int C) {
    int N = X->val.rows, HW = X->val.cols / C;
    Tensor y = X->val;
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int k = 0; k < HW; k++) y.v[n * C * HW + c * HW + k] *= Sc->val.v[n * C + c];
    return record(std::move(y), {X, Sc}, [X, Sc, N, C, HW](Node& o) {
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
            float s = 0, sc = Sc->val.v[n * C + c];
            for (int k = 0; k < HW; k++) {
                int i = n * C * HW + c * HW + k;
                s += o.grad.v[i] * X->val.v[i];
                if (X->requiresGrad) gradOf(X).v[i] += o.grad.v[i] * sc;
            }
            if (Sc->requiresGrad) gradOf(Sc).v[n * C + c] += s;
        }
    });
}
