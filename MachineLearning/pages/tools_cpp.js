// tools_cpp.js — 第 19–21 篇的 C++ 代码块（C++17 + 标准库 + 本系列的 minml 库；按页面顺序拼起来可以直接编译，tools/tk_cpp_check.js 检查）。
// 与页面里的 JS（tools_common.js）逐条对应：分词、BM25、哈希特征编码器、RRF、两级路由、BPE 计数、JSON Schema 校验、工具搜索。
const TK_CPP = {
    // ═══════════════ 第 19 篇 ═══════════════
    tooldef: `
        // 一个工具 = 名字 + 给模型看的描述 + 参数的 JSON Schema + 行为提示（MCP annotations）
        struct ToolDef {
            std::string server, name;           // MCP 服务器名 + 工具名；交给模型时拼成 mcp__server__name
            std::string description;            // 模型选工具时几乎只看这句话：写清楚「做什么、什么时候用」
            std::string inputSchemaJson;        // {"type":"object","properties":{…},"required":[…]}
            bool readOnly = false, destructive = false;
        };

        std::string apiName(const ToolDef& t) { return "mcp__" + t.server + "__" + t.name; }

        // 渲染成 Anthropic Messages API 的 tools[] 一项（OpenAI 的格式只是字段名不同：parameters / type:function）
        std::string toAnthropicJson(const ToolDef& t) {
            return "{\\"name\\":\\"" + apiName(t) + "\\",\\"description\\":\\"" + t.description + "\\",\\"input_schema\\":" + t.inputSchemaJson + "}";
        }`,
    bpecount: `
        // 字节级 BPE 计数器（第 06 篇的算法）：只为数 token，不需要还原文本
        struct BPECounter {
            std::unordered_map<uint32_t, int> rank;               // (a << 16 | b) → 合并顺序
            std::vector<std::pair<int, int>> merges;              // 第 r 条规则：a + b → 256 + r
        };

        std::vector<std::string> pretokenize(const std::string& text);   // cl100k 风格的预切分（正则；实现略）

        int countChunk(const BPECounter& B, const std::string& chunk) {
            std::vector<int> ids;
            ids.assign(chunk.begin(), chunk.end());
            for (int& v : ids) v &= 0xFF;
            while (ids.size() > 1) {
                int best = INT_MAX; size_t at = 0;
                for (size_t j = 0; j + 1 < ids.size(); j++) {
                    auto it = B.rank.find((uint32_t(ids[j]) << 16) | uint32_t(ids[j + 1]));
                    if (it != B.rank.end() && it->second < best) { best = it->second; at = j; }
                }
                if (best == INT_MAX) break;
                std::vector<int> out;
                for (size_t j = 0; j < ids.size(); j++) {
                    if (j + 1 < ids.size() && ids[j] == B.merges[best].first && ids[j + 1] == B.merges[best].second) { out.push_back(256 + best); j++; }
                    else out.push_back(ids[j]);
                }
                ids.swap(out);
                (void)at;
            }
            return (int)ids.size();
        }

        int countTokens(const BPECounter& B, const std::string& text) {
            int n = 0;
            for (const std::string& c : pretokenize(text)) n += countChunk(B, c);
            return n;
        }`,
    bm25: `
        // 词项：英文词（拆 snake_case / camelCase、去复数）+ 中文单字与相邻两字（实现略：规则见页面 JS 的 terms()）
        std::vector<std::string> terms(const std::string& text);

        struct BM25Index {
            int N = 0; float avgLen = 0, k1 = 1.2f, b = 0.75f;
            std::vector<float> len;                                                     // 每个文档的词项数
            std::unordered_map<std::string, std::vector<std::pair<int, int>>> postings; // 词 → [(文档, 出现次数)]
        };

        BM25Index buildBM25(const std::vector<std::string>& docs) {
            BM25Index I; I.N = (int)docs.size(); I.len.resize(I.N);
            for (int i = 0; i < I.N; i++) {
                std::unordered_map<std::string, int> tf;
                for (const std::string& t : terms(docs[i])) tf[t]++;
                for (const auto& kv : tf) { I.postings[kv.first].push_back({ i, kv.second }); I.len[i] += (float)kv.second; }
                I.avgLen += I.len[i] / (float)I.N;
            }
            return I;
        }

        // score(d, q) = Σ_{t∈q} idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avg|d|))
        std::vector<float> scoreBM25(const BM25Index& I, const std::string& query) {
            std::vector<float> s;
            s.assign(I.N, 0.f);
            std::set<std::string> qt; for (const std::string& t : terms(query)) qt.insert(t);
            for (const std::string& t : qt) {
                auto it = I.postings.find(t); if (it == I.postings.end()) continue;
                float n = (float)it->second.size(), idf = std::log(1.f + (I.N - n + 0.5f) / (n + 0.5f));
                for (const auto& p : it->second) { float tf = (float)p.second; s[p.first] += idf * tf * (I.k1 + 1) / (tf + I.k1 * (1 - I.b + I.b * I.len[p.first] / I.avgLen)); }
            }
            return s;
        }`,
    encoder: `
        // 哈希特征：英文词 / 词内三字母 / 中文字 / 相邻两字 → FNV-1a → 桶号（2^14 个桶）
        uint32_t fnv1a(const std::string& s) { uint32_t h = 2166136261u; for (unsigned char c : s) { h ^= c; h *= 16777619u; } return h; }
        using Features = std::vector<std::pair<int, float>>;                                   // (桶号, 权重)
        Features hashFeatures(const std::string& text, int buckets);                            // 规则同页面 JS 的 features()（实现略）

        // 编码器：e = Σ w_f·E[f] / ‖w‖ → h = tanh(e + b0) → z = W1·h + b1 → z / ‖z‖ （请求和工具描述用同一个编码器）
        struct Encoder { int NB = 1 << 14, D = 64, D2 = 64; std::vector<float> E, b0, W1, b1; };

        std::vector<float> embedText(const Encoder& m, const std::string& text) {
            std::vector<float> h, z;
            h.assign(m.D, 0.f); z.assign(m.D2, 0.f);
            float n2 = 0;
            for (const auto& f : hashFeatures(text, m.NB)) { for (int j = 0; j < m.D; j++) h[j] += m.E[(size_t)f.first * m.D + j] * f.second; n2 += f.second * f.second; }
            float inv = 1.f / std::sqrt(std::max(n2, 1.f)), zz = 0;
            for (int j = 0; j < m.D; j++) h[j] = std::tanh(h[j] * inv + m.b0[j]);
            for (int k = 0; k < m.D2; k++) { float s = m.b1[k]; for (int j = 0; j < m.D; j++) s += m.W1[(size_t)k * m.D + j] * h[j]; z[k] = s; zz += s * s; }
            for (float& v : z) v /= std::sqrt(zz) + 1e-8f;
            return z;
        }

        // 训练目标（批内负例的 InfoNCE）：一批 B 个 (请求, 工具)，第 i 个请求应该和第 i 个工具最像
        //   L = −(1/B) Σ_i log softmax_j(q_i·d_j / τ)_i      τ = 0.05
        float infoNCE(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& Dc, float tau) {
            const int B = (int)Q.size(); float L = 0;
            for (int i = 0; i < B; i++) {
                std::vector<float> s; s.resize(B); float m = -1e30f;
                for (int j = 0; j < B; j++) { float d = 0; for (size_t k = 0; k < Q[i].size(); k++) d += Q[i][k] * Dc[j][k]; s[j] = d / tau; m = std::max(m, s[j]); }
                float zsum = 0; for (float v : s) zsum += std::exp(v - m);
                L -= s[i] - m - std::log(zsum);
            }
            return L / B;
        }`,
    hybrid: `
        // 倒数排名融合（RRF）：只用名次，不用管两路分数的量纲；w = 每一路的权重
        std::vector<float> rrfFuse(const std::vector<std::vector<int>>& rankings, const std::vector<float>& w, int n, int k = 60) {
            std::vector<float> s;
            s.assign(n, 0.f);
            for (size_t l = 0; l < rankings.size(); l++)
                for (size_t r = 0; r < rankings[l].size(); r++) s[rankings[l][r]] += w[l] / (float)(k + r + 1);
            return s;
        }

        std::vector<int> topK(const std::vector<float>& s, int k) {
            std::vector<int> idx;
            idx.resize(s.size()); std::iota(idx.begin(), idx.end(), 0);
            k = std::min<int>(k, (int)idx.size());
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return s[a] > s[b]; });
            idx.resize(k);
            return idx;
        }`,
    router: `
        // 两级路由：先在 125 个「应用 / MCP 服务器」的描述里挑 m 个，再只在它们的工具里找
        std::vector<int> routeTwoStage(const Encoder& m, const std::string& query,
                                       const std::vector<std::vector<float>>& appEmb, const std::vector<std::vector<float>>& toolEmb,
                                       const std::vector<int>& toolApp, int keepApps, int k) {
            std::vector<float> q = embedText(m, query), as, ts;
            as.resize(appEmb.size()); ts.assign(toolEmb.size(), -1e30f);
            for (size_t a = 0; a < appEmb.size(); a++) { float d = 0; for (size_t j = 0; j < q.size(); j++) d += q[j] * appEmb[a][j]; as[a] = d; }
            std::vector<int> apps = topK(as, keepApps);
            std::set<int> keep;
            keep.insert(apps.begin(), apps.end());
            for (size_t t = 0; t < toolEmb.size(); t++) {
                if (!keep.count(toolApp[t])) continue;            // 第一级挑错了应用 → 正确的工具根本没机会（误差会传下去）
                float d = 0; for (size_t j = 0; j < q.size(); j++) d += q[j] * toolEmb[t][j]; ts[t] = d;
            }
            return topK(ts, k);
        }`,
    jsonschema: `
        // 最小的 JSON 值类型（够表达工具参数）+ JSON Schema 子集校验：type / enum / required / items
        struct Json {
            enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
            bool b = false; double num = 0; std::string str;
            std::vector<Json> arr; std::map<std::string, Json> obj;
        };
        struct Schema {
            std::string type;                                     // "object" | "string" | "number" | "integer" | "boolean" | "array"
            std::vector<std::string> enumValues;
            std::map<std::string, Schema> properties; std::vector<std::string> required;
            std::shared_ptr<Schema> items;
        };

        bool typeMatches(const Schema& s, const Json& v) {
            if (s.type == "object") return v.kind == Json::Object;
            if (s.type == "array") return v.kind == Json::Array;
            if (s.type == "string") return v.kind == Json::String;
            if (s.type == "boolean") return v.kind == Json::Bool;
            if (s.type == "integer") return v.kind == Json::Number && std::floor(v.num) == v.num;
            if (s.type == "number") return v.kind == Json::Number;
            return true;                                          // 没写 type = 任何值都行
        }

        void validate(const Schema& s, const Json& v, const std::string& at, std::vector<std::string>& errs) {
            if (!typeMatches(s, v)) { errs.push_back(at + ": expected " + s.type); return; }
            if (!s.enumValues.empty() && std::find(s.enumValues.begin(), s.enumValues.end(), v.str) == s.enumValues.end()) errs.push_back(at + ": not an allowed value");
            if (v.kind == Json::Object) {
                for (const std::string& r : s.required) if (!v.obj.count(r)) errs.push_back(at + ": missing required property \\"" + r + "\\"");
                for (const auto& kv : v.obj) {
                    auto p = s.properties.find(kv.first);
                    if (p == s.properties.end()) errs.push_back(at + ": unknown property \\"" + kv.first + "\\"");
                    else validate(p->second, kv.second, at + "." + kv.first, errs);
                }
            }
            if (v.kind == Json::Array && s.items) for (size_t i = 0; i < v.arr.size(); i++) validate(*s.items, v.arr[i], at + "[" + std::to_string(i) + "]", errs);
        }`,
    toolsearch: `
        // 按需加载：模型一开始只看到 tool_search（+ 少数常用工具）；它调用 tool_search 时，Harness 用检索器挑出 k 个工具，把完整定义放进上下文
        struct Catalog { std::vector<ToolDef> tools; Encoder enc; std::vector<std::vector<float>> toolEmb; BM25Index bm25; };

        std::vector<int> hybridSearch(const Catalog& c, const std::string& query, int k, float wBM25) {
            std::vector<float> dense, lex = scoreBM25(c.bm25, query);
            dense.resize(c.tools.size());
            std::vector<float> q = embedText(c.enc, query);
            for (size_t t = 0; t < c.tools.size(); t++) { float d = 0; for (size_t j = 0; j < q.size(); j++) d += q[j] * c.toolEmb[t][j]; dense[t] = d; }
            std::vector<float> fused = rrfFuse({ topK(lex, 200), topK(dense, 200) }, { wBM25, 1.f }, (int)c.tools.size());
            return topK(fused, k);
        }

        // tool_search 的返回值：k 个工具的完整定义（之后模型就能直接调用它们）
        std::string handleToolSearch(const Catalog& c, const std::string& query, int k) {
            std::string out = "[";
            for (int i : hybridSearch(c, query, k, 0.5f)) out += (out.size() > 1 ? "," : "") + toAnthropicJson(c.tools[i]);
            return out + "]";
        }`,
    action: `
        // 动作类型：直接回答 / 调一个工具 / 写一段代码 / 操作界面 / 先问清楚 —— 哈希特征上的多分类逻辑回归
        enum class Action { Answer, Tool, Code, Gui, Ask };

        struct ActionClassifier { int NB = 1 << 13, C = 5; std::vector<float> W, b; };   // W: NB × C

        std::vector<float> actionProbs(const ActionClassifier& m, const std::string& text) {
            std::vector<float> z = m.b;
            Features f = hashFeatures(text, m.NB);
            float n2 = 0; for (const auto& kv : f) n2 += kv.second * kv.second;
            float inv = 1.f / std::sqrt(std::max(n2, 1.f));
            for (const auto& kv : f) for (int c = 0; c < m.C; c++) z[c] += m.W[(size_t)kv.first * m.C + c] * kv.second * inv;
            float mx = *std::max_element(z.begin(), z.end()), zs = 0;
            for (float& v : z) { v = std::exp(v - mx); zs += v; }
            for (float& v : z) v /= zs;
            return z;
        }`,
};
if (typeof module !== 'undefined' && module.exports) module.exports = TK_CPP;
