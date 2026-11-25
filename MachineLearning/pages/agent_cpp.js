// agent_cpp.js — 第 16–18 篇的 C++ 代码块（与 agent_common.js 的 JS 参考实现一一对应）。按章节分组：后面的篇把前面篇的代码放进自己的 Common 面板。
// 编译检查：node tools/check_cpp.js 16_tool_calling.lab.js（依次是 minml 库 → Common → 本篇各步骤）
const AGT_CPP = {
    // ═══════════════ 第 16 篇：工具调用 ═══════════════
    chat: `
        // 特殊 token 的 id（词表最前面的 9 个，与 JS 的 AGT.SP 相同）
        const int PAD = 0, EOS = 1, SYS = 2, USER = 3, BOT = 4, CALL = 5, ECALL = 6, RES = 7, ERES = 8;

        // 这段文字是谁写的：系统提示、用户、Harness（<bot> 与工具结果）、模型
        enum Role { ROLE_SYS, ROLE_USER, ROLE_HARNESS, ROLE_BOT };
        struct Segment { Role role; std::string text; };
        struct Tokenized { std::vector<int> ids; std::vector<int> role; };

        // 词表：9 个特殊 token + 训练数据里出现过的全部字符（与 JS 的 AGT.itos 顺序相同；构造略）
        Vocab agentVocab();

        // 系统提示：列出本轮可用的工具（每个工具一个字），没有就写「无」
        std::string systemPrompt(const std::vector<std::string>& tools) {
            std::string s = "可用:";
            if (tools.empty()) return s + "无";
            for (const std::string& t : tools) s += t;
            return s;
        }

        // 一段对话 → token 序列，同时记下每个 token 是谁写的（训练时据此决定哪些位置计入损失）
        Tokenized serialize(const Vocab& v, const std::vector<Segment>& segs) {
            Tokenized t;
            for (const Segment& s : segs)
                for (int id : encode(v, s.text)) { t.ids.push_back(id); t.role.push_back(s.role); }
            return t;
        }

        // 一条专家示范：有计算器就先调用它，再用结果回答；没有就直接回答（模型只能自己算）
        std::vector<Segment> calcExample(const std::vector<std::string>& tools, const std::string& expr, const std::string& result) {
            std::vector<Segment> segs = { { ROLE_SYS, "<sys>" + systemPrompt(tools) }, { ROLE_USER, "<user>" + expr + "等于多少？" }, { ROLE_HARNESS, "<bot>" } };
            if (std::find(tools.begin(), tools.end(), "算") != tools.end()) {
                segs.push_back({ ROLE_BOT, "<call>算(" + expr + ")</call>" });
                segs.push_back({ ROLE_HARNESS, "<res>" + result + "</res>" });     // 由 Harness 执行工具后写入，不是模型生成的
            }
            segs.push_back({ ROLE_BOT, expr + "等于" + result + "。<eos>" });
            return segs;
        }`,
    train: `
        // 训练目标：y[t] = ids[t + 1]；下一个 token 如果不是模型写的（系统提示、用户、Harness 写的工具结果），记 −1 = 不计损失
        Batch maskedTargets(const std::vector<Tokenized>& samples, int T) {
            Batch b;
            b.x = ints((int)samples.size() * T, PAD);
            b.y = ints((int)samples.size() * T, -1);
            for (size_t i = 0; i < samples.size(); i++) {
                const Tokenized& s = samples[i];
                int n = std::min((int)s.ids.size(), T + 1);
                for (int t = 0; t < T && t < n; t++) b.x[i * T + t] = s.ids[t];
                for (int t = 0; t + 1 < n; t++) if (s.role[t + 1] == ROLE_BOT) b.y[i * T + t] = s.ids[t + 1];
            }
            return b;
        }

        // 一步 SFT：和预训练（第 08 篇）是同一个 trainStep，交叉熵自动跳过 −1 的位置
        float sftStep(GPT& m, AdamW& opt, const std::vector<Tokenized>& samples, int T, float lr) {
            Batch b = maskedTargets(samples, T);
            return trainStep(m, opt, b, (int)samples.size(), T, lr);
        }`,
    session: `
        // 带 KV cache 的解码会话：每个 token 只前向一次，把它的 K、V 存进 cache；logits = 下一个 token 的打分
        struct Session { KVCache cache; std::vector<int> ids; std::vector<float> logits; int forwards = 0; };

        Session newSession(const GPT& m) {
            Session s;
            s.cache.k.assign(m.cfg.L, floats(m.cfg.T * m.cfg.d));
            s.cache.v.assign(m.cfg.L, floats(m.cfg.T * m.cfg.d));
            return s;
        }

        void feed(const GPT& m, Session& s, int tok) {
            s.logits = gptStep(m, s.cache, tok, (int)s.ids.size());
            s.ids.push_back(tok);
            s.forwards++;
        }

        // 模型「要不要调用工具」就是 <bot> 之后第一个 token 的分布：P(<call>) 有多大
        float callProbability(const GPT& m, const Vocab& v, const std::string& sys, const std::string& question) {
            Session s = newSession(m);
            for (int id : encode(v, "<sys>" + sys + "<user>" + question + "<bot>")) feed(m, s, id);
            std::vector<float> p = s.logits;
            softmaxInPlace(p, 1.f);
            return p[CALL];
        }`,
    loop: `
        // 工具的返回值，和一组可用工具：名字 + 执行函数（Harness 只通过 run 调用，不关心工具内部怎么实现）
        struct ToolResult { std::string text; bool ok = true; };
        struct ToolBox { std::vector<std::string> names; std::function<ToolResult(const std::string&, const std::string&)> run; };

        // 计算器：只认「整数 运算符 整数」，否则返回「错误」
        std::string calc(const std::string& expr) {
            size_t k = expr.find_first_of("+-");
            std::string op = "+";
            if (k == std::string::npos) { k = expr.find("×"); op = "×"; } else op = expr.substr(k, 1);
            if (k == std::string::npos || k == 0) return "错误";
            std::string a = expr.substr(0, k), b = expr.substr(k + op.size());
            if (b.empty() || a.find_first_not_of("0123456789") != std::string::npos || b.find_first_not_of("0123456789") != std::string::npos) return "错误";
            int x = std::stoi(a), y = std::stoi(b);
            return std::to_string(op == "+" ? x + y : op == "-" ? x - y : x * y);
        }

        // 动态世界：每只动物「现在」在哪里（每局随机，模型背不下来，只能查）
        struct World { std::map<std::string, std::string> where; };

        ToolBox qaTools(World& w, const std::vector<std::string>& names) {
            ToolBox b;
            b.names = names;
            b.run = [&w](const std::string& tool, const std::string& arg) {
                ToolResult r;
                if (tool == "算") r.text = calc(arg);
                else if (tool == "查") r.text = w.where.count(arg) ? w.where[arg] : "没有";
                else { r.text = "没有这个工具"; r.ok = false; }
                return r;
            };
            return b;
        }

        // 解析模型写出的调用「算(37+48)」：工具名是第一个字，参数在括号里
        bool parseCall(const std::string& text, std::string& tool, std::string& arg) {
            size_t l = text.find('('), r = text.rfind(')');
            if (l == std::string::npos || r == std::string::npos || r < l || r + 1 != text.size()) return false;
            tool = text.substr(0, l);
            arg = text.substr(l + 1, r - l - 1);
            return !tool.empty();
        }

        struct EpisodeOptions { bool greedy = true; float temperature = 1.f; bool constrain = false; int toolChoice = 0; int maxCalls = 16; int maxTokens = 400; };   // toolChoice：0 auto，1 none，2 required
        struct Episode { std::vector<int> ids; std::string answer, reason; int calls = 0, forwards = 0; };

        // 对 logits 做掩码：tool_choice（禁止 / 强制 <call>）与约束解码（只允许合乎语法的下一个 token）——第 06、07 步实现
        void maskLogits(std::vector<float>& logits, const Vocab& v, const std::vector<int>& ctx, bool inCall, const std::string& callText, const std::vector<std::string>& tools, int calls, const EpisodeOptions& o);

        // Harness 主循环：生成 → 遇到 </call> 就解析、执行、把 <res>结果</res> 喂回去 → 继续，直到 <eos>
        Episode runEpisode(const GPT& m, const Vocab& v, const std::string& sys, const std::string& user, ToolBox& tools, const EpisodeOptions& o, Rng& r) {
            Episode e;
            Session s = newSession(m);
            std::vector<int> ctx = encode(v, "<sys>" + sys + "<user>" + user + "<bot>");
            size_t fed = 0;
            bool inCall = false;
            std::string callText;
            for (int generated = 0; ; generated++) {
                if (generated >= o.maxTokens) { e.reason = "tokens"; break; }
                if ((int)ctx.size() + 2 > m.cfg.T) { e.reason = "context"; break; }
                while (fed < ctx.size()) feed(m, s, ctx[fed++]);               // 只喂新来的 token：KV cache 里已有的不重算
                std::vector<float> logits = s.logits;
                maskLogits(logits, v, ctx, inCall, callText, tools.names, e.calls, o);
                int id = o.greedy ? argmaxIndex(logits) : sampleIndex(decodeDist(logits, o.temperature, 0, 1.f), r);
                ctx.push_back(id);
                if (id == CALL) { inCall = true; callText.clear(); continue; }
                if (inCall && id != ECALL) { callText += v.itos[id]; continue; }  // 流式累积调用文本
                if (id == ECALL && inCall) {                                   // 一个完整的调用：解析 → 执行 → 写回结果
                    inCall = false; e.calls++;
                    std::string tool, arg, res;
                    if (!parseCall(callText, tool, arg)) res = "格式错误";
                    else if (std::find(tools.names.begin(), tools.names.end(), tool) == tools.names.end()) res = "没有这个工具";
                    else res = tools.run(tool, arg).text;
                    for (int t : encode(v, "<res>" + res + "</res>")) ctx.push_back(t);
                    continue;
                }
                if (id == EOS) { e.reason = "eos"; break; }
                if (id < 9) { e.reason = "bad-token"; break; }                 // 只有 Harness 才能写的特殊 token，或者没有 <call> 就写出的 </call>
                e.answer += v.itos[id];
            }
            e.ids = ctx; e.forwards = s.forwards;
            return e;
        }`,
    grammar: `
        // 工具调用的语法（<call> 之后）：工具名 ( 参数 ) </call>；参数的形状由工具决定
        //   算：1–3 位数字 运算符 1–3 位数字；查：动物名；去：房间名；拿 / 放：物品名；看：空
        // allowedNext 返回在已经写出的 prefix 之后，下一个合法的 token（id 集合）
        std::vector<std::string> argChoices(const std::string& tool);   // 每个工具的参数候选（动物 / 房间 / 物品名，列表略）

        std::set<int> allowedNext(const Vocab& v, const std::string& prefix, const std::vector<std::string>& tools) {
            std::set<int> ids;
            std::vector<std::string> ch = utf8Chars(prefix);
            if (ch.empty()) { for (const std::string& t : tools) ids.insert(tokenId(v, t)); return ids; }
            if (ch.size() == 1) { ids.insert(tokenId(v, "(")); return ids; }
            if (ch.back() == ")") { ids.insert(ECALL); return ids; }
            std::string inner;
            for (size_t i = 2; i < ch.size(); i++) inner += ch[i];
            if (ch[0] == "算") {
                size_t op = inner.find_first_of("+-");
                if (op == std::string::npos) op = inner.find("×");
                std::string digits = op == std::string::npos ? inner : inner.substr(op + (inner[op] == '+' || inner[op] == '-' ? 1 : std::string("×").size()));
                for (char c = '0'; c <= '9'; c++) if (digits.size() < 3) ids.insert(tokenId(v, std::string(1, c)));
                if (op == std::string::npos && !digits.empty()) for (std::string o : { "+", "-", "×" }) ids.insert(tokenId(v, o));
                if (op != std::string::npos && !digits.empty()) ids.insert(tokenId(v, ")"));
                return ids;
            }
            for (const std::string& w : argChoices(ch[0])) {                 // 名字类参数：沿着候选词逐字匹配
                if (w.compare(0, inner.size(), inner) == 0 && w.size() > inner.size()) ids.insert(tokenId(v, utf8Chars(w.substr(inner.size()))[0]));
                if (w == inner) ids.insert(tokenId(v, ")"));
            }
            return ids;
        }

        // tool_choice 与约束解码：都只是在采样之前把一部分 logits 改成 −∞
        void maskLogits(std::vector<float>& logits, const Vocab& v, const std::vector<int>& ctx, bool inCall, const std::string& callText, const std::vector<std::string>& tools, int calls, const EpisodeOptions& o) {
            const float NEG = -1e30f;
            bool atStart = ctx.back() == BOT || ctx.back() == ERES;
            if (o.toolChoice == 1 || calls >= o.maxCalls) logits[CALL] = NEG;                       // none：禁止调用
            if (o.toolChoice == 2 && atStart && calls == 0)                                       // required：第一步只能调用
                for (size_t i = 0; i < logits.size(); i++) if ((int)i != CALL) logits[i] = NEG;
            if (inCall && o.constrain) {
                std::set<int> ok = allowedNext(v, callText, tools);
                if (!ok.empty()) for (size_t i = 0; i < logits.size(); i++) if (!ok.count((int)i)) logits[i] = NEG;
            }
            logits[PAD] = NEG;
        }`,
    // ═══════════════ 第 17 篇：Agent ═══════════════
    house: `
        // 房子：5 个房间、5 件物品；布局固定（客厅连着厨房、卧室、书房，书房连着院子）
        const std::vector<std::string> ROOMS = { "客厅", "厨房", "卧室", "书房", "院子" };
        const std::vector<std::string> ITEMS = { "钥匙", "苹果", "书", "球", "杯子" };
        const std::vector<std::vector<int>> ADJ = { { 1, 2, 3 }, { 0 }, { 0 }, { 0, 4 }, { 3 } };

        struct House { std::vector<int> at; int room = 0; int hand = -1; int steps = 0; };   // at[i] = 第 i 件物品在哪个房间；hand = 手里拿着的物品（−1 = 空手）

        // 每局随机：物品放进随机的房间，智能体从随机的房间出发
        House newHouse(Rng& r) {
            House h;
            for (size_t i = 0; i < ITEMS.size(); i++) h.at.push_back(randint(r, (int)ROOMS.size()));
            h.room = randint(r, (int)ROOMS.size());
            return h;
        }

        int indexOf(const std::vector<std::string>& names, const std::string& s) {
            for (size_t i = 0; i < names.size(); i++) if (names[i] == s) return (int)i;
            return -1;
        }

        // 观察：「厨房:苹果,杯子」——只有当前房间里能看见的东西（手里拿着的不算）
        std::string observe(const House& h, int room) {
            std::string items;
            for (size_t i = 0; i < ITEMS.size(); i++) if (h.at[i] == room && h.hand != (int)i) items += (items.empty() ? "" : ",") + ITEMS[i];
            return ROOMS[room] + ":" + (items.empty() ? std::string("空") : items);
        }

        // 一次工具调用改变房子的状态：看 / 去（走到相邻房间并看到它）/ 拿 / 放
        ToolResult houseStep(House& h, const std::string& tool, const std::string& arg) {
            h.steps++;
            ToolResult r;
            if (tool == "看") { r.text = observe(h, h.room); return r; }
            if (tool == "去") {
                int to = indexOf(ROOMS, arg);
                const std::vector<int>& nb = ADJ[h.room];
                if (to < 0 || std::find(nb.begin(), nb.end(), to) == nb.end()) { r.text = "去不了"; r.ok = false; return r; }
                h.room = to; r.text = observe(h, to); return r;
            }
            int it = indexOf(ITEMS, arg);
            if (tool == "拿") {
                if (it < 0 || h.at[it] != h.room || h.hand == it) { r.text = "这里没有"; r.ok = false; return r; }
                if (h.hand >= 0) { r.text = "手里有了"; r.ok = false; return r; }
                h.hand = it; r.text = "拿到了"; return r;
            }
            if (tool == "放") {
                if (it < 0 || h.hand != it) { r.text = "手里没有"; r.ok = false; return r; }
                h.at[it] = h.room; h.hand = -1; r.text = "放好了"; return r;
            }
            r.text = "没有这个工具"; r.ok = false;
            return r;
        }

        // 房子的工具箱：Harness 的 runEpisode（第 16 篇）通过它执行「看 去 拿 放」
        ToolBox houseTools(House& h) {
            ToolBox b;
            b.names = { "看", "去", "拿", "放" };
            b.run = [&h](const std::string& tool, const std::string& arg) { return houseStep(h, tool, arg); };
            return b;
        }`,
    expert: `
        // 最短路径（BFS）：从 a 走到 b 依次经过的房间（不含 a）
        std::vector<int> shortestPath(int a, int b) {
            std::vector<int> prev = ints((int)ROOMS.size(), -1), seen = ints((int)ROOMS.size(), 0), q;
            q.push_back(a); seen[a] = 1;
            for (size_t k = 0; k < q.size(); k++) for (int v : ADJ[q[k]]) if (!seen[v]) { seen[v] = 1; prev[v] = q[k]; q.push_back(v); }
            std::vector<int> out;
            for (int v = b; v != a; v = prev[v]) out.insert(out.begin(), v);
            return out;
        }

        // 专家示范：先「看」，再深度优先挨个房间走过去（去 = 走过去并看到它），找到目标就停；每一步都记成一段对话
        struct Expert { House h; std::vector<Segment> turns; std::vector<int> seenAt; };   // seenAt[i] = 第 i 件物品在哪看到的（−1 = 还没看到）

        void expertAct(Expert& e, const std::string& tool, const std::string& arg) {
            ToolResult r = houseStep(e.h, tool, arg);
            e.turns.push_back({ ROLE_BOT, "<call>" + tool + "(" + arg + ")</call>" });
            e.turns.push_back({ ROLE_HARNESS, "<res>" + r.text + "</res>" });
            if (r.ok && (tool == "看" || tool == "去"))
                for (size_t i = 0; i < ITEMS.size(); i++) if (e.h.at[i] == e.h.room && e.h.hand != (int)i) e.seenAt[i] = e.h.room;
        }

        // 深度优先探索，直到 item 被看到；返回 true = 找到了
        bool exploreFor(Expert& e, int u, std::vector<int>& visited, int item) {
            if (e.seenAt[item] >= 0) return true;
            for (int v : ADJ[u]) {
                if (visited[v]) continue;
                visited[v] = 1;
                expertAct(e, "去", ROOMS[v]);
                if (exploreFor(e, v, visited, item)) return true;
                expertAct(e, "去", ROOMS[u]);                                    // 回溯
            }
            return e.seenAt[item] >= 0;
        }

        // 「把 item 拿到 target」的专家轨迹
        Expert expertMove(const House& h0, int item, int target) {
            Expert e; e.h = h0; e.seenAt = ints((int)ITEMS.size(), -1);
            expertAct(e, "看", "");
            std::vector<int> visited = ints((int)ROOMS.size(), 0);
            visited[e.h.room] = 1;
            exploreFor(e, e.h.room, visited, item);
            if (e.seenAt[item] == target) { e.turns.push_back({ ROLE_BOT, ITEMS[item] + "已经在" + ROOMS[target] + "了。<eos>" }); return e; }
            for (int v : shortestPath(e.h.room, e.seenAt[item])) expertAct(e, "去", ROOMS[v]);
            expertAct(e, "拿", ITEMS[item]);
            for (int v : shortestPath(e.h.room, target)) expertAct(e, "去", ROOMS[v]);
            expertAct(e, "放", ITEMS[item]);
            e.turns.push_back({ ROLE_BOT, ITEMS[item] + "拿到" + ROOMS[target] + "了。<eos>" });
            return e;
        }`,
    agent: `
        // 让模型自己完成一个房子任务：就是第 16 篇的 runEpisode，工具箱换成房子；成功与否看环境的最终状态
        struct HouseRun { Episode ep; House h; bool success = false; };

        HouseRun runMoveTask(const GPT& m, const Vocab& v, const House& h0, int item, int target, const EpisodeOptions& o, Rng& r) {
            HouseRun out; out.h = h0;
            ToolBox box = houseTools(out.h);
            out.ep = runEpisode(m, v, systemPrompt(box.names), "把" + ITEMS[item] + "拿到" + ROOMS[target], box, o, r);
            out.success = out.ep.reason == "eos" && out.h.at[item] == target && out.h.hand != item;   // 看环境，不看模型说了什么
            return out;
        }`,
    evalAgent: `
        // 失败分类：没说完（预算 / 上下文用完）、写坏格式、动作不合法（去不了、这里没有…）、说完了但没做到
        enum Failure { OK_RUN, OUT_OF_BUDGET, BAD_FORMAT, INVALID_ACTION, WRONG_RESULT };

        Failure classify(const HouseRun& run, const Vocab& v) {
            if (run.success) return OK_RUN;
            if (run.ep.reason == "tokens" || run.ep.reason == "context") return OUT_OF_BUDGET;
            if (run.ep.reason == "bad-token") return BAD_FORMAT;
            std::string text = decode(v, run.ep.ids);
            for (std::string bad : { "去不了", "这里没有", "手里没有", "手里有了", "格式错误", "没有这个工具" })
                if (text.find("<res>" + bad + "</res>") != std::string::npos) return INVALID_ACTION;
            return WRONG_RESULT;
        }

        // n 局随机任务：每种结果各有几局
        std::vector<int> evaluateMove(const GPT& m, const Vocab& v, int n, const EpisodeOptions& o, Rng& r) {
            std::vector<int> count = ints(5, 0);
            for (int i = 0; i < n; i++) {
                House h = newHouse(r);
                int item = randint(r, (int)ITEMS.size()), target = randint(r, (int)ROOMS.size());
                count[classify(runMoveTask(m, v, h, item, target, o, r), v)]++;
            }
            return count;
        }`,
    search: `
        // 多试几次：同一局采样 N 条轨迹，用「环境里的结果」当验证器，只要有一条成功就算成功（pass@N）
        // 真实世界里验证器不总是有：代码可以跑测试，网页任务可以检查页面状态，开放问答就难了
        bool bestOfN(const GPT& m, const Vocab& v, const House& h0, int item, int target, int N, float temperature, Rng& r) {
            EpisodeOptions o; o.greedy = false; o.temperature = temperature;
            for (int k = 0; k < N; k++) if (runMoveTask(m, v, h0, item, target, o, r).success) return true;   // 每次都从同一个初始状态重来
            return false;
        }

        // 「X 在哪？」这种只要回答的任务没有环境状态可查：多数投票（self-consistency）
        std::string majorityAnswer(const std::vector<std::string>& answers) {
            std::map<std::string, int> votes;
            for (const std::string& a : answers) votes[a]++;
            std::string best; int bc = -1;
            for (const auto& kv : votes) if (kv.second > bc) { bc = kv.second; best = kv.first; }
            return best;
        }`,
    flaky: `
        // 会出故障的工具：每次调用以概率 p「超时」，什么也不做（训练数据里专家遇到超时会原样重试一次）
        ToolBox flakyTools(House& h, float p, Rng& r) {
            ToolBox inner = houseTools(h), b = inner;
            b.run = [inner, p, &r](const std::string& tool, const std::string& arg) {
                if ((float)uniform(r) < p) { ToolResult t; t.text = "超时"; t.ok = false; return t; }
                return inner.run(tool, arg);
            };
            return b;
        }`,
    // ═══════════════ 第 18 篇：Harness ═══════════════
    hopts: `
        // ═══ Harness 的设置：上下文预算与策略、自动重试、权限、预算、死循环检测 ═══
        enum ContextPolicy { CTX_NONE, CTX_TRUNCATE, CTX_KEEP_TASK, CTX_COMPACT };
        struct HarnessOptions {
            int window = 192;                       // 上下文 token 预算（≤ 模型的 T）
            ContextPolicy policy = CTX_NONE;        // 放不下时怎么办
            bool cache = true;                      // 复用 KV cache（上下文被改写时只保留公共前缀）
            int retry = 0;                          // 工具「超时」时 Harness 自己重试几次（模型看不到）
            int maxCalls = 16, maxTokens = 400;     // 预算
            bool loopStop = false;                  // 同一个调用连续 3 次 → 停
            std::set<std::string> deniedRooms;      // 权限：不允许进入的房间
        };
        struct HarnessEvent { std::string type, detail; };
        struct ManagedRun { std::vector<int> ids; std::string answer, reason; int calls = 0, forwards = 0; std::vector<HarnessEvent> events; };`,
    ctxpol: `
        // 正在写的这一段从哪里开始：最后一个 </res> 之后（半句回答或半个调用）——改写上下文时原样保留
        int currentTurn(const std::vector<int>& ctx, int prefixLen) {
            int cut = (int)(std::find(ctx.rbegin(), ctx.rend(), ERES).base() - ctx.begin());   // 找不到 = 0
            return std::max(cut, prefixLen);
        }

        // 保留 sys + 任务（前 prefixLen 个 token）和正在写的这一段，从最旧的一次「调用 + 结果」开始删，直到放得下
        std::vector<int> keepTask(const std::vector<int>& ctx, int prefixLen, int budget) {
            const int cut = currentTurn(ctx, prefixLen), cur = (int)ctx.size() - cut;
            std::vector<int> mid;
            mid.assign(ctx.begin() + prefixLen, ctx.begin() + cut);
            while (prefixLen + (int)mid.size() + cur > budget && !mid.empty()) {
                auto it = std::find(mid.begin(), mid.end(), ERES);
                mid.erase(mid.begin(), it == mid.end() ? mid.end() : it + 1);
            }
            std::vector<int> out(ctx.begin(), ctx.begin() + prefixLen);
            out.insert(out.end(), mid.begin(), mid.end());
            out.insert(out.end(), ctx.begin() + cut, ctx.end());
            return out;
        }

        // 按策略把上下文改写到 budget 以内（extra = 接下来至少还要写的 token 数）；memo = Harness 自己记下的状态摘要
        bool fitContext(std::vector<int>& ctx, int prefixLen, int extra, const HarnessOptions& o, const std::vector<int>& memo) {
            int budget = o.window;
            if ((int)ctx.size() + extra <= budget) return true;
            if (o.policy == CTX_TRUNCATE) { ctx.erase(ctx.begin(), ctx.begin() + std::max(0, (int)ctx.size() + extra - budget / 2)); return true; }   // 一次截掉最旧的一半窗口（连任务一起截掉）
            if (o.policy == CTX_KEEP_TASK) { ctx = keepTask(ctx, prefixLen, budget - extra); return (int)ctx.size() + extra <= budget; }
            if (o.policy == CTX_COMPACT) {                                              // 历史整个换成一条记忆（正在写的这一段保留）
                std::vector<int> out(ctx.begin(), ctx.begin() + prefixLen);
                out.insert(out.end(), memo.begin(), memo.end());
                out.insert(out.end(), ctx.begin() + currentTurn(ctx, prefixLen), ctx.end());
                ctx = out;
                return (int)ctx.size() + extra <= budget;
            }
            return false;                                                                // CTX_NONE：只能结束
        }

        using Seen = std::map<int, std::string>;                                         // 房间 → 在那里看到的东西（Harness 自己记的）

        // 记忆（compact 用）：「记:在书房;客厅:书;书房:钥匙;手:苹果」——Harness 根据环境的真实记录写，不靠模型
        std::vector<int> memoTokens(const Vocab& v, const House& h, const Seen& seen) {
            std::string s = "<res>记:在" + ROOMS[h.room];
            for (const auto& kv : seen) s += ";" + ROOMS[kv.first] + ":" + kv.second;
            if (h.hand >= 0) s += ";手:" + ITEMS[h.hand];
            return encode(v, s + "</res>");
        }`,
    kvreuse: `
        // 两个 token 序列的公共前缀长度：上下文被改写之后，这一段的 KV cache 还能直接用
        int commonPrefix(const std::vector<int>& a, const std::vector<int>& b) {
            int k = 0;
            while (k < (int)a.size() && k < (int)b.size() && a[k] == b[k]) k++;
            return k;
        }

        // 把会话退回到前 k 个 token：KV cache 按位置存放，k 之后的条目直接作废，下次 feed 会覆盖
        void rewind(Session& s, int k) { s.ids.resize(k); }`,
    perm: `
        // 权限检查：在执行之前拦下不允许的调用（不管模型多么「想」做）。返回空串 = 放行，否则 = 拒绝的理由
        std::string checkPermission(const HarnessOptions& o, const std::string& tool, const std::string& arg) {
            if (tool == "去" && o.deniedRooms.count(arg)) return "不允许进入" + arg;
            return "";
        }`,
    retry: `
        // Harness 层的重试：工具超时就悄悄再调一次，最多 retry 次；模型只看到最后的结果（上下文里没有「超时」）
        ToolResult callWithRetry(ToolBox& box, const std::string& tool, const std::string& arg, int retry, std::vector<HarnessEvent>& ev) {
            ToolResult r = box.run(tool, arg);
            for (int k = 0; k < retry && r.text == "超时"; k++) { ev.push_back({ "retry", tool + "(" + arg + ")" }); r = box.run(tool, arg); }
            return r;
        }`,
    runManaged: `
        // 带上下文管理、权限、重试、预算、死循环检测的 Harness 主循环（在第 16 篇 runEpisode 的基础上）
        ManagedRun runManaged(const GPT& m, const Vocab& v, const std::string& user, House& h, ToolBox& box, const HarnessOptions& o, Rng& r) {
            ManagedRun run;
            Session s = newSession(m);
            std::vector<int> ctx = encode(v, "<sys>" + systemPrompt(box.names) + "<user>" + user + "<bot>");
            const int prefixLen = (int)ctx.size();
            Seen seen;                                                                   // Harness 自己记下看到过什么（写记忆用）
            EpisodeOptions eo; eo.maxCalls = o.maxCalls;
            bool inCall = false;
            std::string callText, lastCall;
            int repeat = 0;
            for (int generated = 0; ; generated++) {
                if (generated >= o.maxTokens) { run.reason = "tokens"; break; }
                if (!fitContext(ctx, prefixLen, inCall ? 2 : 4, o, memoTokens(v, h, seen))) { run.reason = "context"; break; }
                int keep = o.cache ? commonPrefix(s.ids, ctx) : 0;                         // 上下文改写后：公共前缀的 KV cache 继续用
                rewind(s, keep);
                for (int t = keep; t < (int)ctx.size(); t++) feed(m, s, ctx[t]);
                if (!o.cache) rewind(s, 0);                                                // 不用 cache：下一步整段重算
                std::vector<float> logits = s.logits;
                maskLogits(logits, v, ctx, inCall, callText, box.names, run.calls, eo);
                int id = argmaxIndex(logits);
                ctx.push_back(id);
                if (id == CALL) { inCall = true; callText.clear(); continue; }
                if (inCall && id != ECALL) { callText += v.itos[id]; continue; }
                if (id == ECALL && inCall) {
                    inCall = false; run.calls++;
                    std::string tool, arg, res;
                    if (!parseCall(callText, tool, arg)) res = "格式错误";
                    else if (o.loopStop && (repeat = (callText == lastCall ? repeat + 1 : 1)) >= 3) { run.events.push_back({ "loop", callText }); run.reason = "loop"; break; }
                    else if (!checkPermission(o, tool, arg).empty()) { res = "没有权限"; run.events.push_back({ "denied", callText }); }
                    else {
                        ToolResult tr = callWithRetry(box, tool, arg, o.retry, run.events);
                        res = tr.text;
                        if (tr.ok && (tool == "看" || tool == "去")) seen[h.room] = tr.text.substr(tr.text.find(':') + 1);
                    }
                    lastCall = callText;
                    for (int t : encode(v, "<res>" + res + "</res>")) ctx.push_back(t);
                    continue;
                }
                if (id == EOS) { run.reason = "eos"; break; }
                if (id < 9) { run.reason = "bad-token"; break; }
                run.answer += v.itos[id];
            }
            run.ids = ctx; run.forwards = s.forwards;
            return run;
        }`,
    evalH: `
        // 评测 Harness：两个配置在同一批种子上各跑一遍，逐局配对比较（比两个独立的成功率更灵敏）
        struct Paired { int both = 0, onlyA = 0, onlyB = 0, neither = 0; };

        Paired pairedCompare(const std::vector<bool>& a, const std::vector<bool>& b) {
            Paired p;
            for (size_t i = 0; i < a.size(); i++) {
                if (a[i] && b[i]) p.both++; else if (a[i]) p.onlyA++; else if (b[i]) p.onlyB++; else p.neither++;
            }
            return p;
        }

        // 二项比例的 95% Wilson 置信区间
        std::pair<float, float> wilson(int k, int n) {
            if (n == 0) return { 0.f, 1.f };
            double z = 1.96, p = (double)k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, w = z * std::sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / d;
            return { (float)std::max(0.0, c - w), (float)std::min(1.0, c + w) };
        }

        // 只看「结果不同」的局（onlyA + onlyB）：如果两个配置一样好，这些局应当一半一半；偏离越多，差别越可信（McNemar 检验的思路）
        float mcnemarZ(const Paired& p) {
            int n = p.onlyA + p.onlyB;
            return n == 0 ? 0.f : (float)((p.onlyA - p.onlyB) / std::sqrt((double)n));
        }`,
};
if (typeof module !== 'undefined' && module.exports) module.exports = AGT_CPP;
