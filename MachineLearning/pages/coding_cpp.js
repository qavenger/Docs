// coding_cpp.js — 第 20 篇（Coding Agent）的 C++ 代码块（C++17 + 标准库；按页面顺序拼起来可以直接编译，tools/tk_cpp_check.js coding_cpp.js … 检查）。
// 与页面里的 JS（coding_common.js）逐条对应：工具注册与分发、仓库 token 统计、Read / Grep / Edit、行级 diff、命令分级与放行策略、仓库地图、Agent 主循环、子 Agent、评测、检查点。
const CA_CPP = {
    agenttools: `
        // 编程 Agent 的工具：每个 = 名字 + 给模型看的描述 + 参数的 JSON Schema + 权限级别 + 实现
        enum class Perm { ReadOnly, Edit, Exec };
        struct ToolResult { std::string content; bool isError = false; };              // 出错也是一种结果：交回模型，让它自己改
        struct Args {                                                                    // 简化：解析好的参数按类型分开存
            std::map<std::string, std::string> s; std::map<std::string, long> i; std::map<std::string, bool> b;
            std::string str(const std::string& k, const std::string& def = "") const { auto it = s.find(k); return it == s.end() ? def : it->second; }
            long num(const std::string& k, long def) const { auto it = i.find(k); return it == i.end() ? def : it->second; }
            bool flag(const std::string& k) const { auto it = b.find(k); return it != b.end() && it->second; }
        };
        struct Tool {
            std::string name, description, schemaJson;
            Perm perm = Perm::ReadOnly;
            std::function<ToolResult(const Args&)> run;
        };
        struct ToolBox { std::vector<Tool> tools; };                                      // 编程 Agent 通常只有 8–15 个

        // 名字 → 工具（返回指针：按名字在向量里查，不用一个工具就复制一份）
        Tool* findTool(ToolBox& box, const std::string& name) {
            for (Tool& t : box.tools) if (t.name == name) return &t;
            return nullptr;
        }

        ToolResult dispatch(ToolBox& box, const std::string& name, const Args& args) {
            Tool* t = findTool(box, name);
            if (!t) return { "<tool_use_error>No such tool: " + name + "</tool_use_error>", true };
            return t->run(args);
        }`,
    repotokens: `
        // 工作区：根目录 + 本次会话读过哪些文件（Edit 之前必须读过）+ 检查点（每个文件第一次被改之前的内容）
        struct Workspace {
            std::filesystem::path root;
            std::set<std::string> readFiles;
            std::map<std::string, std::string> checkpoints;
        };

        std::string readFileText(const std::filesystem::path& p) { std::ifstream in; in.open(p, std::ios::binary); std::ostringstream ss; ss << in.rdbuf(); return ss.str(); }

        std::vector<std::string> splitLines(const std::string& text) {
            std::vector<std::string> lines; std::string cur;
            for (char ch : text) { if (ch == '\\n') { lines.push_back(cur); cur.clear(); } else if (ch != '\\r') cur += ch; }
            if (!cur.empty()) lines.push_back(cur);
            return lines;
        }

        // 整个仓库有多少 token：遍历文件，逐个用 BPE 计数（跳过 .git 和二进制文件）
        using FileTokens = std::vector<std::pair<std::string, int>>;                  // (相对路径, token 数)
        FileTokens repoTokens(const std::filesystem::path& root, const BPECounter& B) {
            FileTokens out;
            for (const auto& e : std::filesystem::recursive_directory_iterator(root)) {
                if (!e.is_regular_file()) continue;
                std::string rel = std::filesystem::relative(e.path(), root).generic_string();
                if (rel.rfind(".git/", 0) == 0) continue;
                std::string text = readFileText(e.path());
                if (text.find('\\0') != std::string::npos) continue;        // 二进制（贴图、.uasset、二进制 .prefab）：模型读不了，要靠引擎的 API
                out.push_back({ rel, countTokens(B, text) });
            }
            return out;
        }`,
    readtool: `
        // Read：cat -n 格式（6 位行号 + 制表符），默认最多 2000 行，超长的行截断；读过的文件记下来
        ToolResult toolRead(Workspace& ws, const std::string& path, int offset = 1, int limit = 2000) {
            std::filesystem::path p = ws.root / path;
            if (!std::filesystem::exists(p)) return { "File does not exist: " + path, true };
            std::vector<std::string> lines = splitLines(readFileText(p));
            ws.readFiles.insert(path);
            int first = std::max(1, offset), last = std::min<int>((int)lines.size(), first - 1 + limit);
            std::string out; char num[16];
            for (int i = first; i <= last; i++) {
                std::snprintf(num, sizeof num, "%6d\\t", i);
                const std::string& t = lines[i - 1];
                out += num + (t.size() > 2000 ? t.substr(0, 2000) + "..." : t) + "\\n";
            }
            if (last < (int)lines.size()) out += "... (" + std::to_string(lines.size() - last) + " more lines; use offset / limit)\\n";
            return { out };
        }`,
    greptool: `
        // Grep：逐个文件、逐行匹配正则，输出 path:line:text（真实实现直接调用 ripgrep：跳过 .gitignore 和二进制文件、多线程、快几个数量级）
        ToolResult toolGrep(const Workspace& ws, const std::string& pattern, const std::string& sub, size_t headLimit = 250) {
            std::regex re;
            try { re = std::regex(pattern); } catch (const std::regex_error& e) { return { std::string("Invalid regex: ") + e.what(), true }; }
            std::vector<std::filesystem::path> files;
            for (const auto& e : std::filesystem::recursive_directory_iterator(ws.root / sub)) if (e.is_regular_file()) files.push_back(e.path());
            std::sort(files.begin(), files.end());
            std::vector<std::string> hits;
            for (const auto& f : files) {
                std::string rel = std::filesystem::relative(f, ws.root).generic_string();
                std::vector<std::string> lines = splitLines(readFileText(f));
                for (size_t i = 0; i < lines.size(); i++) if (std::regex_search(lines[i], re)) hits.push_back(rel + ":" + std::to_string(i + 1) + ":" + lines[i]);
            }
            if (hits.empty()) return { "No matches found" };
            std::string out;
            for (size_t i = 0; i < hits.size() && i < headLimit; i++) out += hits[i] + "\\n";
            if (hits.size() > headLimit) out += "... (" + std::to_string(hits.size() - headLimit) + " more)\\n";
            return { out };
        }`,
    edittool: `
        size_t countOccurrences(const std::string& text, const std::string& needle) {
            size_t n = 0;
            for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) n++;
            return n;
        }

        // Edit：精确的字符串替换。三条规则：读过才能改；old_string 必须恰好出现一次（除非 replace_all）；改之前存检查点
        ToolResult toolEdit(Workspace& ws, const std::string& path, const std::string& oldS, const std::string& newS, bool replaceAll = false) {
            if (!ws.readFiles.count(path)) return { "File has not been read yet. Read it first before writing to it.", true };
            if (oldS == newS) return { "No changes to make: old_string and new_string are exactly the same.", true };
            std::filesystem::path p = ws.root / path;
            std::string text = readFileText(p);
            size_t n = oldS.empty() ? 0 : countOccurrences(text, oldS);
            if (n == 0) return { "String to replace not found in file.", true };
            if (n > 1 && !replaceAll) return { "Found " + std::to_string(n) + " matches of the string to replace, but replace_all is false. Add more context to make it unique.", true };
            ws.checkpoints.emplace(path, text);                                       // emplace：已经有了就不覆盖 = 保留最早的版本
            std::string out; size_t from = 0;
            for (size_t at = text.find(oldS); at != std::string::npos; at = replaceAll ? text.find(oldS, from) : std::string::npos) {
                out += text.substr(from, at - from) + newS;
                from = at + oldS.size();
            }
            out += text.substr(from);
            std::ofstream(p, std::ios::binary) << out;
            return { "The file " + path + " has been updated." };
        }`,
    diff: `
        // 行级 diff：最长公共子序列（LCS）动态规划，再回溯出「保留 / 删除 / 新增」—— git diff 的统一格式就是在它外面加 @@ 块头和 3 行上下文
        struct DiffOp { char kind; std::string line; };                               // ' ' 保留  '-' 删除  '+' 新增
        std::vector<DiffOp> lineDiff(const std::vector<std::string>& a, const std::vector<std::string>& b) {
            size_t n = a.size(), m = b.size();
            std::vector<std::vector<int>> L;
            L.assign(n + 1, std::vector<int>(m + 1, 0));                                // L[i][j] = a[i..] 与 b[j..] 的 LCS 长度
            for (size_t i = n; i-- > 0;) for (size_t j = m; j-- > 0;) L[i][j] = a[i] == b[j] ? L[i + 1][j + 1] + 1 : std::max(L[i + 1][j], L[i][j + 1]);
            std::vector<DiffOp> ops; size_t i = 0, j = 0;
            while (i < n && j < m) {
                if (a[i] == b[j]) { ops.push_back({ ' ', a[i] }); i++; j++; }
                else if (L[i + 1][j] >= L[i][j + 1]) ops.push_back({ '-', a[i++] });
                else ops.push_back({ '+', b[j++] });
            }
            while (i < n) ops.push_back({ '-', a[i++] });
            while (j < m) ops.push_back({ '+', b[j++] });
            return ops;
        }`,
    bashtool: `
        // 命令分级：只读白名单 → 自动放行；会执行代码 / 改状态 → 要确认；拒绝名单 → 不跑
        enum class CmdClass { Read, Exec, Deny };
        std::vector<std::string> shellWords(const std::string& cmd);                   // 按 shell 规则切词：引号、管道 |（实现见页面 JS 的 tokenize）

        CmdClass classifyCommand(const std::string& cmd) {
            static const std::set<std::string> readOnly = { "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "pwd", "echo" };
            static const std::set<std::string> deny = { "rm", "curl", "wget", "sudo", "chmod", "mv", "cp", "dd", "ssh" };
            static const std::set<std::string> gitRead = { "status", "diff", "log", "show", "blame" };
            std::vector<std::string> w = shellWords(cmd);
            CmdClass worst = CmdClass::Read;
            for (size_t i = 0; i < w.size(); i++) {
                if (w[i].rfind(">", 0) == 0) worst = CmdClass::Exec;                   // > 重定向 = 写文件（2>&1 不以 > 开头）
                if (i > 0 && w[i - 1] != "|") continue;                                  // 只看每段管道的第一个词
                if (deny.count(w[i])) return CmdClass::Deny;
                if (w[i] == "git") { if (i + 1 >= w.size() || !gitRead.count(w[i + 1])) worst = CmdClass::Exec; continue; }
                if (!readOnly.count(w[i])) worst = CmdClass::Exec;                        // 不认识的命令一律当成会改东西
            }
            return worst;
        }

        // 放行策略：工具的权限级别 × 模式 × 用户的放行规则（前缀，例如 "npm test"）
        enum class Mode { Default, AcceptEdits, Plan, Bypass };
        enum class Decision { Allow, Ask, Deny };
        Decision decide(Perm toolPerm, const std::string& command, Mode mode, const std::vector<std::string>& allowPrefixes) {
            if (toolPerm == Perm::ReadOnly) return Decision::Allow;
            if (toolPerm == Perm::Edit) return mode == Mode::Plan ? Decision::Deny : mode == Mode::Default ? Decision::Ask : Decision::Allow;
            CmdClass c = classifyCommand(command);
            if (c == CmdClass::Deny) return Decision::Deny;                              // 拒绝名单在任何模式下都生效
            if (c == CmdClass::Read) return Decision::Allow;
            if (mode == Mode::Plan) return Decision::Deny;
            if (mode == Mode::Bypass) return Decision::Allow;                            // 只该在一次性的容器 / 虚拟机里用
            for (const std::string& p : allowPrefixes) if (command == p || command.rfind(p + " ", 0) == 0) return Decision::Allow;
            return Decision::Ask;
        }

        // 执行：子进程 + 超时 + 截断（超过 30000 字符只留开头，并告诉模型被截断了）
        struct ProcResult { std::string output; int exitCode = 0; bool timedOut = false; };
        ProcResult runProcess(const std::string& cmd, const std::filesystem::path& cwd, int timeoutMs);   // posix_spawn / CreateProcess + 管道 + 计时器（平台相关，实现略）

        ToolResult toolBash(const Workspace& ws, const std::string& cmd, int timeoutMs = 120000) {
            ProcResult r = runProcess(cmd, ws.root, timeoutMs);
            std::string out = r.output;
            if (out.size() > 30000) out = out.substr(0, 30000) + "\\n... [output truncated: " + std::to_string(r.output.size()) + " chars]";
            if (r.timedOut) out += "\\nCommand timed out after " + std::to_string(timeoutMs / 1000) + " s";
            if (r.exitCode) out += "\\nExit code " + std::to_string(r.exitCode);
            return { out, r.exitCode != 0 };
        }`,
    repomap: `
        // 仓库地图（Aider 的做法）：定义 / 引用 → 文件图 → 个性化 PageRank → 在 token 预算内放尽量多的高分定义
        struct Tag { std::string file, name, text; int line = 0; bool isDef = false; };
        std::vector<Tag> extractTags(const std::string& file, const std::string& text);   // 真实实现用 tree-sitter 的 tags 查询（每种语言一份）；页面 JS 用正则近似

        struct RepoGraph { std::vector<std::string> files; std::vector<std::map<int, double>> out; };   // out[u][v]：u 引用了 v 里定义的符号

        RepoGraph buildGraph(const std::vector<std::string>& files, const std::vector<Tag>& tags, const std::set<std::string>& mentioned) {
            RepoGraph g; g.files = files; g.out.resize(files.size());
            std::map<std::string, int> idx; for (size_t i = 0; i < files.size(); i++) idx[files[i]] = (int)i;
            std::map<std::string, std::set<int>> definers;
            std::map<std::tuple<int, int, std::string>, int> refs;
            for (const Tag& t : tags) if (t.isDef) definers[t.name].insert(idx[t.file]);
            for (const Tag& t : tags) {
                if (t.isDef) continue;
                auto d = definers.find(t.name); if (d == definers.end()) continue;
                for (int v : d->second) if (v != idx[t.file]) refs[{ idx[t.file], v, t.name }]++;
            }
            for (const auto& kv : refs) {
                int u = std::get<0>(kv.first), v = std::get<1>(kv.first); const std::string& name = std::get<2>(kv.first);
                bool snake = name.find('_') != std::string::npos, upper = false, lower = false;
                for (unsigned char ch : name) { upper = upper || (ch >= 'A' && ch <= 'Z'); lower = lower || (ch >= 'a' && ch <= 'z'); }
                double mul = 1;                                                           // 规则取自 Aider 的 repomap.py
                if ((snake || (upper && lower)) && name.size() >= 8) mul *= 10;           // 长而具体的名字（computeDamage）比 get / run 有信息量
                if (mentioned.count(name)) mul *= 10;                                     // 需求里提到的符号
                if (name[0] == '_') mul *= 0.1;                                           // 私有符号
                if (definers[name].size() > 5) mul *= 0.1;                                // 到处都有定义的名字（toString、init）
                g.out[u][v] += mul * std::sqrt((double)kv.second);                        // 引用次数开方：防止一个文件刷屏
            }
            return g;
        }

        // r = (1 − d)·p + d·Σ_u r_u·w(u→v) / Σ_v' w(u→v')      p = 个性化向量（任务提到的文件 / 词）  d = 0.85
        std::vector<double> pageRank(const RepoGraph& g, const std::vector<double>& p, int iters = 40, double d = 0.85) {
            size_t n = g.files.size();
            std::vector<double> r = p, nr;
            for (int it = 0; it < iters; it++) {
                nr.assign(n, 0.0); double dangling = 0;
                for (size_t v = 0; v < n; v++) nr[v] = (1 - d) * p[v];
                for (size_t u = 0; u < n; u++) {
                    double total = 0; for (const auto& e : g.out[u]) total += e.second;
                    if (total == 0) { dangling += r[u]; continue; }                     // 不引用别人的文件：它的分数按 p 散回去
                    for (const auto& e : g.out[u]) nr[e.first] += d * r[u] * e.second / total;
                }
                for (size_t v = 0; v < n; v++) nr[v] += d * dangling * p[v];
                r.swap(nr);
            }
            return r;
        }

        struct RankedDef { std::string file, text; int line = 0; double rank = 0; };

        // 取排名前 m 的定义，按文件分组、组内按行号排；不相邻的行之间写 ⋮（页面 JS 还会补上方法所在的 class 行）
        std::string renderMap(const std::vector<RankedDef>& ranked, int m) {
            std::map<std::string, std::vector<const RankedDef*>> byFile;
            for (int i = 0; i < m && i < (int)ranked.size(); i++) byFile[ranked[i].file].push_back(&ranked[i]);
            std::string out;
            for (auto& kv : byFile) {
                std::sort(kv.second.begin(), kv.second.end(), [](const RankedDef* a, const RankedDef* b) { return a->line < b->line; });
                out += kv.first + ":\\n"; int last = -2;
                for (const RankedDef* d : kv.second) { if (d->line != last + 1) out += "⋮\\n"; out += "│" + d->text + "\\n"; last = d->line; }
                out += "⋮\\n";
            }
            return out;
        }

        // 二分查找：渲染结果的 token 数 ≤ 预算时，最多能放几个定义
        std::string fitToBudget(const std::vector<RankedDef>& ranked, int budget, const BPECounter& B) {
            int lo = 1, hi = (int)ranked.size(); std::string best = renderMap(ranked, 1);
            while (lo <= hi) {
                int mid = (lo + hi) / 2; std::string s = renderMap(ranked, mid);
                if (countTokens(B, s) <= budget) { best = s; lo = mid + 1; } else hi = mid - 1;
            }
            return best;
        }`,
    agentloop: `
        // 编程 Agent 的主循环（第 17 篇的循环 + 本篇的工具和权限）：模型决定做什么，Harness 检查权限、执行、把结果追加进上下文
        struct Message { std::string role, text; };
        struct ToolUse { std::string id, name; Args args; };
        struct ModelReply { std::string text; std::vector<ToolUse> toolUses; };
        ModelReply callModel(const std::string& system, const std::vector<Message>& context, const ToolBox& tools);   // 调模型 API（工具定义随请求发送；实现略）
        bool askUser(const std::string& toolName, const Args& args);                                                 // 在终端 / IDE 里弹出确认（实现略）

        const std::string SYSTEM_PROMPT = "You are a coding agent working in a git repository. Use Grep/Glob to find code before reading it, "
                                          "read only what you need, make minimal edits with Edit, and run the tests after every change.";

        std::string runCodingAgent(ToolBox& box, const std::string& task, Mode mode, const std::vector<std::string>& allow, int maxTurns = 100) {
            std::vector<Message> ctx = { { "user", task } };
            for (int turn = 0; turn < maxTurns; turn++) {
                ModelReply r = callModel(SYSTEM_PROMPT, ctx, box);
                ctx.push_back({ "assistant", r.text });
                if (r.toolUses.empty()) return r.text;                                    // 不再调用工具 = 做完了（或者要问用户）
                for (const ToolUse& u : r.toolUses) {
                    Tool* t = findTool(box, u.name);
                    Decision d = t ? decide(t->perm, u.args.str("command"), mode, allow) : Decision::Deny;
                    bool ok = d == Decision::Allow || (d == Decision::Ask && askUser(u.name, u.args));
                    ToolResult res = ok ? dispatch(box, u.name, u.args) : ToolResult{ "Permission denied: the user did not allow this call.", true };
                    ctx.push_back({ "tool", res.content });                                // 被拒绝也告诉模型：它会换一种做法
                }
            }
            return "stopped: turn limit reached";
        }`,
    subagent: `
        // 子 Agent：同一个循环，但有自己的上下文和更少的工具；主 Agent 只收到它最后的报告
        std::string runSubagent(ToolBox& readOnlyTools, const std::string& prompt) {
            return runCodingAgent(readOnlyTools, prompt, Mode::Plan, {}, 30);         // 调查类任务：只读、限轮数、只回报结论
        }

        // Task 工具：把子 Agent 包成一个普通工具，主 Agent 像调 Grep 一样调它
        Tool makeTaskTool(ToolBox& readOnlyTools) {
            Tool t;
            t.name = "Task";
            t.description = "Launch a sub-agent with its own context window for a self-contained search or analysis task; only its final report comes back.";
            t.schemaJson = R"({"type":"object","properties":{"description":{"type":"string"},"prompt":{"type":"string"}},"required":["description","prompt"]})";
            t.perm = Perm::ReadOnly;
            t.run = [&readOnlyTools](const Args& a) { return ToolResult{ runSubagent(readOnlyTools, a.str("prompt")), false }; };
            return t;
        }`,
    verify: `
        // SWE-bench 式评测：补丁打上去以后跑全部测试（包括 Agent 看不到的隐藏测试）
        //   FAIL_TO_PASS：修之前失败、修之后必须通过的用例（证明修了这个 bug）
        //   PASS_TO_PASS：修之前就通过的用例，一个都不能坏（证明没改坏别的）
        struct TestReport { std::map<std::string, bool> passed; };
        TestReport runTestSuite(const Workspace& ws, const std::vector<std::string>& testFiles);   // 跑测试并解析每条用例的结果（实现略）

        struct Verdict { int f2pOk = 0, f2pTotal = 0, p2pOk = 0, p2pTotal = 0; bool resolved = false; };

        bool passed(const TestReport& r, const std::string& test) { auto it = r.passed.find(test); return it != r.passed.end() && it->second; }   // 没跑到（例如模块加载失败）也算失败

        Verdict evaluatePatch(const Workspace& ws, const std::vector<std::string>& failToPass, const std::vector<std::string>& passToPass, const std::vector<std::string>& testFiles) {
            TestReport r = runTestSuite(ws, testFiles);
            Verdict v;
            for (const std::string& t : failToPass) { v.f2pTotal++; if (passed(r, t)) v.f2pOk++; }
            for (const std::string& t : passToPass) { v.p2pTotal++; if (passed(r, t)) v.p2pOk++; }
            v.resolved = v.f2pOk == v.f2pTotal && v.p2pOk == v.p2pTotal;
            return v;
        }`,
    checkpoint: `
        // 检查点与回滚：toolEdit 第一次改某个文件前把原样存进 ws.checkpoints；用户说「撤回」时整体写回去
        void rewind(Workspace& ws) {
            for (const auto& kv : ws.checkpoints) std::ofstream(ws.root / kv.first, std::ios::binary) << kv.second;
            ws.checkpoints.clear();
        }

        // 项目记忆：仓库根目录的 CLAUDE.md / AGENTS.md 在会话开始时整段放进上下文（构建命令、测试命令、约定、禁区）
        std::string projectMemory(const Workspace& ws) {
            for (const char* name : { "CLAUDE.md", "AGENTS.md" }) {
                std::filesystem::path p = ws.root / name;
                if (std::filesystem::exists(p)) return "<project-instructions>\\n" + readFileText(p) + "</project-instructions>";
            }
            return "";
        }`,
};
if (typeof module !== 'undefined' && module.exports) module.exports = CA_CPP;
