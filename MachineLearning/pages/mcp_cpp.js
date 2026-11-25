// mcp_cpp.js — 第 21 篇的 C++ 代码块（C++17 + 标准库；tools/tk_cpp_check.js mcp_cpp.js:… 检查）。
// 与页面里的 JS（mcp_common.js）对应：JSON-RPC 分派、工具注解、引擎场景与属性、脚本沙箱、批量操作、Computer Use 的心跳循环。
// 正文用 String.raw，反斜杠原样保留（C++ 里的 \n、\" 不会被 JS 吃掉）。
const MC_CPP = {
    // ═══════════════ 第 21 篇 ═══════════════
    mcpserver: String.raw`
        // MCP 服务器的一端：收 JSON-RPC 消息，按 method 分派。传输层可以是 stdio（本地进程）或 HTTP（远程）
        struct JsonRpc { std::string jsonrpc = "2.0"; long id = 0; std::string method; std::string paramsJson; };

        // 工具注解：告诉客户端「这个工具会不会改东西、能不能撤销」，客户端据此决定要不要确认
        struct Annotations {
            bool readOnlyHint = false;
            bool destructiveHint = false;      // 会删东西：默认要用户确认
            bool idempotentHint = false;
            bool openWorldHint = false;        // 会碰外部世界（网络、磁盘）
        };

        struct McpTool {
            std::string name, description, inputSchemaJson;
            Annotations annotations;
            std::function<std::string(const std::string& argsJson, bool& isError)> invoke;   // 返回 structuredContent
        };

        struct McpServer {
            std::string name, version, protocolVersion = "2025-06-18";
            std::vector<McpTool> tools;
            const McpTool* find(const std::string& n) const { for (const McpTool& t : tools) if (t.name == n) return &t; return nullptr; }
        };

        // 最小的 JSON 读取：取一个字符串字段 / 一个对象字段（真实实现用现成的 JSON 库）
        std::string q(const std::string& s) { return "\"" + s + "\""; }
        std::string extractString(const std::string& json, const std::string& key) {
            std::string pat = q(key) + ":";
            size_t at = json.find(pat);
            if (at == std::string::npos) return "";
            at = json.find('"', at + pat.size());
            if (at == std::string::npos) return "";
            size_t end = json.find('"', at + 1);
            return json.substr(at + 1, end - at - 1);
        }
        std::string extractObject(const std::string& json, const std::string& key) {
            std::string pat = q(key) + ":";
            size_t at = json.find(pat);
            if (at == std::string::npos) return "{}";
            size_t open = json.find('{', at);
            if (open == std::string::npos) return "{}";
            int depth = 0;
            for (size_t i = open; i < json.size(); i++) {
                if (json[i] == '{') depth++;
                else if (json[i] == '}' && --depth == 0) return json.substr(open, i - open + 1);
            }
            return "{}";
        }

        // 分派：initialize → tools/list → tools/call。未知方法回 JSON-RPC 错误；工具自身失败回 isError 的结果
        std::string dispatch(McpServer& srv, const JsonRpc& req) {
            std::ostringstream out;
            if (req.method == "initialize") {
                out << R"({"jsonrpc":"2.0","id":)" << req.id << R"(,"result":{"protocolVersion":")" << srv.protocolVersion
                    << R"(","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":")" << srv.name << R"(","version":")" << srv.version << R"("}}})";
            } else if (req.method == "tools/list") {
                out << R"({"jsonrpc":"2.0","id":)" << req.id << R"(,"result":{"tools":[)";
                for (size_t i = 0; i < srv.tools.size(); i++) {
                    const McpTool& t = srv.tools[i];
                    if (i) out << ",";
                    out << R"({"name":")" << t.name << R"(","description":")" << t.description << R"(","inputSchema":)" << t.inputSchemaJson
                        << R"(,"annotations":{"readOnlyHint":)" << (t.annotations.readOnlyHint ? "true" : "false")
                        << R"(,"destructiveHint":)" << (t.annotations.destructiveHint ? "true" : "false")
                        << R"(,"idempotentHint":)" << (t.annotations.idempotentHint ? "true" : "false")
                        << R"(,"openWorldHint":)" << (t.annotations.openWorldHint ? "true" : "false") << "}}";
                }
                out << "]}}";
            } else if (req.method == "tools/call") {
                std::string name = extractString(req.paramsJson, "name"), args = extractObject(req.paramsJson, "arguments");
                const McpTool* t = srv.find(name);
                if (!t) return R"({"jsonrpc":"2.0","id":)" + std::to_string(req.id) + R"(,"error":{"code":-32602,"message":"unknown tool"}})";
                bool isError = false;
                std::string sc = t->invoke(args, isError);                            // 业务错误也走结果，不抛异常
                out << R"({"jsonrpc":"2.0","id":)" << req.id << R"(,"result":{"content":[{"type":"text","text":)" << q(sc)
                    << R"(}],"structuredContent":)" << sc << R"(,"isError":)" << (isError ? "true" : "false") << "}}";
            } else {
                out << R"({"jsonrpc":"2.0","id":)" << req.id << R"(,"error":{"code":-32601,"message":"method not found"}})";
            }
            return out.str();
        }

        // 权限：注解只是「提示」，真正的闸门在客户端——destructive 的工具默认每次都要问用户
        enum class Ask { Never, Destructive, Always };
        Ask needConfirm(const McpTool& t, bool trustedServer) {
            if (t.annotations.readOnlyHint) return Ask::Never;
            if (t.annotations.destructiveHint) return Ask::Always;
            return trustedServer ? Ask::Never : Ask::Destructive;                    // 不信任的服务器：改东西就问
        }`,
    engine: String.raw`
        // 引擎侧：一个最小场景（actor + 属性）和按名字/类型查改的接口。真实实现里这些就是 UE 的 AActor / UActorComponent
        struct Actor {
            std::string id, type, cls, parent;
            std::map<std::string, std::string> props;                                 // 属性先按字符串存；真实里是 FProperty 反射
        };
        struct Engine {
            std::vector<Actor> actors;
            Actor* find(const std::string& id) { for (Actor& a : actors) if (a.id == id) return &a; return nullptr; }
            std::vector<Actor*> byType(const std::string& t) { std::vector<Actor*> out; for (Actor& a : actors) if (t.empty() || a.type == t) out.push_back(&a); return out; }
        };

        // set_property 的实现：对象、属性两层检查，失败时把「哪里不对」写成一句话交回模型
        std::string setProperty(Engine& e, const std::string& id, const std::string& prop, const std::string& value) {
            Actor* a = e.find(id);
            if (!a) return R"({"error":"no actor named )" + id + R"("})";
            auto it = a->props.find(prop);
            if (it == a->props.end()) return R"j({"error":"actor )j" + id + " (" + a->cls + ") has no property " + q(prop) + "}";
            std::string before = it->second;
            it->second = value;
            return R"({"id":)" + q(id) + R"(,"property":)" + q(prop) + R"(,"before":)" + q(before) + R"(,"after":)" + q(value) + R"(,"saved":false})";
        }

        // 关卡里的 actor 常常是几千个：列出全部既读不完也传不动，接口必须分页
        std::string listActors(Engine& e, const std::string& type, int offset, int limit) {
            std::vector<Actor*> all = e.byType(type);
            int from = std::max(0, offset - 1), to = std::min<int>((int)all.size(), from + std::min(limit, 50));
            std::string out = R"({"total":)" + std::to_string(all.size()) + R"(,"offset":)" + std::to_string(from + 1) + R"(,"actors":[)";
            for (int i = from; i < to; i++) out += (i > from ? "," : "") + q(all[i]->id);
            return out + "]}";
        }`,
    sandbox: String.raw`
        // 代码即动作：服务器把同一套 API 暴露成脚本函数，一次调用里跑完整个循环。
        // 脚本是「数据」也是「代码」——所以要在沙箱里跑：限语句数、只给这几个函数，不给文件系统和网络
        struct ScriptApi {
            Engine& engine;
            int calls = 0, maxCalls = 500;                                            // 语句数上限：防止死循环
            std::vector<std::string> log;
        };

        struct ScriptError { std::string message; };

        // dim("light", 1.6)：把所有灯的强度乘以 1.6——9 次工具调用变成 1 次
        void apiDim(ScriptApi& s, const std::string& type, double factor) {
            if (++s.calls > s.maxCalls) throw ScriptError{ "script exceeded 500 statements" };
            int n = 0;
            for (Actor& a : s.engine.actors) {
                auto it = a.props.find("intensity");
                if (a.type != type || it == a.props.end()) continue;
                it->second = std::to_string((long)std::lround(std::stod(it->second) * factor));
                n++;
            }
            s.log.push_back("dim(" + type + ", " + std::to_string(factor) + ") -> " + std::to_string(n) + " actors");
        }

        void apiSet(ScriptApi& s, const std::string& id, const std::string& prop, const std::string& value) {
            if (++s.calls > s.maxCalls) throw ScriptError{ "script exceeded 500 statements" };
            std::string r = setProperty(s.engine, id, prop, value);
            if (r.find(R"("error")") != std::string::npos) throw ScriptError{ r };     // 脚本里出错就停下，错误原样回给模型
            s.log.push_back("set(" + id + ", " + prop + ", " + value + ")");
        }

        // 聚合在服务器里做：模型只要一个数字，不必把 1651 个名字都读进上下文
        std::string apiCount(ScriptApi& s, const std::string& type) {
            if (++s.calls > s.maxCalls) throw ScriptError{ "script exceeded 500 statements" };
            return std::to_string(s.engine.byType(type).size());
        }`,
    batch: String.raw`
        // 两种做法：逐个调用（每次一个 JSON-RPC 来回）vs 一段脚本（一次调用跑完）
        // 代价 = 往返次数 × (请求 token + 响应 token)；往返还带网络 / 进程切换的延迟
        struct Cost { int roundTrips = 0, tokens = 0; };

        Cost oneByOne(int n, int tokPerCall) {
            Cost c;
            for (int i = 0; i < n; i++) { c.roundTrips++; c.tokens += tokPerCall; }   // 每个对象一次 tools/call
            return c;
        }

        Cost oneScript(int scriptTokens, int resultTokens) {
            Cost c; c.roundTrips = 1; c.tokens = scriptTokens + resultTokens; return c;
        }`,
    cu: String.raw`
        // Computer Use：没有 API 的软件只能「看屏幕、点鼠标」。每一步 = 一张截图 + 一次点击
        struct Screenshot { int width = 1456, height = 819; };
        int visionTokens(const Screenshot& s) { return (s.width * s.height) / 750; }   // ≈ 1590：Anthropic 的视觉 token 估算

        struct Action { std::string thought; int x = 0, y = 0; std::string kind; };     // kind: click / type / key / scroll

        struct ComputerUseCost { int steps = 0, screenshots = 0, tokens = 0; };

        // 每个动作要「看 → 想 → 点 → 再看一眼确认」：至少两张截图
        ComputerUseCost planCost(const std::vector<Action>& actions, const Screenshot& s) {
            ComputerUseCost c; c.steps = (int)actions.size();
            c.screenshots = 2 * c.steps;
            c.tokens = c.screenshots * visionTokens(s) + 40 * c.steps;                  // 40：一句思考 + 坐标
            return c;
        }`,
};
if (typeof module !== 'undefined' && module.exports) module.exports = MC_CPP;
