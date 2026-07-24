# Meal Agent

一个基于用户画像的流式饮食推荐 AI Agent，支持多轮对话、价格追踪和每日推荐。

## 功能概览

- **流式推荐对话** — 与 Agent 多轮交流，获取个性化菜谱推荐，回答实时以 SSE token 流输出
- **用户画像记忆** — 自动提取饮食偏好、过敏原、口味约束，跨会话持久化
- **价格追踪** — 查询食材和电子产品当前价格，折线图展示历史走势
- **每日推荐** — 定时生成午/晚餐推荐，支持懒加载（首次访问时触发）
- **营养与食品安全** — 基于向量知识库回答营养搭配和食材禁忌问题
- **API 鉴权 + 限流** — X-API-Key 认证，Redis 原子令牌桶限流

## 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek（via `langchain-openai` 兼容接口） |
| Agent 编排 | LangGraph StateGraph |
| 后端 | FastAPI + sse-starlette（Python 3.13） |
| 向量存储 | ChromaDB + `BAAI/bge-small-zh` 嵌入模型 |
| 关系数据库 | MySQL（用户画像、价格历史）|
| 轻量持久化 | SQLite（对话记忆、Dead-letter 队列）|
| 缓存 | Redis（画像缓存、限流计数器） |
| 前端 | Next.js 16 App Router + React 19 + Tailwind CSS |
| 部署 | Railway（后端容器）+ Cloudflare Workers（前端） |

## 架构

```
Browser
  └─→ Cloudflare Worker /api/backend/*   (Next.js Route Handler — 服务端代理)
            └─→ Railway FastAPI :8000     (X-API-Key 在服务端注入)
                      ├─→ Redis           (限流、画像缓存、话题缓存)
                      ├─→ MySQL           (用户画像、价格历史)
                      └─→ Railway Volume /app/runtime-data
                                ├─ ChromaDB     (向量知识库 + 对话记忆)
                                ├─ SQLite       (users.db, dead_letter.db)
                                └─ HuggingFace  (bge-small-zh 模型缓存)
```

浏览器只与同源的 Cloudflare Worker 通信，**永不**直接访问 Railway 域名。API Key 仅存在于服务端环境变量，不进入浏览器 bundle。

## Agent 图结构

```
memory → supervisor → (intent 路由)
                          ├─→ meal          菜谱推荐（SSE 流式输出）
                          ├─→ price         价格查询
                          ├─→ update        画像更新 → (needs_meal?) → meal 或 END
                          ├─→ nutrition     营养搭配
                          ├─→ food_safety   食品安全
                          └─→ chat          通用对话
```

每个 Agent 节点通过 `ToolExecutor` 执行工具调用，统一超时保护和降级策略。

## 快速开始（本地开发）

### 前置条件

- Python 3.13
- Node.js 20+
- Redis（本地或 Docker）
- MySQL（本地或 Docker）

### 后端

```bash
# 1. 克隆仓库
git clone https://github.com/biang5464/meal-agent.git
cd meal-agent

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填写：
#   DEEPSEEK_API_KEY=sk-...
#   MYSQL_HOST / MYSQL_PASSWORD / MYSQL_DATABASE
#   REDIS_URL=redis://localhost:6379/0

# 4. 启动服务
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 前端

```bash
cd frontend

# 1. 安装依赖
npm install

# 2. 配置本地环境变量（不提交）
cat > .env.local <<'EOF'
MEAL_AGENT_BACKEND_URL=http://localhost:8000
MEAL_AGENT_API_KEY=
EOF

# 3. 启动开发服务器
npm run dev
# → http://localhost:3000
```

## 主要接口

### 后端 API（Railway）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/recommend` | SSE 流式推荐，body: `{user_id, message, chat_history}` |
| `GET` | `/api/price-history/{query_term}` | 价格历史，`?mode=single\|multi&days=30` |
| `GET` | `/api/tracked-terms` | 可追踪的商品关键词列表 |
| `GET` | `/api/daily-recommendation` | 获取今日推荐（懒加载） |
| `POST` | `/api/daily-recommendation/generate` | 手动触发推荐生成 |

> 生产环境所有接口需携带 `X-API-Key: <MEAL_AGENT_API_KEY>` 请求头。

### 前端代理路由

Cloudflare Worker 将 `/api/backend/*` 的请求透明转发到 Railway，并自动注入 `X-API-Key`。前端只需调用 `/api/backend/recommend` 等相对路径。

## 测试

```bash
# 仅运行非网络测试（推荐）
.venv/Scripts/python.exe -m pytest -q -m "not network" --tb=short

# 运行特定模块
.venv/Scripts/python.exe -m pytest tests/test_phase10d6_cloudflare.py -q

# 前端 Cloudflare Worker 运行时烟雾测试（需 wrangler 和已编译的 worker）
cd frontend && npm run test:cloudflare-runtime
```

当前测试覆盖：867 个非网络测试，包含以下模块：

- Agent 行为（菜谱、价格、营养、食品安全、画像更新）
- Context Manager（约束提取、删除语义、跨域过滤）
- 工具层（ToolExecutor、超时保护、Dead-letter 队列）
- 缓存层（Redis 降级、FakeRedis 集成）
- API 鉴权与限流
- Cloudflare Worker 代理（50 项结构检查 + 33 项运行时验证）

## 部署

### 后端（Railway）

1. 在 Railway 创建新项目，关联此仓库（自动识别 `railway.toml` 和 `Dockerfile`）
2. 在 Railway Dashboard 添加 Redis 和 MySQL 插件
3. 挂载 Volume 到 `/app/runtime-data`（**不要**挂到 `/app/data`）
4. 设置环境变量（参见下表）
5. 保持单副本运行（SQLite + ChromaDB + APScheduler 不支持多进程共享）

**Railway 必填环境变量：**

```
APP_ENV=production
DEEPSEEK_API_KEY=sk-...
API_AUTH_ENABLED=true
MEAL_AGENT_API_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">
RATE_LIMIT_ENABLED=true
ALLOWED_ORIGINS=https://<your-cloudflare-worker-domain>
CHROMA_PERSIST_DIR=/app/runtime-data/chroma
SQLITE_DB_PATH=/app/runtime-data/users.db
DEAD_LETTER_DB_PATH=/app/runtime-data/dead_letter.db
HF_HOME=/app/runtime-data/huggingface
NUTRITION_DIR=/app/data/nutrition
FOOD_SAFETY_DIR=/app/data/food_safety
```

> 首次部署会下载 `BAAI/bge-small-zh` 嵌入模型（~130 MB）。`railway.toml` 中 `healthcheckTimeout = 600` 留出 10 分钟等待时间。

### 前端（Cloudflare Workers）

```bash
cd frontend

# 构建
npm run build   # Next.js 编译
# 或完整的 CF 构建 + 预览
npm run preview

# 部署（需要 wrangler login）
npm run deploy
```

**Cloudflare Dashboard 环境变量**（通过 Dashboard 或 `wrangler secret put` 设置，不写入仓库）：

```
APP_ENV=production
MEAL_AGENT_BACKEND_URL=https://<your-railway-domain>
MEAL_AGENT_API_KEY=<same key as Railway>
```

`wrangler.jsonc` 中设有 `"keep_vars": true`，确保 Git 自动部署不会覆盖 Dashboard 中配置的变量。

## 安全设计

| 层 | 机制 |
|---|---|
| 鉴权 | `X-API-Key` 由 `AuthMiddleware`（纯 ASGI）校验，不阻断 SSE 流 |
| 限流 | Redis Lua 原子脚本，`/recommend` 10次/分钟，只读接口 60次/分钟 |
| Key 隔离 | API Key 仅存在 Railway 和 Cloudflare 服务端环境变量，不进入浏览器 bundle |
| IP 隐私 | Client IP 经 SHA-256 哈希后才写入 Redis，不存储原始 IP |
| 错误脱敏 | 上游错误返回固定通用消息，内部细节仅服务端日志记录 |
| Redis 故障 | 生产+成本类接口失败关闭（503），不调用 LLM |

## 项目结构

```
meal-agent/
├── agents/            # LangGraph 节点（meal, price, update, chat, context_manager…）
├── core/              # 基础设施（缓存、数据库、鉴权、限流、嵌入运行时）
├── tools/             # 工具函数（菜谱搜索、价格爬虫、营养知识库、超时配置）
├── models/            # SQLAlchemy ORM 模型
├── data/
│   ├── nutrition/     # 营养文档种子数据（提交到仓库）
│   └── food_safety/   # 食品安全文档种子数据（提交到仓库）
├── tests/             # 867 个测试
├── frontend/          # Next.js 前端
│   ├── app/
│   │   ├── api/backend/[...path]/route.ts   # Cloudflare Worker 代理
│   │   ├── components/                       # PriceChart, DailyRecommendation
│   │   └── page.tsx                          # 聊天主界面
│   ├── scripts/cloudflare-runtime-smoke.mjs  # 运行时烟雾测试
│   └── wrangler.jsonc                         # Cloudflare Worker 配置
├── main.py            # FastAPI 入口
├── Dockerfile         # 多阶段构建（Python 3.13-slim）
├── railway.toml       # Railway 部署配置
└── .env.example       # 环境变量说明模板
```

## 已知限制

- 单副本运行，暂不支持水平扩展（SQLite + ChromaDB 不支持多进程写入）
- 首次启动较慢，需下载嵌入模型（约 1-2 分钟）
- 价格数据依赖外部爬虫接口，部分商品覆盖有限

## License

MIT
