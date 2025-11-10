"""
neural_bot_prod_postgres.py
Production-ready Telegram neural bot with persistent memory (Postgres preferred, falls back to SQLite),
neural intent analysis (AIAgentController integrated), admin protections, and a config file.

How to use:
1. Create a config file `config.yaml` (example below) or set environment variables.
2. Install dependencies:
   pip install python-telegram-bot==22.5 openai==1.54.0 google-generativeai==0.8.3
   pip install sympy==1.13.1 requests==2.32.3 nest_asyncio PyYAML psycopg2-binary

3. Run: python neural_bot_prod_postgres.py

Security: DO NOT store API keys in source. Use environment variables or a secrets manager.
"""

import os
import logging
import asyncio
import json
import sqlite3
from datetime import datetime, date
import nest_asyncio

nest_asyncio.apply()

# External libs
import requests
import openai
import google.generativeai as genai
from sympy import sympify, integrate, Symbol
import yaml

# Database driver for Postgres
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG = True
except Exception:
    HAS_PSYCOPG = False

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ParseMode

# ---------------- Logging ----------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
CONFIG_PATH = os.getenv("BOT_CONFIG_PATH", "config.yaml")

DEFAULT_CONFIG = {
    "telegram_token": None,
    "openai_api_key": None,
    "gemini_api_key": None,
    "serpapi_api_key": None,
    "database_url": None,  # e.g. postgres://user:pass@host:5432/dbname
    "use_postgres": True,
    "daily_cap": 10000,
    "admin_ids": [],  # List of Telegram user IDs allowed to run admin commands
    "memory_limit_per_user": 5000,
}


def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
            cfg.update(file_cfg)
            logger.info(f"Loaded configuration from {path}")
    else:
        logger.warning(f"Config file {path} not found; falling back to environment variables.")

    # Override from environment variables (explicit wins)
    cfg["telegram_token"] = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_token")
    cfg["openai_api_key"] = os.getenv("OPENAI_API_KEY") or cfg.get("openai_api_key")
    cfg["gemini_api_key"] = os.getenv("GEMINI_API_KEY") or cfg.get("gemini_api_key")
    cfg["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY") or cfg.get("serpapi_api_key")
    cfg["database_url"] = os.getenv("DATABASE_URL") or cfg.get("database_url")

    # admin ids from env (comma-separated)
    env_admins = os.getenv("BOT_ADMIN_IDS")
    if env_admins:
        try:
            cfg["admin_ids"] = [int(x.strip()) for x in env_admins.split(",") if x.strip()]
        except Exception:
            logger.exception("Failed to parse BOT_ADMIN_IDS env var")

    return cfg

CONFIG = load_config()

TELEGRAM_BOT_TOKEN = CONFIG.get("telegram_token")
OPENAI_API_KEY = CONFIG.get("openai_api_key")
GEMINI_API_KEY = CONFIG.get("gemini_api_key")
SERPAPI_API_KEY = CONFIG.get("serpapi_api_key")
DAILY_CAP = CONFIG.get("daily_cap", 10000)
ADMIN_IDS = set(CONFIG.get("admin_ids", []))
MEMORY_LIMIT_PER_USER = CONFIG.get("memory_limit_per_user", 5000)
USE_POSTGRES = bool(CONFIG.get("use_postgres", True))
DATABASE_URL = CONFIG.get("database_url")

# validate critical keys
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment or config.yaml")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or config.yaml")

openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------- Memory Manager (Postgres preferred, fallback SQLite) ----------------
class MemoryManagerBase:
    def add_message(self, user_id: int, role: str, text: str, ts: str = None):
        raise NotImplementedError

    def get_recent(self, user_id: int = None, limit: int = 20):
        raise NotImplementedError

    def export_day_to_jsonl(self, day_iso: str = None, out_path: str = None):
        raise NotImplementedError

    def stats_for_day(self, day_iso: str = None):
        raise NotImplementedError

    def enforce_user_limit(self, user_id: int):
        raise NotImplementedError


class PostgresMemoryManager(MemoryManagerBase):
    def __init__(self, dsn: str, daily_cap: int = 10000):
        if not HAS_PSYCOPG:
            raise RuntimeError("psycopg2 is required for Postgres support. Install psycopg2-binary.")
        self.dsn = dsn
        self.daily_cap = daily_cap
        self._ensure_schema()

    def _conn(self):
        conn = psycopg2.connect(self.dsn)
        return conn

    def _ensure_schema(self):
        conn = self._conn()
        with conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    role TEXT,
                    text TEXT,
                    ts TIMESTAMP WITH TIME ZONE DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
                """
            )
        conn.close()

    def add_message(self, user_id: int, role: str, text: str, ts: str = None):
        conn = self._conn()
        with conn:
            cur = conn.cursor()
            if ts:
                cur.execute("INSERT INTO messages(user_id, role, text, ts) VALUES (%s, %s, %s, %s)", (user_id, role, text, ts))
            else:
                cur.execute("INSERT INTO messages(user_id, role, text) VALUES (%s, %s, %s)", (user_id, role, text))
        conn.close()
        # enforce daily cap (global) and per-user limit
        self._enforce_daily_cap_for_date((ts or datetime.utcnow().isoformat()).split("T")[0])
        self.enforce_user_limit(user_id)

    def _enforce_daily_cap_for_date(self, day_iso: str):
        conn = self._conn()
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages WHERE to_char(ts,'YYYY-MM-DD') = %s", (day_iso,))
            count = cur.fetchone()[0]
            if count > self.daily_cap:
                to_delete = count - self.daily_cap
                logger.info(f"Daily cap exceeded for {day_iso}: deleting {to_delete} oldest messages.")
                # delete oldest for that day
                cur.execute(
                    "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE to_char(ts,'YYYY-MM-DD') = %s ORDER BY ts ASC LIMIT %s)",
                    (day_iso, to_delete),
                )
        conn.close()

    def enforce_user_limit(self, user_id: int):
        # enforce MEMORY_LIMIT_PER_USER per user (global config)
        conn = self._conn()
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages WHERE user_id = %s", (user_id,))
            count = cur.fetchone()[0]
            if count > MEMORY_LIMIT_PER_USER:
                to_delete = count - MEMORY_LIMIT_PER_USER
                cur.execute(
                    "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE user_id = %s ORDER BY ts ASC LIMIT %s)",
                    (user_id, to_delete),
                )
        conn.close()

    def get_recent(self, user_id: int = None, limit: int = 20):
        conn = self._conn()
        with conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            if user_id:
                cur.execute("SELECT * FROM messages WHERE user_id = %s ORDER BY ts DESC LIMIT %s", (user_id, limit))
            else:
                cur.execute("SELECT * FROM messages ORDER BY ts DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        conn.close()
        # return as list of dicts oldest-first
        return [dict(r) for r in reversed(rows)]

    def export_day_to_jsonl(self, day_iso: str = None, out_path: str = None):
        day_iso = day_iso or date.today().isoformat()
        out_path = out_path or f"export_{day_iso}.jsonl"
        conn = self._conn()
        with conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("SELECT role, text, user_id, ts FROM messages WHERE to_char(ts,'YYYY-MM-DD') = %s ORDER BY ts ASC", (day_iso,))
            rows = cur.fetchall()
        conn.close()
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                entry = {"role": r["role"], "content": r["text"], "user_id": r["user_id"], "ts": r["ts"].isoformat()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(rows)} messages for {day_iso} -> {out_path}")
        return out_path

    def stats_for_day(self, day_iso: str = None):
        day_iso = day_iso or date.today().isoformat()
        conn = self._conn()
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages WHERE to_char(ts,'YYYY-MM-DD') = %s", (day_iso,))
            count = cur.fetchone()[0]
        conn.close()
        return {"day": day_iso, "count": count}


class SQLiteMemoryManager(MemoryManagerBase):
    def __init__(self, db_path: str = "bot_memory.db", daily_cap: int = 10000):
        self.db_path = db_path
        self.daily_cap = daily_cap
        self._ensure_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        conn = self._conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    role TEXT,
                    text TEXT,
                    ts TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON messages(ts);")
        conn.close()

    def add_message(self, user_id: int, role: str, text: str, ts: str = None):
        ts = ts or datetime.utcnow().isoformat()
        conn = self._conn()
        with conn:
            conn.execute("INSERT INTO messages(user_id, role, text, ts) VALUES (?, ?, ?, ?)", (user_id, role, text, ts))
        conn.close()
        self._enforce_daily_cap_for_date(ts.split("T")[0])
        self.enforce_user_limit(user_id)

    def _enforce_daily_cap_for_date(self, day_iso: str):
        conn = self._conn()
        with conn:
            cur = conn.execute("SELECT COUNT(*) as c FROM messages WHERE ts LIKE ? || '%'", (day_iso,))
            row = cur.fetchone()
            count = row["c"] if row else 0
            if count > self.daily_cap:
                to_delete = count - self.daily_cap
                logger.info(f"Daily cap exceeded for {day_iso}: deleting {to_delete} oldest messages.")
                conn.execute(
                    "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE ts LIKE ? || '%' ORDER BY ts ASC LIMIT ?)", (day_iso, to_delete)
                )
        conn.close()

    def enforce_user_limit(self, user_id: int):
        conn = self._conn()
        with conn:
            cur = conn.execute("SELECT COUNT(*) as c FROM messages WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            count = row["c"] if row else 0
            if count > MEMORY_LIMIT_PER_USER:
                to_delete = count - MEMORY_LIMIT_PER_USER
                conn.execute(
                    "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE user_id = ? ORDER BY ts ASC LIMIT ?)",
                    (user_id, to_delete),
                )
        conn.close()

    def get_recent(self, user_id: int = None, limit: int = 20):
        conn = self._conn()
        with conn:
            if user_id:
                cur = conn.execute("SELECT * FROM messages WHERE user_id = ? ORDER BY ts DESC LIMIT ?", (user_id, limit))
            else:
                cur = conn.execute("SELECT * FROM messages ORDER BY ts DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    def export_day_to_jsonl(self, day_iso: str = None, out_path: str = None):
        day_iso = day_iso or date.today().isoformat()
        out_path = out_path or f"export_{day_iso}.jsonl"
        conn = self._conn()
        with conn:
            cur = conn.execute("SELECT role, text, user_id, ts FROM messages WHERE ts LIKE ? || '%' ORDER BY ts ASC", (day_iso,))
            rows = cur.fetchall()
        conn.close()
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                entry = {"role": r["role"], "content": r["text"], "user_id": r["user_id"], "ts": r["ts"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(rows)} messages for {day_iso} -> {out_path}")
        return out_path

    def stats_for_day(self, day_iso: str = None):
        day_iso = day_iso or date.today().isoformat()
        conn = self._conn()
        with conn:
            cur = conn.execute("SELECT COUNT(*) as c FROM messages WHERE ts LIKE ? || '%'", (day_iso,))
            row = cur.fetchone()
        conn.close()
        return {"day": day_iso, "count": row["c"] if row else 0}


# Initialize the appropriate memory manager
if USE_POSTGRES and DATABASE_URL:
    try:
        memory = PostgresMemoryManager(dsn=DATABASE_URL, daily_cap=DAILY_CAP)
        logger.info("Using PostgresMemoryManager")
    except Exception as e:
        logger.exception("Failed to initialize Postgres memory, falling back to SQLite")
        memory = SQLiteMemoryManager(daily_cap=DAILY_CAP)
else:
    memory = SQLiteMemoryManager(daily_cap=DAILY_CAP)
    logger.info("Using SQLiteMemoryManager")

# ---------------- Original AIAgentController (integrated) ----------------
class AIAgentController:
    """Central AI Agent that controls all bot operations (integrated back into flow)

    It tries a neural intent analysis using OpenAI and falls back to a rule-based heuristic.
    """

    SYSTEM_PROMPT = """You are an advanced AI Agent Controller for a Telegram bot. Your role is to:

    1. ANALYZE user intent and context from messages
    2. DECIDE which tools to use (calculator, search, code generator, or none)
    3. REASON through complex queries step-by-step
    4. SYNTHESIZE information from multiple sources
    5. RESPOND naturally and concisely

    Return a JSON object exactly in this shape:
    {
      "intent": "math|search|code|chat",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation",
      "requires_tools": ["calculator","search","code"] or []
    }
    """

    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    async def analyze_intent(self, message: str, user_id: int) -> dict:
        # Try neural analysis via OpenAI ChatCompletion
        try:
            prompt = f"{self.SYSTEM_PROMPT}\n\nAnalyze this message: '{message}' and output a JSON object as described."
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": f"Analyze: {message}"}],
                temperature=0.1,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON block if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            # attempt to parse
            decision = json.loads(content)
            # sanitize
            decision.setdefault("confidence", 0.75)
            decision.setdefault("requires_tools", [])
            return decision
        except Exception:
            # fallback to rule-based
            return self._fallback_intent(message)

    def _fallback_intent(self, msg: str) -> dict:
        ml = msg.lower()
        if any(k in ml for k in ["calculate", "solve", "integrate"]) or any(s in msg for s in ["+", "-", "*", "/", "^"]):
            return {"intent": "math", "confidence": 0.8, "reasoning": "Contains math keywords/operators", "requires_tools": ["calculator"]}
        elif any(k in ml for k in ["search", "find", "latest", "news", "current"]):
            return {"intent": "search", "confidence": 0.8, "reasoning": "Search keywords detected", "requires_tools": ["search"]}
        elif any(k in ml for k in ["code", "function", "script", "program", "write"]):
            return {"intent": "code", "confidence": 0.8, "reasoning": "Code generation request", "requires_tools": ["code"]}
        else:
            return {"intent": "chat", "confidence": 0.7, "reasoning": "General conversation", "requires_tools": []}


# Instantiate agent
agent = AIAgentController(openai_api_key=OPENAI_API_KEY)

# ---------------- Helper Tools ----------------

def calculator_tool(query: str) -> str:
    try:
        expr = sympify(query)
        result = expr.evalf()
        return f"Result: {result}"
    except Exception:
        try:
            if "integrate" in query.lower() or "∫" in query:
                parts = query.replace("∫", "").split("from")
                func = parts[0].replace("integrate", "").strip()
                if len(parts) > 1 and "to" in parts[1]:
                    lims = parts[1].split("to")
                    lower, upper = float(lims[0].strip()), float(lims[1].strip())
                    x = Symbol('x')
                    result = integrate(sympify(func), (x, lower, upper)).evalf()
                    return f"Integral: {result}"
        except Exception:
            logger.exception("Calc error")
    return None


def search_tool(query: str) -> str:
    if not SERPAPI_API_KEY:
        return None
    try:
        url = "https://serpapi.com/search.json"
        r = requests.get(url, params={"q": query, "api_key": SERPAPI_API_KEY, "num": 3}, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:3]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', '')[:200]
            results.append(f"{title}\n{snippet}")
        return "\n\n".join(results) if results else None
    except Exception:
        logger.exception("Search error")
        return None


def code_tool(prompt: str) -> str:
    return "CODE_GENERATION_REQUESTED"

# ---------------- Response generator with memory ----------------
SYSTEM_PROMPT_CORE = "You are a concise AI assistant in a Telegram bot."

async def generate_response_with_memory(message: str, decision: dict, user_id: int, tool_outputs: dict):
    ts = datetime.utcnow().isoformat()
    memory.add_message(user_id=user_id, role="user", text=message, ts=ts)

    recent = memory.get_recent(user_id=user_id, limit=12)
    context_snippets = []
    for m in recent:
        text = m.get("text") if isinstance(m, dict) else m["text"]
        if len(text) > 1000:
            text = text[:1000] + "..."
        context_snippets.append({"role": m.get("role"), "content": text})

    tool_text = ""
    if tool_outputs:
        parts = []
        for tname, tout in tool_outputs.items():
            if tout:
                parts.append(f"[{tname}]: {tout}")
        if parts:
            tool_text = "\n\nTool results:\n" + "\n".join(parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT_CORE}]
    if context_snippets:
        summary = "Recent conversation context:\n"
        for s in context_snippets:
            summary += f"- {s['role']}: {s['content']}\n"
        messages.append({"role": "system", "content": summary})
    if tool_text:
        messages.append({"role": "system", "content": tool_text})
    messages.append({"role": "user", "content": message})

    try:
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=900,
        )
        assistant_text = resp.choices[0].message.content.strip()
        model_used = "gpt-4o-mini (OpenAI)"
    except Exception as e:
        logger.warning(f"OpenAI failed: {e}, attempting Gemini fallback.")
        assistant_text = None
        model_used = "gemini-fallback"
        try:
            if GEMINI_API_KEY:
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = "\n\n".join([m["content"] for m in messages if m["role"] in ("system", "user")])
                gresp = await asyncio.to_thread(model.generate_content, prompt)
                assistant_text = gresp.text.strip()
        except Exception:
            logger.exception("All LLM providers failed.")
            assistant_text = "Sorry — I cannot reach the AI services right now."

    memory.add_message(user_id=user_id, role="bot", text=assistant_text, ts=datetime.utcnow().isoformat())
    logger.info(f"Responded via {model_used}; memory stats: {memory.stats_for_day()['count']} messages today")
    return assistant_text, model_used

# ---------------- Telegram handlers with admin protections ----------------

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "🤖 **Advanced AI Assistant**\n\n"
        "I use neural models to understand your messages. Send anything and I'll help.\n\n"
        "_Note: messages are stored to improve context. Contact the bot owner to request data deletion._"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text or ""
    user_id = update.effective_user.id
    logger.info(f"Message from {user_id}: {msg[:120]}")
    await update.message.chat.send_action("typing")

    # Step 1: Neural intent analysis
    decision = await agent.analyze_intent(msg, user_id)

    # Step 2: Execute required tools
    tool_outputs = {}
    if "calculator" in decision.get("requires_tools", []):
        tool_outputs["calculator"] = calculator_tool(msg)
    if "search" in decision.get("requires_tools", []):
        tool_outputs["search"] = search_tool(msg)
    if "code" in decision.get("requires_tools", []):
        tool_outputs["code"] = code_tool(msg)

    # Step 3: Generate final response
    try:
        response_text, model = await generate_response_with_memory(msg, decision, user_id, tool_outputs)
    except Exception:
        logger.exception("Error generating response")
        response_text = "Sorry — an internal error occurred."

    try:
        await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(response_text)

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        await update.message.reply_text("Unauthorized — admin only.")
        return
    out = memory.export_day_to_jsonl()
    await update.message.reply_text(f"Exported memory -> `{out}`", parse_mode=ParseMode.MARKDOWN)

async def delete_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        await update.message.reply_text("Unauthorized — admin only.")
        return
    # Admin confirmed delete; in production consider second-step confirmation
    if isinstance(memory, PostgresMemoryManager):
        conn = memory._conn()
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages;")
        conn.close()
    else:
        conn = memory._conn()
        with conn:
            conn.execute("DELETE FROM messages;")
        conn.close()
    await update.message.reply_text("All messages deleted from memory (admin action).")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        await update.message.reply_text("Unauthorized — admin only.")
        return
    stats = memory.stats_for_day()
    await update.message.reply_text(f"Memory stats for {stats['day']}: {stats['count']} messages")

async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_admin(user.id):
        await update.message.reply_text("Unauthorized — admin only.")
        return
    await update.message.reply_text("Shutting down...")
    # Async shutdown
    asyncio.get_event_loop().stop()

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")

# ---------------- Runner ----------------
async def run_bot():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("export_memory", export_command))
    app.add_handler(CommandHandler("delete_all_memory", delete_all_command))
    app.add_handler(CommandHandler("stats_memory", stats_command))
    app.add_handler(CommandHandler("shutdown", shutdown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    logger.info("Bot started. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run_bot())

