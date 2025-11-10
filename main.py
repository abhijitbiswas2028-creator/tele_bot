# =======================================
# ADVANCED NEURAL TELEGRAM BOT
# =======================================

# Step 1: Install dependencies
!pip install -q python-telegram-bot==22.5 openai==1.54.0 google-generativeai==0.8.3 sympy==1.13.1 requests==2.32.3 nest_asyncio

# Step 2: Configuration
TELEGRAM_BOT_TOKEN = "7895572950:AAEKGc5VlYCuzek-4HqN_3o7rsr61y5UlO0"
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyAWToMo0p6oktLIOVRE5t8kUVjdgAcjXCM"
SERPAPI_API_KEY = "f61032922f189efd65105c8a96efe69bf8c75a717f07f6b4cfa10e2b11a1cbaf"

# Step 3: Imports
import logging
import asyncio
import os
import json
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ParseMode
import openai
import google.generativeai as genai
import requests
from sympy import sympify, integrate, Symbol

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# =======================================
# NEURAL AGENT SYSTEM
# =======================================

class AIAgentController:
    """Central AI Agent that controls all bot operations"""

    SYSTEM_PROMPT = """You are an advanced AI Agent Controller for a Telegram bot. Your role is to:

1. ANALYZE user intent and context from messages
2. DECIDE which tools to use (calculator, search, code generator, or none)
3. REASON through complex queries step-by-step
4. SYNTHESIZE information from multiple sources
5. RESPOND naturally and concisely

**Decision Framework:**
- Math/Calculations: Contains operators (+,-,*,/,^), keywords (calculate, solve, integrate), or symbolic expressions
- Web Search: Requests current info, news, facts, or contains "search", "find", "latest"
- Code Generation: Requests programming help, contains "write", "code", "function", "script"
- General Chat: Conversational queries, explanations, advice

**Response Rules:**
- Be concise and direct
- Use bullet points for lists
- Show calculations clearly
- Cite sources when using search
- Format code with proper syntax
- NO unnecessary verbosity

Analyze the following and respond with a JSON decision:
{
  "intent": "math|search|code|chat",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "requires_tools": ["tool1", "tool2"] or []
}"""

    def __init__(self):
        self.conversation_history = {}

    async def analyze_intent(self, message: str, user_id: int) -> dict:
        """Use neural network (GPT) to analyze user intent"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this message: '{message}'"}
                ],
                temperature=0.3,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision = json.loads(content)
            logger.info(f"🧠 Agent Decision: {decision['intent']} (confidence: {decision['confidence']})")
            return decision

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Fallback to rule-based
            return self._fallback_intent(message)

    def _fallback_intent(self, msg: str) -> dict:
        """Rule-based fallback if neural analysis fails"""
        ml = msg.lower()

        if any(k in ml for k in ["calculate", "solve", "integrate"]) or any(s in msg for s in ["+", "-", "*", "/", "^"]):
            return {"intent": "math", "confidence": 0.8, "reasoning": "Contains math keywords/operators", "requires_tools": ["calculator"]}
        elif any(k in ml for k in ["search", "find", "latest", "news", "current"]):
            return {"intent": "search", "confidence": 0.8, "reasoning": "Search keywords detected", "requires_tools": ["search"]}
        elif any(k in ml for k in ["code", "function", "script", "program", "write"]):
            return {"intent": "code", "confidence": 0.8, "reasoning": "Code generation request", "requires_tools": ["code"]}
        else:
            return {"intent": "chat", "confidence": 0.7, "reasoning": "General conversation", "requires_tools": []}

# Initialize Agent
agent = AIAgentController()

# =======================================
# TOOL FUNCTIONS
# =======================================

def calculator_tool(query: str) -> str:
    """Scientific calculator with integral support"""
    try:
        expr = sympify(query)
        result = expr.evalf()
        return f"**Result:** `{result}`"
    except:
        try:
            if "integrate" in query.lower():
                parts = query.replace("integrate", "").replace("∫", "").split("from")
                func = parts[0].strip()
                if len(parts) > 1 and "to" in parts[1]:
                    lims = parts[1].split("to")
                    lower, upper = float(lims[0].strip()), float(lims[1].strip())
                    x = Symbol('x')
                    result = integrate(sympify(func), (x, lower, upper)).evalf()
                    return f"**Integral:** `{result}`"
        except Exception as e:
            logger.error(f"Calc error: {e}")
    return None

def search_tool(query: str) -> str:
    """Web search using SerpAPI"""
    try:
        url = "https://serpapi.com/search.json"
        r = requests.get(url, params={"q": query, "api_key": SERPAPI_API_KEY, "num": 3}, timeout=10)
        r.raise_for_status()
        data = r.json()

        results = []
        for item in data.get("organic_results", [])[:3]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', '')[:200]
            results.append(f"**{title}**\n{snippet}")

        return "\n\n".join(results) if results else None
    except Exception as e:
        logger.error(f"Search error: {e}")
        return None

def code_tool(prompt: str) -> str:
    """Generate code (placeholder for AI to handle)"""
    return "CODE_GENERATION_REQUESTED"

# =======================================
# AI RESPONSE GENERATOR
# =======================================

async def generate_response(message: str, decision: dict, tool_outputs: dict) -> tuple:
    """Generate final response using AI with tool context"""

    context_parts = [f"User Query: {message}\n"]

    if tool_outputs:
        context_parts.append("Tool Results:")
        for tool, output in tool_outputs.items():
            if output:
                context_parts.append(f"- {tool}: {output}")

    context = "\n".join(context_parts)

    system_prompt = f"""You are a highly capable AI assistant. Respond to the user's query using the provided context.

Intent: {decision['intent']}
Reasoning: {decision['reasoning']}

Guidelines:
- Be direct and concise
- If tool results provided, incorporate them naturally
- For math: explain the result briefly
- For search: summarize findings with key points
- For code: provide clean, working code with comments
- For chat: be helpful and conversational

NO preambles, NO unnecessary text."""

    # Try ChatGPT
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip(), "GPT-4o-mini"

    except Exception as e:
        logger.warning(f"GPT failed: {e}, using Gemini")

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"{system_prompt}\n\n{context}"
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return resp.text.strip(), "Gemini-2.5-Flash"

        except Exception as e2:
            logger.error(f"All AI models failed: {e2}")
            return "AI services temporarily unavailable.", "Error"

# =======================================
# TELEGRAM HANDLERS
# =======================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start"""
    welcome = (
        "🤖 **Advanced AI Assistant**\n\n"
        "I use neural networks to understand your needs:\n"
        "• 🧮 Math & calculations\n"
        "• 🔍 Web search\n"
        "• 💻 Code generation\n"
        "• 💬 General chat\n\n"
        "Just send me a message!"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler with neural decision making"""
    msg = update.message.text
    user_id = update.effective_user.id

    logger.info(f"📨 Message from {user_id}: {msg[:50]}...")
    await update.message.chat.send_action("typing")

    # Step 1: Neural intent analysis
    decision = await agent.analyze_intent(msg, user_id)

    # Step 2: Execute required tools
    tool_outputs = {}

    if "calculator" in decision.get("requires_tools", []):
        result = calculator_tool(msg)
        if result:
            tool_outputs["calculator"] = result

    if "search" in decision.get("requires_tools", []):
        result = search_tool(msg)
        if result:
            tool_outputs["search"] = result

    if "code" in decision.get("requires_tools", []):
        tool_outputs["code"] = code_tool(msg)

    # Step 3: Generate final response
    response, model = await generate_response(msg, decision, tool_outputs)

    logger.info(f"✅ Response: {len(response)} chars via {model}")

    # Step 4: Send response
    try:
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    except:
        await update.message.reply_text(response)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    logger.error(f"Error: {context.error}")

# =======================================
# BOT RUNNER
# =======================================

async def run_bot():
    """Run the bot"""
    print("\n🚀 Initializing Neural Bot...\n")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    print("=" * 60)
    print("🎉 NEURAL BOT IS RUNNING!")
    print("=" * 60)
    print("🧠 AI Agent Controller: ACTIVE")
    print("📱 Send messages to your bot on Telegram")
    print("=" * 60 + "\n")

    try:
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n🛑 Shutting down...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        print("✅ Bot stopped")

# Start the bot
asyncio.get_event_loop().run_until_complete(run_bot())
