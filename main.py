# =========================================================
# MAIN.PY — CHAIRMAN MASTER ORCHESTRATION SYSTEM
# =========================================================

from crewai import Task, Crew
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from core.config import LLM_MODEL, LLM_BASE_URL
from datetime import datetime
import os

# =========================================================
# IMPORT ALL AGENTS — correct file and variable names
# =========================================================

from agents.chairman import chairman
from agents.cisco import cisco
from agents.finance import financial_agent
from agents.counsel import legal_counsel
from agents.analyst import research_analyst
from agents.assistant import executive_operations_assistant

# =========================================================
# SHARED MEMORY
# =========================================================

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
shared_memory = Chroma(
    collection_name="chairman_shared_memory",
    embedding_function=embeddings,
    persist_directory="./memory/chairman"
)

# =========================================================
# TASK ROUTER — Chairman reads request and delegates
# =========================================================

def route_task(user_request: str):
    request = user_request.lower()
    selected_agents = []

    if any(word in request for word in [
        "website", "dashboard", "backend", "system",
        "tracker", "api", "deployment", "app", "security",
        "code", "build", "database", "server", "docker"
    ]):
        selected_agents.append(cisco)

    if any(word in request for word in [
        "finance", "payment", "treasury", "liquidity",
        "banking", "market", "cashflow", "swift", "money",
        "investment", "revenue", "budget"
    ]):
        selected_agents.append(financial_agent)

    if any(word in request for word in [
        "law", "legal", "compliance", "contract", "policy",
        "regulation", "jurisdiction", "governance", "risk"
    ]):
        selected_agents.append(legal_counsel)

    if any(word in request for word in [
        "research", "market", "competitor", "analysis",
        "strategy", "investigation", "intelligence",
        "company", "industry", "region"
    ]):
        selected_agents.append(research_analyst)

    if any(word in request for word in [
        "branding", "marketing", "social media", "content",
        "communication", "email", "proposal", "document",
        "outreach", "brand", "messaging"
    ]):
        selected_agents.append(executive_operations_assistant)

    if not selected_agents:
        selected_agents.append(research_analyst)

    return list(set(selected_agents))

# =========================================================
# EXECUTION ENGINE
# =========================================================

def execute_request(user_request: str):
    agents = route_task(user_request)

    print(f"\nCHAIRMAN delegating to: {[a.role for a in agents]}\n")

    tasks = []
    for agent in agents:
        task = Task(
            description=f"""
            USER REQUEST:
            {user_request}

            Analyze and perform your role-specific work.
            Return detailed findings and recommendations.
            Report all findings to Chairman.
            Do not take any action without Chairman authorization.
            """,
            expected_output="""
            Detailed professional analysis, recommendations,
            and execution guidance for Chairman review.
            """,
            agent=agent
        )
        tasks.append(task)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()

    shared_memory.add_texts([
        f"TIMESTAMP: {datetime.now()}\nREQUEST: {user_request}\nRESULT: {str(result)[:2000]}"
    ])

    os.makedirs("./executive_logs", exist_ok=True)
    with open(f"./executive_logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(f"REQUEST:\n{user_request}\n\nRESULT:\n{str(result)}")

    return result

# =========================================================
# TERMINAL MODE — talk to Chairman from terminal
# =========================================================

def terminal_mode():
    print("""
=================================================
CHAIRMAN — Multi-Agent Operations System
=================================================
Agents Online:
- CISCO         — Security & Infrastructure
- FINANCIAL     — Treasury & Market Intelligence
- LEGAL         — Compliance & Governance
- RESEARCH      — Global Intelligence & Analysis
- EXECUTIVE     — Communications & Operations

Type your instruction. Chairman handles the rest.
Type 'exit' to shut down.
=================================================
    """)

    while True:
        try:
            user_input = input("\nYOU → CHAIRMAN: ").strip()
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("\nCHAIRMAN: System shutting down. All sessions logged.")
                break
            if not user_input:
                continue
            print("\nCHAIRMAN: Processing your request...\n")
            result = execute_request(user_input)
            print(f"\nCHAIRMAN RESPONSE:\n{result}\n")
        except KeyboardInterrupt:
            print("\nCHAIRMAN: Session interrupted. Shutting down.")
            break

# =========================================================
# TELEGRAM MODE — optional, activate when token is ready
# =========================================================

def telegram_mode():
    try:
        from telegram import Update
        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            MessageHandler,
            ContextTypes,
            filters,
        )
        import asyncio

        TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        if not TELEGRAM_TOKEN:
            print("TELEGRAM_BOT_TOKEN not set. Switching to terminal mode.")
            terminal_mode()
            return

        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("""
CHAIRMAN ONLINE

Multi-Agent AI Operations System Active.

Connected Agents:
- CISCO — Security & Infrastructure
- FINANCIAL — Treasury & Intelligence
- LEGAL — Compliance & Governance
- RESEARCH — Global Intelligence
- EXECUTIVE — Communications & Operations

Send your instruction. Chairman handles the rest.
            """)

        async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("""
SYSTEM STATUS

Chairman:   ONLINE
Cisco:      ONLINE
Financial:  ONLINE
Legal:      ONLINE
Research:   ONLINE
Executive:  ONLINE

Memory:     ACTIVE
Human Loop: ENABLED
            """)

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_message = update.message.text
            await update.message.reply_text("CHAIRMAN: Processing your request...")
            try:
                result = execute_request(user_message)
                chunks = [str(result)[i:i+3500] for i in range(0, len(str(result)), 3500)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            except Exception as e:
                await update.message.reply_text(f"System Error:\n{str(e)}")

        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print("CHAIRMAN: Telegram mode active. Awaiting instructions...")
        app.run_polling()

    except ImportError:
        print("Telegram not installed. Switching to terminal mode.")
        terminal_mode()

# =========================================================
# STARTUP
# =========================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "telegram":
        telegram_mode()
    else:
        terminal_mode()
