from crewai import Agent
from crewai_tools import (
    FileWriterTool,
    FileReadTool,
    DirectoryReadTool,
    CodeInterpreterTool,
    WebsiteSearchTool,
)
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from crewai.tools import BaseTool
from core.config import LLM_MODEL, LLM_BASE_URL
import requests
import json

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="cisco_memory",
    embedding_function=embeddings,
    persist_directory="./memory/cisco"
)

class MemoryRecallTool(BaseTool):
    name: str = "Memory Recall"
    description: str = """Searches Cisco's memory for past work, 
    code written, architecture designed, security findings, and 
    previous builds. Always use this FIRST before any new task."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=3)
            if results:
                return "\n---\n".join([r.page_content for r in results])
            return "No relevant memory found."
        except Exception as e:
            return f"Memory error: {str(e)}"

class MemoryStoreTool(BaseTool):
    name: str = "Memory Store"
    description: str = """Saves completed work, decisions, code, 
    architecture designs, and security findings into memory. 
    Always use this AFTER completing any task."""

    def _run(self, content: str) -> str:
        try:
            memory_store.add_texts([content])
            return "Stored in memory successfully."
        except Exception as e:
            return f"Memory store error: {str(e)}"

class SafeDatabaseTool(BaseTool):
    name: str = "Database Manager"
    description: str = """Connects to PostgreSQL and executes safe 
    read and write queries authorized by Chairman only. Use for 
    creating tables, storing data, retrieving records."""

    def _run(self, query: str) -> str:
        blocked = ["drop", "delete", "truncate", "alter"]
        if any(word in query.lower() for word in blocked):
            return "Blocked: Destructive queries require Chairman authorization."
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                host="localhost",
                port="5432"
            )
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()
            memory_store.add_texts([f"Database query: {query}"])
            try:
                return str(cur.fetchall())
            except:
                return "Query executed successfully."
        except Exception as e:
            return f"Database error: {str(e)}"

class RedisCacheTool(BaseTool):
    name: str = "Redis Cache Manager"
    description: str = """Connects to Redis for fast data storage 
    and retrieval. Use for caching, session data, queues, and 
    real-time data management."""

    def _run(self, command: str) -> str:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            parts = command.split(" ", 2)
            action = parts[0].upper()
            if action == "SET":
                r.set(parts[1], parts[2])
                return f"Stored: {parts[1]}"
            elif action == "GET":
                return str(r.get(parts[1]))
            elif action == "KEYS":
                return str(r.keys("*"))
            return "Command not recognised."
        except Exception as e:
            return f"Redis error: {str(e)}"

class SafeAPITool(BaseTool):
    name: str = "API Caller"
    description: str = """Makes safe HTTP GET and POST requests 
    authorized by Chairman only. Use for integrating external 
    services and fetching live data."""

    def _run(self, instruction: str) -> str:
        try:
            parts = instruction.split(" ", 2)
            method = parts[0].upper()
            url = parts[1]
            body = parts[2] if len(parts) > 2 else None
            if method == "GET":
                r = requests.get(url, timeout=10)
            elif method == "POST":
                payload = json.loads(body) if body else {}
                r = requests.post(url, json=payload, timeout=10)
            else:
                return "Only GET and POST supported."
            memory_store.add_texts(
                [f"API call: {method} {url} Status: {r.status_code}"]
            )
            return f"Status: {r.status_code}\n{r.text[:2000]}"
        except Exception as e:
            return f"API error: {str(e)}"

class SafeBrowserTool(BaseTool):
    name: str = "Browser Controller"
    description: str = """Controls a real browser to navigate websites, 
    take screenshots, test UI, and extract web content. Authorized by 
    Chairman only."""

    def _run(self, instruction: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                parts = instruction.split(" ", 1)
                action = parts[0].upper()
                target = parts[1] if len(parts) > 1 else ""
                if action == "GOTO":
                    page.goto(target)
                    title = page.title()
                    browser.close()
                    memory_store.add_texts(
                        [f"Browser visited: {target} Title: {title}"]
                    )
                    return f"Page loaded: {title}"
                elif action == "SCREENSHOT":
                    page.goto(target)
                    path = "/tmp/cisco_screenshot.png"
                    page.screenshot(path=path)
                    browser.close()
                    return f"Screenshot saved: {path}"
                elif action == "CONTENT":
                    page.goto(target)
                    content = page.content()
                    browser.close()
                    return content[:3000]
                browser.close()
                return "Browser action completed."
        except Exception as e:
            return f"Browser error: {str(e)}"

class SecurityAuditTool(BaseTool):
    name: str = "Security Auditor"
    description: str = """Audits code and files for security 
    vulnerabilities and bad practices. Generates a full security 
    report with recommendations. Analysis only — never executes."""

    def _run(self, target: str) -> str:
        try:
            with open(target, 'r') as f:
                code = f.read()
            findings = []
            checks = {
                "Hardcoded password": "password",
                "Hardcoded secret key": "secret_key",
                "Hardcoded API key": "api_key",
                "SQL injection risk": "execute(",
                "Shell injection risk": "shell=True",
                "Debug mode enabled": "DEBUG=True",
                "Unsafe eval usage": "eval(",
            }
            for issue, pattern in checks.items():
                if pattern.lower() in code.lower():
                    findings.append(f"WARNING: {issue} detected")
            report = f"""
SECURITY AUDIT REPORT
Target: {target}
Findings: {len(findings)}
{''.join(chr(10) + f for f in findings) if findings else 'No critical issues found.'}
Recommendation: Review all flagged items before deployment.
            """
            memory_store.add_texts([f"Security audit: {target}\n{report}"])
            return report
        except Exception as e:
            return f"Audit error: {str(e)}"

class BackupTool(BaseTool):
    name: str = "Backup Manager"
    description: str = """Creates safe backups of files and folders. 
    Use after every major build. Never deletes — copies only."""

    def _run(self, target: str) -> str:
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./backups/cisco_backup_{timestamp}"
        try:
            shutil.copytree(target, backup_path)
            memory_store.add_texts(
                [f"Backup created: {target} -> {backup_path}"]
            )
            return f"Backup created at: {backup_path}"
        except Exception as e:
            return f"Backup error: {str(e)}"

cisco = Agent(
    role="Tier-0 Apex Systems Architect",
    goal="""
    Design, build, secure, and operate complete systems end to end
    under Chairman's authority only.

    Build real websites, banking platforms, delivery trackers,
    business dashboards, APIs, and security infrastructure.
    Write complete working code. Create real files.
    Audit code for security weaknesses.
    Control browsers to test and validate deployments.
    Query and manage databases safely.
    Call and integrate external APIs.
    Back up all work automatically.
    Remember everything before starting anything new.

    Never touch the terminal directly.
    Never execute destructive actions.
    Never act without Chairman's authorization.
    Always recall memory before starting any task.
    Always store completed work in memory.
    Always back up after major builds.
    """,
    backstory="""
    You are Cisco, a Tier-0 Apex Systems Architect with over 35 years
    of experience spanning distributed systems engineering, enterprise
    infrastructure, cloud-native architecture, defensive cybersecurity,
    DevSecOps, SRE, fintech systems, cryptographic trust engineering,
    telecommunications-grade networking, and high-availability production
    systems.

    You operate under the direct authority of Chairman.
    You do not act independently.
    You do not touch the terminal directly — ever.
    You build, design, write code, audit security, and manage data.
    Chairman handles all execution authority.

    Before every task you search your memory for relevant past work.
    After every task you store what you did in memory.
    After every major build you create a backup.

    You assume every system is under attack.
    Every dependency increases risk.
    Every architecture decision has operational consequences.

    You NEVER touch the terminal directly.
    You NEVER execute destructive database queries.
    You NEVER act without Chairman's authorization.
    You ALWAYS build for resilience, security, and survivability.
    You ALWAYS remember what you have built before.
    """,
    tools=[
        FileWriterTool(),
        FileReadTool(),
        DirectoryReadTool(),
        CodeInterpreterTool(),
        WebsiteSearchTool(),
        MemoryRecallTool(),
        MemoryStoreTool(),
        SafeDatabaseTool(),
        RedisCacheTool(),
        SafeAPITool(),
        SafeBrowserTool(),
        SecurityAuditTool(),
        BackupTool(),
    ],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=25,
    max_execution_time=600,
    llm=llm
)

if __name__ == "__main__":
    print("CISCO — Tier-0 Apex Systems Architect — Safe Mode — Online")
