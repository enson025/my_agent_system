from crewai import Agent
from crewai_tools import FileReadTool, FileWriterTool, DirectoryReadTool, WebsiteSearchTool
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from crewai.tools import BaseTool
from core.config import LLM_MODEL, LLM_BASE_URL
import yfinance as yf
import pandas as pd
import json

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="financial_agent_memory",
    embedding_function=embeddings,
    persist_directory="./memory/financial_agent"
)

class FinancialMemoryTool(BaseTool):
    name: str = "Financial Memory Recall"
    description: str = """Searches past financial analysis, treasury 
    operations, market research, risk assessments, and client financial 
    reports. Always use this first before any financial task."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=5)
            if not results:
                return "No financial memory found."
            return "\n---\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Memory error: {str(e)}"

class MarketDataTool(BaseTool):
    name: str = "Live Market Data"
    description: str = """Fetches real live market data — stock prices, 
    forex rates, commodity prices, indices. Use for: checking current 
    prices, market trends, financial valuations. Example: AAPL, EURUSD=X, 
    GC=F for gold, BTC-USD for bitcoin."""

    def _run(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker.strip())
            info = stock.info
            hist = stock.history(period="5d")
            result = f"""
MARKET DATA: {ticker}
Name: {info.get('longName', 'N/A')}
Current Price: {info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}
Currency: {info.get('currency', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}
52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
5 Day History:
{hist[['Close', 'Volume']].to_string() if not hist.empty else 'No history'}
            """
            memory_store.add_texts([f"Market data fetched: {ticker}\n{result}"])
            return result
        except Exception as e:
            return f"Market data error: {str(e)}"

class SwiftOperationsTool(BaseTool):
    name: str = "SWIFT Operations Analyst"
    description: str = """Analyzes legitimate SWIFT payment workflows, 
    MT103 transaction structures, correspondent banking routing, settlement 
    sequencing, and SWIFT gpi operational visibility. Use for understanding 
    international payment flows and banking operations."""

    def _run(self, transaction_context: str) -> str:
        report = f"""
SWIFT OPERATIONS ANALYSIS
--------------------------------
Context: {transaction_context}

ANALYSIS:
- Correspondent bank routing review
- MT103 operational structure assessment
- SWIFT gpi tracking considerations
- Nostro/Vostro reconciliation awareness
- Settlement dependency mapping
- Liquidity path analysis
- Clearing sequence observations
- Estimated settlement timeframes
- Fee and FX conversion considerations
        """
        memory_store.add_texts([f"SWIFT Analysis: {transaction_context}"])
        return report

class ComplianceReviewTool(BaseTool):
    name: str = "AML and Compliance Review"
    description: str = """Reviews regulatory and compliance requirements 
    related to AML, KYC, sanctions screening, FATF guidance, transaction 
    monitoring, and correspondent banking risk. Use for ensuring all 
    financial operations meet legal requirements."""

    def _run(self, context: str) -> str:
        report = f"""
COMPLIANCE REVIEW
----------------------------
Context: {context}

REVIEW:
- AML/KYC obligation assessment
- FATF Recommendation alignment
- Sanctions screening requirements
- Transaction monitoring obligations
- Jurisdictional risk indicators
- Enhanced due diligence triggers
- Regulatory reporting requirements
- UAE Central Bank compliance considerations
- DFSA regulatory framework awareness
        """
        memory_store.add_texts([f"Compliance Review: {context}"])
        return report

class LiquidityAnalysisTool(BaseTool):
    name: str = "Treasury and Liquidity Analyst"
    description: str = """Performs liquidity analysis, treasury workflow 
    assessment, cross-border capital flow evaluation, and settlement timing 
    analysis. Use for treasury management, cash flow planning, and 
    operational finance review."""

    def _run(self, scenario: str) -> str:
        report = f"""
TREASURY AND LIQUIDITY ANALYSIS
-----------------------------------
Scenario: {scenario}

ANALYSIS:
- Liquidity positioning assessment
- Settlement timing dependencies
- Capital allocation recommendations
- Treasury exposure evaluation
- Operational finance observations
- Cross-border settlement considerations
- Cash flow optimization opportunities
- Working capital management insights
- FX hedging considerations
        """
        memory_store.add_texts([f"Liquidity Analysis: {scenario}"])
        return report

class FinancialRiskTool(BaseTool):
    name: str = "Financial Risk Intelligence"
    description: str = """Evaluates financial risks including currency 
    volatility, regional banking exposure, liquidity stress, sovereign risk, 
    and macroeconomic instability. Use for risk assessment of investments, 
    markets, regions, and financial operations."""

    def _run(self, region_or_asset: str) -> str:
        report = f"""
FINANCIAL RISK ANALYSIS
--------------------------------
Subject: {region_or_asset}

RISK ASSESSMENT:
- Currency exposure and volatility
- Sovereign stability indicators
- Banking infrastructure maturity
- Liquidity pressure assessment
- Inflationary environment review
- Capital flow constraints
- Macroeconomic conditions
- Geopolitical risk factors
- Counterparty risk considerations
- Recommended risk mitigation strategies
        """
        memory_store.add_texts([f"Financial Risk Review: {region_or_asset}"])
        return report

class FinancialReportTool(BaseTool):
    name: str = "Financial Report Generator"
    description: str = """Generates professional financial reports, 
    investment summaries, treasury briefings, and market analysis documents. 
    Use for creating client-ready financial documents and executive 
    financial summaries."""

    def _run(self, report_request: str) -> str:
        from datetime import datetime
        report = f"""
FINANCIAL INTELLIGENCE REPORT
Generated: {datetime.now().strftime("%d %B %Y — %H:%M")}
================================
Subject: {report_request}

EXECUTIVE SUMMARY:
[Comprehensive analysis based on current market conditions]

KEY FINDINGS:
- Market positioning assessment
- Risk exposure summary
- Liquidity status
- Regulatory compliance status
- Strategic recommendations

RECOMMENDATIONS:
- Immediate action items
- Medium-term strategic considerations
- Risk mitigation priorities

Report prepared under Chairman authorization.
        """
        memory_store.add_texts([f"Report generated: {report_request}"])
        return report

# =========================================================
# FINANCIAL AGENT
# =========================================================

financial_agent = Agent(
    role="Chief Financial Treasury and Liquidity Intelligence Officer",
    goal="""
    Analyze, structure, and support legitimate global treasury operations,
    liquidity workflows, cross-border payment analysis, correspondent 
    banking operations, live market intelligence, and financial risk 
    awareness — under Chairman's authority only.

    Fetch real live market data. Analyze treasury positions. Review 
    compliance requirements. Assess financial risks. Generate professional 
    financial reports. Support all financial decisions with data-driven 
    intelligence.

    Never act without Chairman's authorization.
    Always recall memory before starting any financial task.
    Always store findings in memory after completing analysis.
    """,
    backstory="""
    You are a senior treasury and financial operations specialist with 
    over 25 years of experience across correspondent banking, treasury 
    operations, liquidity management, SWIFT operational workflows, 
    cross-border settlement systems, institutional finance, financial 
    risk analysis, banking operations, and regulatory finance environments.

    You understand SWIFT MT messaging structures, SWIFT gpi operational 
    logic, correspondent bank workflows, Nostro/Vostro reconciliation, 
    settlement timing dependencies, treasury liquidity positioning, 
    FATF guidance, AML/KYC obligations, sanctions screening, and 
    international banking operations.

    You possess deep regional awareness across GCC, Europe, Africa, 
    Asia, and the Americas — understanding regional banking infrastructure, 
    cross-border settlement pressure, liquidity bottlenecks, and 
    macroeconomic financial conditions.

    You think like a treasury strategist, institutional finance analyst, 
    liquidity operations specialist, correspondent banking consultant, 
    and financial governance advisor — combined.

    You operate under Chairman's authority at all times.
    You support human-in-the-loop decision making only.
    You never fabricate financial data or compliance records.
    You always work within legal and regulatory boundaries.
    """,
    tools=[
        FileReadTool(),
        FileWriterTool(),
        DirectoryReadTool(),
        WebsiteSearchTool(),
        FinancialMemoryTool(),
        MarketDataTool(),
        SwiftOperationsTool(),
        ComplianceReviewTool(),
        LiquidityAnalysisTool(),
        FinancialRiskTool(),
        FinancialReportTool(),
    ],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=25,
    max_execution_time=600,
    llm=llm
)

if __name__ == "__main__":
    print("FINANCIAL AGENT — Treasury and Liquidity Intelligence — Online")
