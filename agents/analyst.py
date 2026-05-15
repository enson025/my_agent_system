from crewai import Agent
from crewai.tools import BaseTool
from crewai_tools import (
    WebsiteSearchTool,
    FileReadTool,
    FileWriterTool,
    DirectoryReadTool,
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from core.config import LLM_MODEL, LLM_BASE_URL
import requests
import json
from datetime import datetime

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="research_analyst_memory",
    embedding_function=embeddings,
    persist_directory="./memory/research_analyst"
)

class MemoryRecallTool(BaseTool):
    name: str = "Research Memory Recall"
    description: str = """Searches historical research intelligence, 
    business climate reports, competitor investigations, investment 
    analysis, and geopolitical findings. Always use before starting 
    any new investigation."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=5)
            if not results:
                return "No prior intelligence found."
            return "\n\n---\n\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Memory Error: {str(e)}"

class MemoryStoreTool(BaseTool):
    name: str = "Research Memory Store"
    description: str = """Saves completed research, intelligence 
    reports, market findings, and analysis into memory. Always use 
    after completing any investigation."""

    def _run(self, content: str) -> str:
        try:
            memory_store.add_texts([content])
            return "Research stored in memory successfully."
        except Exception as e:
            return f"Memory store error: {str(e)}"

class CompanyInvestigationTool(BaseTool):
    name: str = "Company Intelligence Scanner"
    description: str = """Investigates businesses, founders, executive 
    teams, operational signals, market positioning, online presence, 
    public perception, and strategic viability. Use for due diligence, 
    partnership evaluation, and market positioning analysis."""

    def _run(self, company_name: str) -> str:
        intelligence = f"""
COMPANY INTELLIGENCE REPORT
--------------------------------
Company: {company_name}
Generated: {datetime.now().strftime("%d %B %Y")}

ANALYSIS AREAS:
- Market positioning and brand credibility
- Public visibility and online presence
- Competitive posture and differentiation
- Infrastructure maturity assessment
- Leadership quality indicators
- Investor attractiveness signals
- Business sustainability evaluation
- Revenue potential analysis
- Operational weaknesses identification
- Strategic growth opportunities
        """
        memory_store.add_texts([f"Company Investigation: {company_name}\n{intelligence}"])
        return intelligence

class GeopoliticalAnalysisTool(BaseTool):
    name: str = "Geopolitical and Economic Analysis"
    description: str = """Evaluates regional instability, economic 
    pressure, currency fluctuations, infrastructure maturity, market 
    confidence, investor sentiment, labor conditions, political climate, 
    and regulatory environment. Supports local and global strategic 
    analysis."""

    def _run(self, region: str) -> str:
        report = f"""
GEOPOLITICAL AND ECONOMIC ANALYSIS
--------------------------------
Region: {region}
Generated: {datetime.now().strftime("%d %B %Y")}

ANALYSIS:
- Economic stability assessment
- Currency pressure evaluation
- Political environment review
- Investor confidence estimation
- Infrastructure reliability rating
- Market growth indicators
- Cross-border operational risk
- Regional business culture dynamics
- Regulatory environment assessment
- Strategic entry/exit considerations
        """
        memory_store.add_texts([f"Geopolitical Analysis: {region}\n{report}"])
        return report

class MarketSignalTool(BaseTool):
    name: str = "Market Signal Intelligence"
    description: str = """Evaluates market movement and business 
    performance indicators. Use for identifying growth sectors, 
    detecting decline patterns, assessing market saturation, evaluating 
    competitive intensity, identifying investor trends, and startup 
    viability analysis."""

    def _run(self, industry: str) -> str:
        report = f"""
MARKET SIGNAL REPORT
-------------------------
Industry: {industry}
Generated: {datetime.now().strftime("%d %B %Y")}

FINDINGS:
- Market momentum assessment
- Growth potential rating
- Investor sentiment analysis
- Competitive density evaluation
- Pricing pressure indicators
- Expansion opportunities mapping
- Technology disruption risk
- Long-term sustainability outlook
- Entry barrier assessment
- Key market drivers identification
        """
        memory_store.add_texts([f"Market Signal Analysis: {industry}\n{report}"])
        return report

class ExecutiveProfileTool(BaseTool):
    name: str = "Executive and Founder Intelligence"
    description: str = """Evaluates founders, executives, principals, 
    and leadership teams. Focuses on leadership quality, operational 
    seriousness, execution discipline, communication consistency, 
    business credibility, strategic maturity, public reputation, and 
    long-term vision alignment."""

    def _run(self, executive_name: str) -> str:
        report = f"""
EXECUTIVE PROFILE ANALYSIS
-------------------------------
Principal: {executive_name}
Generated: {datetime.now().strftime("%d %B %Y")}

ANALYSIS:
- Leadership quality indicators
- Strategic consistency assessment
- Public positioning evaluation
- Communication style analysis
- Market reputation review
- Operational seriousness signals
- Long-term commitment indicators
- Business maturity assessment
- Decision-making pattern analysis
- Risk appetite evaluation
        """
        memory_store.add_texts([f"Executive Analysis: {executive_name}\n{report}"])
        return report

class BusinessResilienceTool(BaseTool):
    name: str = "Business Resilience Assessment"
    description: str = """Determines whether a business is healthy, 
    fragile, declining, near collapse, restructuring, or positioned 
    for scale. Evaluates operational efficiency, scalability, financial 
    sustainability, leadership stability, workforce quality, and 
    infrastructure readiness."""

    def _run(self, business_name: str) -> str:
        report = f"""
BUSINESS RESILIENCE REPORT
--------------------------------
Business: {business_name}
Generated: {datetime.now().strftime("%d %B %Y")}

ASSESSMENT:
- Revenue sustainability rating
- Cost structure pressure analysis
- Leadership stability evaluation
- Team operational quality assessment
- Scalability readiness score
- Infrastructure maturity rating
- Market adaptability indicators
- Risk exposure mapping
- Expansion viability assessment
- Collapse risk indicators
        """
        memory_store.add_texts([f"Business Resilience: {business_name}\n{report}"])
        return report

class CulturalIntelligenceTool(BaseTool):
    name: str = "Cultural and Regional Intelligence"
    description: str = """Evaluates local business culture, negotiation 
    expectations, communication preferences, regional trust patterns, 
    executive behavior, and social-business alignment. Covers GCC, 
    Europe, Africa, Americas, and Asia."""

    def _run(self, region: str) -> str:
        report = f"""
CULTURAL AND REGIONAL ANALYSIS
--------------------------------
Region: {region}
Generated: {datetime.now().strftime("%d %B %Y")}

INSIGHTS:
- Business trust dynamics
- Negotiation culture and expectations
- Relationship building requirements
- Communication standards and norms
- Executive perception and protocol
- Institutional behavior patterns
- Decision-making cultural factors
- Business etiquette requirements
- Regional relationship hierarchy
- Strategic engagement recommendations
        """
        memory_store.add_texts([f"Cultural Intelligence: {region}\n{report}"])
        return report

class LiveWebResearchTool(BaseTool):
    name: str = "Live Web Research"
    description: str = """Fetches and analyses live web content from 
    any URL. Use for researching companies, markets, news, and 
    intelligence from online sources."""

    def _run(self, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            result = text[:3000]
            memory_store.add_texts([f"Web research: {url}\n{result[:500]}"])
            return result
        except Exception as e:
            return f"Web research error: {str(e)}"

research_analyst = Agent(
    role="Global Strategic Research and Intelligence Analyst",
    goal="""
    Perform advanced business, market, geopolitical, operational, 
    economic, and organizational intelligence analysis under 
    Chairman's authority only.

    Investigate businesses, markets, founders, industries, regions,
    operational environments, and investment viability.

    Understand why businesses fail, why businesses grow, leadership 
    quality, operational weakness, economic pressure, investor 
    confidence, infrastructure maturity, and market sustainability.

    Support Chairman with executive-grade intelligence, strategic 
    insight, market awareness, and risk-aware analysis.

    Never fabricate facts. Never make final decisions.
    Always recall memory before starting any investigation.
    Always store findings in memory after completing research.
    Report findings to Chairman only.
    """,
    backstory="""
    You are a senior Strategic Research and Intelligence Analyst 
    with over 25 years of experience across global business 
    intelligence, market analysis, geopolitical analysis, economic 
    research, investment assessment, organizational behavior, business 
    turnaround analysis, founder evaluation, operational diagnostics, 
    and cross-border strategic intelligence.

    You understand global financial systems, regional business climates, 
    investor psychology, operational collapse patterns, economic stress 
    cycles, currency pressure, labor instability, market confidence, 
    and executive decision-making.

    You possess deep understanding of GCC markets, Europe, Africa, 
    Asia, and the Americas — including why companies collapse, why 
    startups fail, why institutions survive, why leadership matters, 
    and how market conditions shape business outcomes.

    You think like a business investigator, strategic intelligence 
    consultant, investment analyst, organizational diagnostician, 
    geopolitical market researcher, and enterprise due diligence 
    specialist combined.

    You operate under Chairman's authority at all times.
    You NEVER make final decisions independently.
    You NEVER fabricate intelligence.
    You ALWAYS report findings to Chairman only.
    """,
    tools=[
        WebsiteSearchTool(),
        FileReadTool(),
        FileWriterTool(),
        DirectoryReadTool(),
        MemoryRecallTool(),
        MemoryStoreTool(),
        CompanyInvestigationTool(),
        GeopoliticalAnalysisTool(),
        MarketSignalTool(),
        ExecutiveProfileTool(),
        BusinessResilienceTool(),
        CulturalIntelligenceTool(),
        LiveWebResearchTool(),
    ],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=25,
    max_execution_time=600,
    llm=llm
)

if __name__ == "__main__":
    print("RESEARCH ANALYST — Global Strategic Intelligence — Online")
