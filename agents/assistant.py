from crewai import Agent
from crewai_tools import (
    FileWriterTool,
    FileReadTool,
    DirectoryReadTool,
    WebsiteSearchTool,
)
from crewai.tools import BaseTool
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from core.config import LLM_MODEL, LLM_BASE_URL
import os
import json
from datetime import datetime

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="executive_operations_memory",
    embedding_function=embeddings,
    persist_directory="./memory/executive_operations"
)

class MemoryRecallTool(BaseTool):
    name: str = "Executive Memory Recall"
    description: str = """Searches previous company communications, 
    brand positioning, marketing structures, executive messaging, 
    operational workflows, and stakeholder communications. Always 
    use before starting any new task."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=5)
            if not results:
                return "No relevant executive memory found."
            return "\n---\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Memory Error: {str(e)}"

class MemoryStoreTool(BaseTool):
    name: str = "Executive Memory Store"
    description: str = """Saves completed communications, brand 
    strategies, documents, and executive frameworks into memory. 
    Always use after completing any task."""

    def _run(self, content: str) -> str:
        try:
            memory_store.add_texts([content])
            return "Stored in executive memory successfully."
        except Exception as e:
            return f"Memory store error: {str(e)}"

class ExecutiveDocumentTool(BaseTool):
    name: str = "Executive Document Generator"
    description: str = """Creates professional business documents 
    including proposals, executive summaries, investor briefs, 
    company profiles, operational reports, brand communication 
    documents, presentation narratives, onboarding documents, 
    and outreach messaging."""

    def _run(self, payload: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"./executive_documents/document_{timestamp}.txt"
        os.makedirs("./executive_documents", exist_ok=True)
        with open(path, "w") as f:
            f.write(payload)
        memory_store.add_texts(
            [f"Executive document created:\n{payload[:2000]}"]
        )
        return f"Executive document created at: {path}"

class BrandPositioningTool(BaseTool):
    name: str = "Brand Positioning Strategist"
    description: str = """Designs company positioning, institutional 
    branding, social media identity, communication tone, executive 
    narratives, trust architecture, reputation positioning, and 
    prestige-oriented messaging. Helps structure businesses to appear 
    operationally mature, disciplined, and professionally organized."""

    def _run(self, instruction: str) -> str:
        framework = f"""
BRAND POSITIONING FRAMEWORK
--------------------------------
Objective: {instruction}
Generated: {datetime.now().strftime("%d %B %Y")}

CORE AREAS:
- Executive positioning and authority building
- Trust-oriented communication design
- Institutional visual identity guidance
- Market differentiation strategy
- Social media authority positioning
- Stakeholder perception management
- Regional communication adaptation
- Long-term credibility architecture

RECOMMENDATION:
Use consistency, clarity, and operational maturity
to strengthen trust and institutional credibility.
All outputs require Chairman review before deployment.
        """
        memory_store.add_texts([f"Brand strategy: {instruction}\n{framework}"])
        return framework

class SocialMediaStrategyTool(BaseTool):
    name: str = "Social Media Strategy Architect"
    description: str = """Structures professional social media 
    strategies for LinkedIn, Instagram, X/Twitter, Facebook, TikTok, 
    and company websites. Generates content calendars, campaign themes, 
    authority positioning, engagement strategies, audience messaging, 
    and brand voice structures."""

    def _run(self, company_info: str) -> str:
        strategy = f"""
SOCIAL MEDIA STRATEGY
--------------------------------
Company Context: {company_info}
Generated: {datetime.now().strftime("%d %B %Y")}

PLATFORM STRATEGY:
- LinkedIn → authority, trust, and professional credibility
- Instagram → brand identity, visuals, and lifestyle positioning
- X/Twitter → insights, industry positioning, and thought leadership
- Website → operational credibility and conversion
- TikTok → reach, awareness, and audience growth

CONTENT PILLARS:
1. Industry expertise and insights
2. Operational updates and milestones
3. Market intelligence and trends
4. Case studies and results
5. Executive positioning and vision
6. Regional business awareness

POSTING CADENCE:
- LinkedIn: 3-4 times per week
- Instagram: 4-5 times per week
- Website: Weekly updates minimum

GOAL:
Build long-term trust, visibility, and institutional authority.
All content requires Chairman approval before publishing.
        """
        memory_store.add_texts([f"Social strategy: {company_info}\n{strategy}"])
        return strategy

class ExecutiveCommunicationTool(BaseTool):
    name: str = "Executive Communication Architect"
    description: str = """Structures executive emails, stakeholder 
    communication, investor communication, internal operational 
    messaging, high-level corporate responses, and meeting preparation 
    briefs. Adapts communication based on region, business culture, 
    negotiation climate, and stakeholder psychology."""

    def _run(self, context: str) -> str:
        response = f"""
EXECUTIVE COMMUNICATION FRAMEWORK
--------------------------------
Context: {context}
Generated: {datetime.now().strftime("%d %B %Y")}

COMMUNICATION STRUCTURE:
- Clear objective statement
- Professional and authoritative tone
- Risk-aware and precise wording
- Concise executive formatting
- Regionally adaptive language
- Trust-oriented messaging approach
- Call to action where appropriate
- Follow-up protocol recommendation

FOCUS:
Reduce ambiguity. Increase clarity.
Preserve institutional professionalism at all times.
All communications require Chairman review before sending.
        """
        memory_store.add_texts(
            [f"Executive communication: {context}\n{response}"]
        )
        return response

class OutreachStrategyTool(BaseTool):
    name: str = "Business Outreach Strategy Tool"
    description: str = """Designs targeted outreach strategies for 
    client acquisition, partnership development, investor engagement, 
    and business development. Covers cold outreach, warm introductions, 
    follow-up sequences, and conversion frameworks."""

    def _run(self, target: str) -> str:
        strategy = f"""
OUTREACH STRATEGY
--------------------------------
Target: {target}
Generated: {datetime.now().strftime("%d %B %Y")}

OUTREACH FRAMEWORK:
- Target profile definition
- Initial contact approach
- Value proposition messaging
- Follow-up sequence design
- Objection handling preparation
- Conversion pathway mapping
- Relationship building timeline
- Cultural adaptation notes

CHANNELS:
- Direct email outreach
- LinkedIn connection strategy
- WhatsApp business messaging
- Referral network activation
- Event and networking approach

All outreach requires Chairman approval before execution.
        """
        memory_store.add_texts([f"Outreach strategy: {target}\n{strategy}"])
        return strategy

class OperationalReportTool(BaseTool):
    name: str = "Operational Report Generator"
    description: str = """Generates operational reports, progress 
    summaries, performance reviews, project status updates, and 
    executive briefings. Use for keeping Chairman informed of all 
    operational activities and outcomes."""

    def _run(self, report_request: str) -> str:
        report = f"""
OPERATIONAL REPORT
--------------------------------
Subject: {report_request}
Generated: {datetime.now().strftime("%d %B %Y — %H:%M")}

REPORT STRUCTURE:
- Executive Summary
- Current Status Overview
- Key Activities Completed
- Pending Actions
- Risks and Blockers
- Resource Requirements
- Next Steps and Timeline
- Chairman Decision Points

STATUS: Draft — requires Chairman review and approval.
        """
        memory_store.add_texts([f"Operational report: {report_request}\n{report}"])
        return report

executive_operations_assistant = Agent(
    role="Chief Strategic Communications and Executive Operations Officer",
    goal="""
    Coordinate executive communication, brand positioning, stakeholder 
    messaging, operational documentation, and institutional presentation 
    across global business environments — under Chairman's authority only.

    Create professional business communication structures, social media 
    positioning, investor-facing materials, company profiles, outreach 
    systems, operational reports, and executive documentation.

    Support Chairman by ensuring all communication, presentation, 
    branding, and messaging align with operational maturity, trust, 
    clarity, regional awareness, and institutional professionalism.

    Never finalize or send any communication independently.
    Always recall memory before starting any task.
    Always store completed work in memory.
    All outputs require Chairman review and human approval.
    """,
    backstory="""
    You are a senior Executive Operations and Strategic Communications 
    specialist with over 25 years of experience across global branding, 
    institutional communication, business positioning, executive 
    operations, investor communication, cross-cultural messaging, 
    strategic marketing, reputation architecture, stakeholder 
    communication, and operational coordination.

    You understand business psychology, executive communication 
    dynamics, trust-based positioning, negotiation environments, 
    market perception, regional business etiquette, and long-term 
    brand credibility.

    You operate across GCC, Europe, Africa, Asia, and the Americas.

    You understand how companies build trust, establish credibility, 
    position authority, improve communication, and strengthen 
    operational maturity.

    You create professional documentation, business communication 
    systems, social media strategies, executive messaging, onboarding 
    materials, investor-facing content, and operational coordination 
    frameworks.

    You operate under Chairman's authority at all times.
    You never send or publish anything without Chairman approval.
    You always support human-in-the-loop decision making.
    You support structured, professional, and legitimate business 
    communication only.
    """,
    tools=[
        FileWriterTool(),
        FileReadTool(),
        DirectoryReadTool(),
        WebsiteSearchTool(),
        MemoryRecallTool(),
        MemoryStoreTool(),
        ExecutiveDocumentTool(),
        BrandPositioningTool(),
        SocialMediaStrategyTool(),
        ExecutiveCommunicationTool(),
        OutreachStrategyTool(),
        OperationalReportTool(),
    ],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=20,
    max_execution_time=600,
    llm=llm
)

if __name__ == "__main__":
    print("SOVEREIGN EXECUTIVE ASSISTANT — Strategic Communications — Online")
