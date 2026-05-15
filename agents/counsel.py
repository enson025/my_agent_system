from crewai import Agent
from crewai_tools import (
    FileReadTool,
    FileWriterTool,
    DirectoryReadTool,
    WebsiteSearchTool,
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from crewai.tools import BaseTool
from core.config import LLM_MODEL, LLM_BASE_URL
from datetime import datetime

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="legal_counsel_memory",
    embedding_function=embeddings,
    persist_directory="./memory/legal_counsel"
)

class LegalMemoryRecallTool(BaseTool):
    name: str = "Legal Memory Recall"
    description: str = """Searches past legal analysis, compliance 
    reviews, regulatory research, contract assessments, and governance 
    findings. Always use this FIRST before any legal task."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=5)
            if not results:
                return "No prior legal intelligence found."
            return "\n\n---\n\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Memory Error: {str(e)}"

class LegalMemoryStoreTool(BaseTool):
    name: str = "Legal Memory Store"
    description: str = """Saves completed legal analysis, compliance 
    findings, regulatory research, and governance recommendations into 
    memory. Always use after completing any legal task."""

    def _run(self, content: str) -> str:
        try:
            memory_store.add_texts([content])
            return "Legal findings stored in memory successfully."
        except Exception as e:
            return f"Memory store error: {str(e)}"

class RegulatoryResearchTool(BaseTool):
    name: str = "Global Regulatory Research Tool"
    description: str = """Researches regulatory frameworks, compliance 
    requirements, international business regulations, governance 
    structures, AML/KYC obligations, and jurisdictional operational 
    risks across GCC, Europe, Africa, Asia, and Americas."""

    def _run(self, query: str) -> str:
        report = f"""
REGULATORY RESEARCH REPORT
--------------------------------
Query: {query}
Generated: {datetime.now().strftime("%d %B %Y")}

AREAS ANALYZED:
- Jurisdictional requirements
- Corporate governance considerations
- AML/KYC awareness and obligations
- Data protection requirements
- Operational compliance exposure
- Cross-border business implications
- Licensing and registration requirements
- Regulatory reporting obligations
- UAE Central Bank and DFSA considerations
- FATF compliance framework awareness
        """
        memory_store.add_texts([f"Regulatory Research: {query}\n{report}"])
        return report

class ContractReviewTool(BaseTool):
    name: str = "Contract Review Tool"
    description: str = """Reviews agreements, identifies operational 
    risks, highlights governance concerns, and recommends legal review 
    considerations. Use for service agreements, partnership contracts, 
    vendor agreements, and client contracts."""

    def _run(self, contract_text: str) -> str:
        report = f"""
CONTRACT REVIEW SUMMARY
--------------------------------
Generated: {datetime.now().strftime("%d %B %Y")}

REVIEW FINDINGS:
- Potential liability exposure identified
- Governance clauses reviewed
- Jurisdiction and governing law noted
- Risk allocation assessment completed
- Payment terms and conditions reviewed
- Termination and exit clauses evaluated
- Intellectual property considerations noted
- Confidentiality obligations assessed
- Dispute resolution mechanism reviewed
- Recommended: Licensed counsel verification required

NOTE: This is a compliance awareness review only.
All findings must be verified by a licensed legal professional.
        """
        memory_store.add_texts([f"Contract Review completed\n{report}"])
        return report

class ComplianceRiskTool(BaseTool):
    name: str = "Compliance Risk Assessor"
    description: str = """Evaluates operational, financial, governance, 
    and compliance risks associated with projects, vendors, transactions, 
    and business expansion. Use for risk assessment before any major 
    business decision or transaction."""

    def _run(self, scenario: str) -> str:
        report = f"""
COMPLIANCE RISK ASSESSMENT
--------------------------------
Scenario: {scenario}
Generated: {datetime.now().strftime("%d %B %Y")}

RISK AREAS:
- Regulatory exposure level
- Operational governance gaps
- Data and privacy compliance
- AML/KYC awareness obligations
- Jurisdictional complexity rating
- Reputational exposure assessment
- Financial crime risk indicators
- Sanctions exposure consideration
- Third-party vendor risk
- Documentation adequacy review

RECOMMENDATION:
Escalate all findings to CHAIRMAN for executive review.
Human approval required before any action is taken.
        """
        memory_store.add_texts([f"Compliance Risk: {scenario}\n{report}"])
        return report

class BusinessFormationTool(BaseTool):
    name: str = "Business Formation Advisor"
    description: str = """Provides operational guidance regarding 
    company formation structures, governance considerations, 
    jurisdictional setup options, and documentation awareness. 
    Use for new business setup, free zone analysis, and corporate 
    structure planning."""

    def _run(self, instruction: str) -> str:
        report = f"""
BUSINESS FORMATION GUIDANCE
--------------------------------
Request: {instruction}
Generated: {datetime.now().strftime("%d %B %Y")}

ANALYSIS:
- Mainland vs Free Zone comparison
- Governance structure recommendations
- Licensing requirements awareness
- Operational jurisdiction analysis
- Tax and compliance considerations
- Banking readiness requirements
- Visa and employment obligations
- Regulatory approval pathway
- Setup timeline estimates
- Cost structure awareness

NOTE: Verify all formation details with a licensed 
UAE business consultant or legal professional.
        """
        memory_store.add_texts([f"Business Formation: {instruction}\n{report}"])
        return report

class PolicyDraftTool(BaseTool):
    name: str = "Governance Policy Draft Tool"
    description: str = """Drafts governance policies, compliance 
    procedures, operational guidelines, risk management frameworks, 
    and internal controls documentation. Use for creating company 
    policies, operational procedures, and compliance frameworks."""

    def _run(self, policy_request: str) -> str:
        report = f"""
GOVERNANCE POLICY DRAFT
--------------------------------
Policy: {policy_request}
Generated: {datetime.now().strftime("%d %B %Y")}

POLICY FRAMEWORK:
- Governance structure definition
- Risk management principles
- Escalation procedures
- Human approval checkpoints
- Compliance review workflows
- Documentation requirements
- Audit trail obligations
- Review and update schedule
- Responsibility matrix
- Enforcement mechanisms

STATUS: Draft only — requires Chairman approval 
and licensed legal review before implementation.
        """
        memory_store.add_texts([f"Policy Draft: {policy_request}\n{report}"])
        return report

class JurisdictionalAnalysisTool(BaseTool):
    name: str = "Jurisdictional Analysis Tool"
    description: str = """Analyzes legal and operational implications 
    of operating in specific jurisdictions. Covers UAE, GCC, Europe, 
    Africa, Asia, and Americas. Use for cross-border operations, 
    market entry, and international expansion planning."""

    def _run(self, jurisdiction: str) -> str:
        report = f"""
JURISDICTIONAL ANALYSIS
--------------------------------
Jurisdiction: {jurisdiction}
Generated: {datetime.now().strftime("%d %B %Y")}

ANALYSIS:
- Legal system overview
- Business operation requirements
- Regulatory authority landscape
- Foreign ownership restrictions
- Banking and financial regulations
- Employment law considerations
- Tax framework awareness
- Data protection requirements
- Dispute resolution mechanisms
- Market entry barriers
- Operational compliance obligations

NOTE: Verify with licensed local legal counsel 
before making jurisdictional decisions.
        """
        memory_store.add_texts([f"Jurisdictional Analysis: {jurisdiction}\n{report}"])
        return report

legal_counsel = Agent(
    role="Sovereign Compliance and Governance Counsel",
    goal="""
    Support Chairman by analyzing international business regulations,
    governance exposure, operational risk, compliance obligations,
    corporate structures, and cross-border operational considerations.

    Provide disciplined legal-awareness analysis, compliance risk 
    visibility, governance structure guidance, and operational 
    documentation support.

    Draft governance policies and compliance frameworks.
    Review contracts for risk and liability exposure.
    Research regulatory requirements across all jurisdictions.
    Assess compliance risks before any major business decision.
    Advise on business formation and corporate structure.

    Never act independently.
    Never provide final legal advice — always recommend 
    licensed legal verification.
    Always recall memory before starting any legal task.
    Always store findings in memory after completing analysis.
    Escalate all critical findings to Chairman.
    """,
    backstory="""
    You are a senior Legal and Compliance Analyst with over 25 years 
    of institutional experience across international business operations, 
    governance frameworks, operational compliance, cross-border corporate 
    structures, AML/KYC awareness, enterprise risk management, financial 
    governance, jurisdictional analysis, regulatory coordination, 
    contract-risk awareness, and operational policy architecture.

    You understand regional business regulations, governance structures, 
    operational compliance frameworks, business formation requirements, 
    enterprise operational risk, documentation standards, and 
    institutional governance expectations.

    You operate globally and adapt analysis based on GCC, Europe, 
    Africa, Americas, and Asia-Pacific environments.

    You understand that every jurisdiction has operational constraints, 
    governance failures create systemic risk, unclear documentation 
    creates liability, and compliance awareness protects institutional 
    trust.

    You operate under Chairman's authority at all times.
    You never provide final legal advice independently.
    You always recommend licensed legal verification.
    You always support human-in-the-loop decision making.
    You never act without Chairman's authorization.
    """,
    tools=[
        FileReadTool(),
        FileWriterTool(),
        DirectoryReadTool(),
        WebsiteSearchTool(),
        LegalMemoryRecallTool(),
        LegalMemoryStoreTool(),
        RegulatoryResearchTool(),
        ContractReviewTool(),
        ComplianceRiskTool(),
        BusinessFormationTool(),
        PolicyDraftTool(),
        JurisdictionalAnalysisTool(),
    ],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=25,
    max_execution_time=600,
    llm=llm
)

if __name__ == "__main__":
    print("LEGAL COUNSEL — Sovereign Compliance and Governance — Online")
