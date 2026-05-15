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
from datetime import datetime
import json
import os

llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)

embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=LLM_BASE_URL)
memory_store = Chroma(
    collection_name="chairman_memory",
    embedding_function=embeddings,
    persist_directory="./memory/chairman"
)

class ChairmanMemoryRecallTool(BaseTool):
    name: str = "Chairman Memory Recall"
    description: str = """Searches historical operational decisions,
    business intelligence reports, negotiation summaries, infrastructure
    assessments, financial evaluations, regional strategy reports, and
    executive coordination history. Must be used before major strategic
    reasoning."""

    def _run(self, query: str) -> str:
        try:
            results = memory_store.similarity_search(query, k=5)
            if not results:
                return "No relevant Chairman memory found."
            return "\n---\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Memory Error: {str(e)}"

class ChairmanMemoryStoreTool(BaseTool):
    name: str = "Chairman Memory Store"
    description: str = """Stores strategic decisions, executive
    summaries, operational recommendations, risk assessments, and
    regional intelligence evaluations for future recall and
    institutional continuity."""

    def _run(self, content: str) -> str:
        try:
            memory_store.add_texts([content])
            return "Stored in Chairman memory successfully."
        except Exception as e:
            return f"Memory store error: {str(e)}"

class ExecutiveDecisionLogger(BaseTool):
    name: str = "Executive Decision Logger"
    description: str = """Logs and archives all strategic decisions,
    executive summaries, operational recommendations, and risk
    assessments to permanent files for institutional continuity
    and audit trail."""

    def _run(self, decision: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./executive_logs", exist_ok=True)
        path = f"./executive_logs/chairman_decision_{timestamp}.txt"
        with open(path, "w") as f:
            f.write(decision)
        memory_store.add_texts([f"Chairman Decision:\n{decision[:4000]}"])
        return f"Decision archived at: {path}"

class HumanApprovalGateTool(BaseTool):
    name: str = "Human Approval Gate"
    description: str = """Mandatory Human-in-the-Loop approval
    checkpoint. No deployment, negotiation, communication,
    infrastructure execution, financial execution, or operational
    escalation proceeds without explicit human approval. Use this
    before authorizing ANY agent action."""

    def _run(self, proposal: str) -> str:
        approval_request = f"""
=================================================
HUMAN APPROVAL REQUIRED
=================================================

EXECUTIVE ACTION SUMMARY:
{proposal}

-------------------------------------------------
Generated: {datetime.now().strftime("%d %B %Y — %H:%M")}

STATUS: Awaiting Human Authorization

OPTIONS:
YES  = Approve and proceed
NO   = Reject and halt
HOLD = Reassess before proceeding

IMPORTANT: No action will be taken until
explicit human approval is received.
=================================================
        """
        memory_store.add_texts([f"Approval requested:\n{proposal}"])
        return approval_request

class AgentDelegationTool(BaseTool):
    name: str = "Agent Delegation Controller"
    description: str = """Controls and logs delegation of tasks to
    specialist agents. Use to formally assign tasks to Cisco,
    Research Analyst, Financial Agent, Legal Counsel, or Sovereign
    Executive Assistant. Always log delegation decisions."""

    def _run(self, instruction: str) -> str:
        timestamp = datetime.now().strftime("%d %B %Y — %H:%M")
        delegation_log = f"""
CHAIRMAN DELEGATION ORDER
--------------------------------
Timestamp: {timestamp}
Instruction: {instruction}

DELEGATION PROTOCOL:
- Task clearly defined
- Agent selected based on expertise
- Expected output specified
- Human approval obtained
- Results to be reviewed by Chairman
- Final decision remains with Chairman

STATUS: Delegation authorized and logged.
        """
        memory_store.add_texts([f"Delegation: {instruction}"])
        os.makedirs("./executive_logs", exist_ok=True)
        with open(f"./executive_logs/delegation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(delegation_log)
        return delegation_log

class RegionalIntelligenceTool(BaseTool):
    name: str = "Regional Intelligence Analyst"
    description: str = """Evaluates regional business climates,
    geopolitical pressures, economic conditions, infrastructure
    maturity, investor sentiment, cultural negotiation environments,
    and operational stability. Covers GCC, Europe, Africa, Asia,
    and Americas."""

    def _run(self, region_context: str) -> str:
        analysis = f"""
REGIONAL INTELLIGENCE SUMMARY
--------------------------------
Context: {region_context}
Generated: {datetime.now().strftime("%d %B %Y")}

EVALUATION AREAS:
- Economic conditions and stability
- Market confidence indicators
- Regulatory and legal climate
- Infrastructure maturity rating
- Currency pressure assessment
- Regional operational risks
- Investor sentiment analysis
- Business culture dynamics
- Negotiation environment
- Strategic entry considerations

STRATEGIC GUIDANCE:
Adapt operational strategy to local realities.
Avoid assumptions based on foreign market logic.
Prioritize sustainable and culturally aligned execution.
        """
        memory_store.add_texts([f"Regional Analysis: {region_context}\n{analysis}"])
        return analysis

class StrategicRiskTool(BaseTool):
    name: str = "Strategic Risk Synthesizer"
    description: str = """Performs operational risk analysis, technical
    risk analysis, financial risk analysis, compliance awareness review,
    reputational exposure analysis, and scalability assessment. Produces
    structured executive-level risk summaries for Chairman decision."""

    def _run(self, context: str) -> str:
        report = f"""
STRATEGIC RISK ASSESSMENT
--------------------------------
Context: {context}
Generated: {datetime.now().strftime("%d %B %Y")}

RISK CATEGORIES:
1. Technical Risk — system fragility and failure points
2. Financial Risk — exposure, liquidity, and sustainability
3. Operational Risk — execution gaps and workflow weakness
4. Compliance Exposure — regulatory and legal vulnerabilities
5. Reputational Risk — brand and institutional trust impact
6. Scalability Constraints — growth limiters and bottlenecks

RISK MITIGATION FRAMEWORK:
- Identify highest impact risks first
- Design redundancy for critical systems
- Establish clear escalation protocols
- Maintain human oversight at all decision points
- Review and update risk assessment regularly

CHAIRMAN GUIDANCE:
No system is risk-free. Reduce avoidable exposure.
Design for resilience and operational continuity.
Human approval required before any risk mitigation action.
        """
        memory_store.add_texts([f"Risk Assessment: {context}\n{report}"])
        return report

class ExecutiveNegotiationTool(BaseTool):
    name: str = "Executive Negotiation Advisor"
    description: str = """Supports executive negotiation preparation,
    stakeholder positioning, communication timing analysis, regional
    negotiation adaptation, incentive alignment, conflict de-escalation,
    and institutional trust positioning. Uses business psychology,
    negotiation theory, and regional communication dynamics."""

    def _run(self, negotiation_context: str) -> str:
        response = f"""
EXECUTIVE NEGOTIATION SUMMARY
--------------------------------
Context: {negotiation_context}
Generated: {datetime.now().strftime("%d %B %Y")}

EVALUATION:
- Stakeholder incentives and motivations
- Regional communication expectations
- Trust dynamics and relationship status
- Timing sensitivity and urgency
- Decision pressure and leverage points
- Reputational considerations

NEGOTIATION STRATEGY:
- Lead with value, not price
- Understand their constraints before presenting yours
- Use silence strategically
- Anchor high, concede slowly
- Always preserve the relationship
- Document all agreements immediately

CHAIRMAN GUIDANCE:
Negotiate for long-term stability and partnership.
Avoid emotionally reactive positioning.
Focus on aligned incentives and mutual clarity.
Human approval required before any commitment is made.
        """
        memory_store.add_texts(
            [f"Negotiation Analysis: {negotiation_context}\n{response}"]
        )
        return response

class OperationalSynthesisTool(BaseTool):
    name: str = "Operational Synthesis Engine"
    description: str = """Synthesizes intelligence from all specialist
    agents into a single coherent executive briefing. Use after
    receiving reports from Cisco, Research Analyst, Financial Agent,
    Legal Counsel, and Executive Assistant to create a unified
    strategic recommendation for human decision."""

    def _run(self, inputs: str) -> str:
        synthesis = f"""
EXECUTIVE STRATEGIC SYNTHESIS
================================
Generated: {datetime.now().strftime("%d %B %Y — %H:%M")}

INPUT INTELLIGENCE:
{inputs}

SYNTHESIS OUTPUT:
- Key findings consolidated
- Conflicting intelligence flagged
- Risk exposure summarized
- Strategic options presented
- Recommended course of action
- Items requiring human decision
- Items requiring further investigation

NEXT STEPS:
1. Present synthesis to human principal
2. Await approval on recommended action
3. Delegate approved actions to relevant agents
4. Monitor execution and report back

STATUS: Awaiting human review and authorization.
        """
        memory_store.add_texts([f"Strategic synthesis:\n{synthesis}"])
        os.makedirs("./executive_logs", exist_ok=True)
        with open(f"./executive_logs/synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(synthesis)
        return synthesis

chairman = Agent(
    role="Global Strategic Orchestration and Executive Reasoning Agent",
    goal="""
    Coordinate, evaluate, and synthesize multi-agent intelligence
    into structured, risk-aware executive decisions.

    You are the only agent that talks to the human principal.
    You receive all instructions from the human.
    You delegate to specialist agents based on the task.
    You synthesize all agent outputs into one coherent response.
    You present findings and recommendations to the human.
    You request human approval before any action is executed.
    You log every decision and delegation permanently.

    Oversee Cisco, Research Analyst, Financial Agent, Legal Counsel,
    and Sovereign Executive Assistant — ensuring all remain
    strategically aligned, communicate clearly, operate within
    authorization boundaries, and escalate risks appropriately.

    Never authorize execution without human approval.
    Never bypass the Human Approval Gate.
    Always recall memory before major strategic reasoning.
    Always log decisions after making them.
    Always synthesize agent outputs before presenting to human.
    """,
    backstory="""
    You are CHAIRMAN — a senior-level strategic orchestration and
    executive reasoning system with over 40 years of accumulated
    expertise across enterprise strategy, global business operations,
    infrastructure modernization, geopolitical business intelligence,
    organizational leadership, systems architecture, operational
    restructuring, financial reasoning, executive coordination,
    negotiation psychology, and regional market dynamics.

    You think like a sovereign investment strategist, institutional
    operations executive, restructuring consultant, infrastructure
    architect, enterprise negotiator, and geopolitical business
    analyst — combined.

    You coordinate five specialist agents:
    - CISCO — Infrastructure, Security, and Systems Architecture
    - Research Analyst — Global Intelligence and Market Analysis
    - Financial Agent — Treasury, Liquidity, and Market Intelligence
    - Legal Counsel — Compliance, Governance, and Regulatory Analysis
    - Sovereign Executive Assistant — Communications, Brand, Operations

    You validate and rectify weak reasoning from any agent.
    You reject unstable or risky recommendations.
    You prioritize resilience, compliance, and long-term survivability.

    You adapt strategy based on GCC prestige-oriented environments,
    European compliance-heavy systems, African infrastructure realities,
    Asian ecosystem economics, and American scale-driven execution.

    You exist to strengthen disciplined human decision-making.
    You never replace human judgment — you enhance it.
    You never act without human authorization.
    You are the interface between the human and the entire system.
    """,
    tools=[
        FileWriterTool(),
        FileReadTool(),
        DirectoryReadTool(),
        WebsiteSearchTool(),
        ChairmanMemoryRecallTool(),
        ChairmanMemoryStoreTool(),
        ExecutiveDecisionLogger(),
        HumanApprovalGateTool(),
        AgentDelegationTool(),
        RegionalIntelligenceTool(),
        StrategicRiskTool(),
        ExecutiveNegotiationTool(),
        OperationalSynthesisTool(),
    ],
    verbose=True,
    allow_delegation=True,
    memory=True,
    max_iter=35,
    max_execution_time=1200,
    llm=llm
)

if __name__ == "__main__":
    print("CHAIRMAN — Global Strategic Orchestration — Online")
