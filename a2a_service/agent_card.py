import os

from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill


NEWS_AGENT_CARD = AgentCard(
    name="News Agent",
    description=(
        "Personalized news analyst that delivers accurate, context-rich briefings on any topic. "
        "Covers world affairs, technology, finance, science, politics, sports, and more. "
        "Searches multiple sources, presents diverse perspectives, and explains why stories matter."
    ),
    supported_interfaces=[
        AgentInterface(
            url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9004"),
            protocol_binding="JSONRPC",
        )
    ],
    version="1.0.0",
    skills=[
        AgentSkill(
            id="news-briefing",
            name="News Briefing",
            description="Top stories and headlines on any topic or a general daily news digest.",
            tags=["news", "briefing", "headlines", "digest", "daily"],
        ),
        AgentSkill(
            id="topic-deep-dive",
            name="Topic Deep Dive",
            description="In-depth analysis of a specific news story or topic with multiple source perspectives.",
            tags=["news", "analysis", "investigation", "context", "explainer"],
        ),
        AgentSkill(
            id="source-monitoring",
            name="Source Monitoring",
            description="Track the latest developments on a specific topic across multiple news outlets.",
            tags=["news", "monitoring", "tracking", "updates"],
        ),
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
)
