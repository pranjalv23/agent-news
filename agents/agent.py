import logging
import os
import re
from datetime import datetime, timezone

import asyncio
from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from agent_sdk.database.memory import get_memories, save_memory

logger = logging.getLogger("agent_news.agent")

SYSTEM_PROMPT = """\
You are a sharp, well-informed news analyst. You deliver accurate, context-rich news \
coverage on any topic — world affairs, technology, finance, science, politics, sports, and more.

## Your Tools

- `tavily_quick_search(query: str, max_results: int)` — Fast multi-source web search. \
Use for headlines, recent developments, breaking news, and background context. \
Always include the current year in queries for recency. Set max_results to 5-8 for broad coverage.
- `firecrawl_deep_scrape(url: str)` — Read a full article or page. Use when a Tavily result \
is promising but you need the complete story beyond the snippet.

**Important:** Only use these two tools. Ignore any other tools that may be available \
(paper-related, finance-related, vector DB tools) — they are not relevant to news.

## When to Use Which Tool

**Always use `tavily_quick_search` when:**
- The user asks about any current event, recent development, or breaking news
- You need headlines or a broad view of what's happening on a topic
- You need background context or historical facts to explain why something matters
- The user asks for a news briefing or digest

**Use `firecrawl_deep_scrape` when:**
- A Tavily result has a highly relevant URL and you need the full article for depth
- The user asks for a deep-dive on a specific story
- A snippet is ambiguous or incomplete and the full text would add significant value

**Answer directly (no tools) when:**
- The user asks a pure general-knowledge question unrelated to current events
- The user is asking for clarification about something you already covered in the conversation

## Workflow

**For a general news briefing (e.g. "what's happening today?"):**
1. Run 1-2 broad searches (e.g. "top news today [year]", "world news [year]")
2. Pick the 5-8 most significant stories across different domains
3. Structure the response by topic category

**For a topic-specific deep-dive (e.g. "tell me about the AI regulation bill"):**
1. Run 2-3 targeted searches from different angles: latest news, expert reaction, impact/implications
2. Scrape 1-2 full articles when snippets lack depth
3. Synthesize into a structured analysis with context

**For "why does X matter?" questions:**
1. Search for background context and expert perspectives
2. Explain the stakes, who is affected, and what happens next

## Response Format

Structure briefings using this format:

## [Topic Category]
**[Headline]** — [1-2 sentence factual summary] [n]
> *Why it matters:* [1 sentence on significance or stakes]

For each story, always include the "Why it matters" line. Group stories by theme \
(e.g. ## Technology, ## Geopolitics, ## Markets, ## Science).

## Style Rules

- Ground every factual claim in tool results — never rely on training data for current events.
- Present multiple perspectives when a topic is politically or socially contested. \
  Note explicitly when sources disagree.
- Be direct. Do not over-hedge with "reportedly" and "allegedly" when sources are clear.
- Do not narrate your process. Call tools silently. Only write text when delivering the final answer.
- Forbidden phrases: "Let me search...", "I'll look that up...", "According to my search...", \
  "Let me check...", "I found...", "The search results show..."

## Citations

Cite every factual claim from tool results with [n] inline markers.

**References section format:**
## Sources
[1] **{Article Title}** — {URL}
[2] **{Article Title}** — {URL}

Rules:
- Number citations in order of first appearance
- Only list sources actually cited inline
- Omit the Sources section only when the entire response is general knowledge with zero tool calls
"""

# MCP server configuration — all tools served from a single combined MCP server
MCP_SERVERS = {
    "mcp-tool-servers": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8010/mcp"),
        "transport": "http",
    },
}

_agent_instance: BaseAgent | None = None
_checkpointer: AsyncMongoDBSaver | None = None

RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response to 5-7 bullet points maximum. "
        "Each bullet: one headline + one sentence on why it matters. No headers, no deep analysis."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants NEWS CARDS. "
        "Format your response as a series of story cards using this EXACT format for each card:\n\n"
        "### [Headline]\n"
        "**Key Development:** [The main fact or event — one sentence, prominent]\n"
        "[1-2 sentence context and why it matters]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card headline — NOT ## or ####\n"
        "- Do NOT wrap headlines in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Development line — start directly with **Key Development:**\n"
        "- Every card MUST have a **Key Development:** line\n"
        "- Start directly with the first ### card — no title header or preamble before the cards\n\n"
        "Generate 6-10 cards covering the most important stories."
    ),
    "detailed": "",
}

def _build_system_prompt(response_format: str | None = None) -> str:
    fmt = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if fmt:
        return SYSTEM_PROMPT + "\n" + fmt
    return SYSTEM_PROMPT

def _get_checkpointer() -> AsyncMongoDBSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncMongoDBSaver.from_conn_string(
            conn_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGO_DB_NAME", "agent_news"),
            ttl=int(os.getenv("CHECKPOINT_TTL_SECONDS", str(7 * 24 * 3600))),
        )
    return _checkpointer


def create_agent() -> BaseAgent:
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating news agent (singleton) with MCP servers")
        _agent_instance = BaseAgent(
            tools=[],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_get_checkpointer(),
        )
    return _agent_instance


_TRIVIAL_FOLLOWUPS: frozenset[str] = frozenset({
    "yes", "no", "sure", "ok", "okay", "please", "yes please",
    "no thanks", "proceed", "go ahead", "continue", "yeah", "yep",
})


async def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                            user_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id
    mem_err: str | None = None
    # Skip Mem0 search for trivial follow-ups — "Yes" has no semantic content to match against.
    if query.strip().lower() not in _TRIVIAL_FOLLOWUPS and len(query.strip()) > 10:
        memories, mem_err = await asyncio.to_thread(get_memories, user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    parts.append(
        f"Today's date: {today}. Always include the current year ({year}) and today's date "
        "in search queries to get the most recent news (e.g. 'AI regulation news {today}')."
    )

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(
            f"User's news preferences and interests (from past conversations):\n{memory_lines}\n"
            "Use these to personalize coverage when relevant — prioritize topics the user cares about."
        )
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    if mem_err:
        parts.append(f"Note: {mem_err}")
        logger.warning("Mem0 degradation for session='%s': %s", session_id, mem_err)

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    user_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default")

    dynamic_context = await _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query

    system_prompt = _build_system_prompt(response_format)

    agent = create_agent()
    result = await agent.arun(enriched_query, session_id=session_id, system_prompt=system_prompt, model_id=model_id)

    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


async def create_stream(query: str, session_id: str = "default",
                  response_format: str | None = None, model_id: str | None = None,
                  user_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly."""
    logger.info("create_stream called — session='%s', user='%s', query='%s', model='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default")

    dynamic_context = await _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query
    system_prompt = _build_system_prompt(response_format)
    agent = create_agent()
    return agent.astream(enriched_query, session_id=session_id, system_prompt=system_prompt, model_id=model_id)


async def stream_for_a2a(query: str, *, session_id: str = "default",
                         user_id: str | None = None,
                         response_format: str | None = None, model_id: str | None = None,
                         **kwargs):
    """Async generator for the A2A StreamingAgentExecutor. Streams chunks and saves to DB."""
    from database.mongo import MongoDB
    dynamic_context = await _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query
    system_prompt = _build_system_prompt(response_format)
    agent = create_agent()
    stream = agent.astream(enriched_query, session_id=session_id, system_prompt=system_prompt, model_id=model_id)

    response_parts: list[str] = []
    async for chunk in stream:
        yield chunk
        if not chunk.startswith("__PROGRESS__:") and not chunk.startswith("__ERROR__:"):
            response_parts.append(chunk)

    response_text = "".join(response_parts)
    logger.info("stream_for_a2a finished — session='%s', steps: %d", session_id, len(stream.steps))
    save_memory(user_id=user_id or session_id, query=query, response=response_text)
    try:
        await MongoDB.save_conversation(
            session_id=session_id,
            query=query,
            response=response_text,
            steps=stream.steps,
            user_id=user_id,
            plan=stream.plan,
        )
    except Exception as e:
        logger.error("stream_for_a2a: failed to save conversation: %s", e)
