import logging

from agent_sdk.a2a.executor import StreamingAgentExecutor
from agents.agent import run_query, stream_for_a2a

logger = logging.getLogger("agent_news.a2a_executor")

class NewsAgentExecutor(StreamingAgentExecutor):
    """A2A executor that streams news agent responses chunk-by-chunk."""
    def __init__(self):
        super().__init__(run_query_fn=run_query, stream_fn=stream_for_a2a)
