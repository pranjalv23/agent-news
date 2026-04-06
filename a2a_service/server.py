import logging

from agent_sdk.a2a.factory import create_a2a_app as _create

from .agent_card import NEWS_AGENT_CARD
from .executor import NewsAgentExecutor

logger = logging.getLogger("agent_news.a2a_server")


def create_a2a_app():
    """Build the A2A Starlette application for the news agent."""
    app = _create(NEWS_AGENT_CARD, NewsAgentExecutor, "agent_news")
    logger.info("A2A application created for News Agent")
    return app
