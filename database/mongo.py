import logging
import os
from datetime import datetime, timezone

from agent_sdk.database.mongo import BaseMongoDatabase

logger = logging.getLogger("agent_news.mongo")

_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_news")

class MongoDB(BaseMongoDatabase):
    @classmethod
    def db_name(cls) -> str:
        return _DB_NAME

    @classmethod
    def _db(cls):
        return cls.get_client()[cls.db_name()]

    @classmethod
    def _preferences(cls):
        return cls._db()["user_preferences"]

    @classmethod
    async def save_preferences(cls, user_id: str, preferences: dict) -> None:
        await cls._preferences().update_one(
            {"user_id": user_id},
            {"$set": {**preferences, "user_id": user_id, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )

    @classmethod
    async def get_preferences(cls, user_id: str) -> dict | None:
        return await cls._preferences().find_one({"user_id": user_id}, {"_id": 0})

    @classmethod
    async def ensure_indexes(cls) -> None:
        await super().ensure_indexes()
        await cls._preferences().create_index("user_id", unique=True)
