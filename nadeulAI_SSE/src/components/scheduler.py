import redis.asyncio as aioredis
import logging
from nadeulAI_SSE.src.confidential.constants import (
    REDIS_HOST,
    REDIS_PORT,
    AI_SERVER_COUNT,
)
from nadeulAI_SSE.src import schemas
import uuid
import json
import asyncio


class Scheduler:
    lock = asyncio.Lock()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    @staticmethod
    async def get_redis_connection():
        return aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )

    @staticmethod
    async def get_schedule_connection():
        return aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True
        )

    @staticmethod
    async def initialize() -> None:
        r_lb = await Scheduler.get_redis_connection()
        r_schedule = await Scheduler.get_schedule_connection()
        await r_lb.flushdb()
        await r_schedule.flushdb()
        await r_lb.set("current_ai_server_idx", -1)
        await r_lb.close()
        await r_schedule.close()

    @staticmethod
    async def scheduling(assigned_transformed_dto: schemas.AssignTransformedDTO) -> str:
        async with Scheduler.lock:
            r_lb = await Scheduler.get_redis_connection()
            r_schedule = await Scheduler.get_schedule_connection()
            idx = 0
            busy_log_flag = False
            while True:
                current_ai_server_idx = int(await r_lb.get("current_ai_server_idx"))
                current_ai_server_idx = (current_ai_server_idx + 1) % AI_SERVER_COUNT
                await r_lb.set("current_ai_server_idx", current_ai_server_idx)

                if (
                    await r_lb.get(f"ai_server_is_busy_{current_ai_server_idx}")
                    is not None
                ):
                    pass
                else:
                    hash_id = Scheduler.make_hash(
                        current_ai_server_idx, assigned_transformed_dto.character_type
                    )
                    await r_schedule.set(
                        hash_id,
                        json.dumps(
                            assigned_transformed_dto.model_dump(), ensure_ascii=False
                        ),
                        ex=19,
                    )
                    await r_lb.set(
                        f"ai_server_is_busy_{current_ai_server_idx}", 1, ex=20
                    )
                    print(current_ai_server_idx)
                    await r_lb.close()
                    await r_schedule.close()
                    return hash_id
                idx += 1
                if idx >= AI_SERVER_COUNT and not busy_log_flag:
                    Scheduler.logger.warning("AI Servers are busy")
                    busy_log_flag = True

    @staticmethod
    def make_hash(assigned_machine: int, character_type: int) -> str:
        random_uuid = uuid.uuid4().hex
        hash_id = f"{str(assigned_machine).zfill(2)}{random_uuid[:12]}{character_type}"
        return hash_id
