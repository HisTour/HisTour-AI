import asyncio
import redis.asyncio as aioredis
import logging
import json
import grpc
from pathlib import Path
from fastapi import HTTPException
from httpx import AsyncClient, TimeoutException

from nadeulAI_SSE.src import schemas
from nadeulAI_SSE.src.constants.signals import START_SIGNAL
from nadeulAI_SSE.src.confidential.constants import (
    REDIS_HOST,
    REDIS_PORT,
    AI_SERVER_BASE_URL,
)
from nadeulAI_SSE.src.proto import llm_pb2, llm_pb2_grpc


async def service(hash: str):
    machine_idx = int(hash[:2])

    r_lb = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r_schedule = aioredis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True
    )
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    assigned_transformed_str = await r_schedule.get(hash)
    if not assigned_transformed_str:
        raise KeyError(f"Hash {hash} not found in Redis")

    assigned_transformed_dto = schemas.AssignTransformedDTO(
        **json.loads(assigned_transformed_str)
    )

    await r_schedule.delete(hash, assigned_transformed_str)
    asyncio.create_task(r_lb.set(f"ai_server_is_busy_{machine_idx}", 1, ex=40))

    try:
        channel = grpc.aio.insecure_channel(AI_SERVER_BASE_URL.format(str(machine_idx)))
        stub = llm_pb2_grpc.LLMServiceStub(channel)

        request = llm_pb2.GenerateRequest(
            qa=assigned_transformed_dto.QA,
            rag_results=assigned_transformed_dto.rag_results,
            top_k=assigned_transformed_dto.top_k,
            character_type=assigned_transformed_dto.character_type,
        )

        is_first = True
        print(request)
        async for response in stub.GenerateStream(request):
            if is_first:
                yield START_SIGNAL
                is_first = False

            result_text = response.text.replace("'", "").replace('"', "")
            result_text = result_text.replace("[말투반영]", "")

            model_output = schemas.Signal(
                type="model_output",
                contents=result_text,
                verbose="질문에 대한 모델 출력입니다.",
            )

            yield model_output

    except grpc.RpcError as e:
        print("gRPC 에러:", e)
        yield "No Response"

    except Exception as e:
        print("예외 발생:", e)
        yield "No Response"

    finally:
        await r_lb.delete(f"ai_server_is_busy_{machine_idx}")
        await r_schedule.close()
        await r_lb.close()
        if "channel" in locals():
            await channel.close()
