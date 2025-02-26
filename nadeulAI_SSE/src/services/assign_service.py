import asyncio
from nadeulAI_SSE.src import schemas
from nadeulAI_SSE.src.components.preprocessor import Preprocessor
from nadeulAI_SSE.src.components.scheduler import Scheduler
from nadeulAI_SSE.src.confidential.constants import BASE_URL
from pathlib import Path
from fastapi import HTTPException


async def service(request: schemas.AssignRequest):
    current_file = Path(__file__).resolve()
    src_root = current_file.parents[1]
    vector_db_path = str(src_root / "database" / "vector_db")

    transformed_dto = Preprocessor.transform(request)

    hash_id = await Scheduler.scheduling(transformed_dto)
    url = f"http://{BASE_URL}/api/v1/sse?hash={hash_id}"
    return url


async def main():
    # 요청 생성
    request = schemas.AssignRequest(
        character=1,
        QA=["안녕하세요?", "반가워요", "잘지내요?"],
        mission_name="테스트용 미션 이름",
        submission_name="테스트용 서브 미션 이름",
        task_sequence=1,
    )

    # Scheduler 초기화 (비동기일 경우 await로 대기)
    await Scheduler.initialize()

    # 서비스 실행 및 hash_id 출력
    hash_id = await service(request)
    print(f"Hash ID: {hash_id}")


if __name__ == "__main__":
    asyncio.run(main())
