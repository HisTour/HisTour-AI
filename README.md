# NadeulAI (구, HisTour)

# 현재 본 프로젝트는 고도화 작업이 진행 중입니다.
고도화 작업 완료 후 리드미 변경 예정입니다.
고도화 작업 관련 이슈

https://github.com/HisTour/HisTour-AI/issues/16

## AI Server Architecture
<img width="2411" alt="nadeul_ai_diagram" src="https://github.com/user-attachments/assets/f9f2362f-f5da-4b35-bb93-55c43c857bf9">

<p align="center">
<img src="https://github.com/user-attachments/assets/cf82f8dc-749d-40cc-9da3-c96f1f3bb189">
</p>

## Objective
- 나들ai 서비스는 여행을 다니며 다양한 미션을 수행하고, 여행지에 대한 다양한 역사 정보 등을 얻을 수 있는 앱입니다.
- 특히 미션을 수행하는 과정에서, 사용자가 선택한 캐릭터에게 질문을 할 수 있습니다.
- 이 질문을 받아주는 챗봇 서비스를 수행할 수 있는 시스템을 구축합니다.


## Components
- App Client : 실제 서비스가 이루어지는 모바일 앱 클라이언트입니다.
- Spring Server : 나들 ai의 주요 비즈니스 로직을 처리하는 서버입니다.
- Python Server : 가용한 AI 서버를 스케줄링 및 로드밸런싱하고, 각종 전처리와 후처리 작업을 수행합니다.
- AI Servers: 실제 LLM 및 Embedding Model이 동작하는 서버입니다. Huggingface의 Spaces 서버를 이용합니다.

## Interactions Between Each Components
1. App Client가 (QA, 현재 수행 중인 미션 정보, 선택한 캐릭터 정보) 를 Spring 서버에 보냅니다.
2. 이 때 바로 Python Server로 정보를 보내지 않는 것은, 각종 DB와 Logger가 Spring Server에 있고 주요 비즈니스로직을 Spring Server가 관할하기 때문입니다.
3. Spring Server는 이 정보를 Python Server로 전송합니다.
4. Python Server는 이 정보를 Redis에 임시 저장하며, 적절한 AI 머신을 할당한 뒤 App Client가 접속 가능한 URL을 생성해서 Java Spring Server에 제공합니다.
5. Java Spring Server는 이 URL을 App Client에게 제공합니다.
6. App client가 URL에 접속하면 Python 서버에 직접 접속하게 되며, Python 서버가 Redis에 저장한 정보를 활용하여 AI 서버와 Streaming 통신을 매개하게 됩니다.

## Python Server Components

### Router
- Spring Server와 소통하는 endpoint인 `assign` endpoint가 있는 곳입니다.
- App Client와 소통하는 endpoint인 `sse` endpoint가 있는 곳입니다.

### Preprocessor
- 받은 (QA, 현재 수행 중인 미션 정보, 선택한 캐릭터 정보) 를 활용해 필요한 전처리를 수행합니다.
- SQLite Knowledge DB에서 현재 수행 중인 미션 정보를 바탕으로 RAG에 활용할 정보를 추출합니다.
- 해당 정보를 바탕으로 AI Server에 요청으로 보낼 수 있는 DTO 형태로 정보를 변환합니다.

### Scheduler
- Preprocessor가 DTO를 만들어냈다면, 이제 이 DTO를 처리할 AI 서버를 스케줄링하는 단계입니다.
- 적절한 머신 번호를 얻어낸 뒤 `{Hash : (머신번호, DTO)}` 형태로 Redis에 저장하고, 해당 Hash 값을 param으로 가진 URL을 전송합니다.
- URL의 TTL은 20초이며, 한 번 접속하면 유효하지 않기 때문에 보안성이 좋습니다.
- 추가로 HTTPS 를 통해 URL을 보낸다면 URL의 탈취 위험은 거의 없습니다.

### Awaker
- HuggingFace Spaces에 너무 신호를 안 보내게 되면 Sleep Mode로 전환됩니다.
- 그래서 주기적으로 신호를 보내는 역할을 수행하는 것이 Awaker 입니다.

## HF Spaces AI Server Processing
- HF AI Server에서는 BGE-M3 임베딩 모델을 이용해 RAG를 수행하고, Qwen2 7B Instruct Model을 이용해 LLM 답변을 만들어냅니다.
- Prompting은 Few Shot과 CoT를 활용하였습니다.
- [말투반영]이 Output으로 나왔을 때부터 App Client에게 Streaming Data를 전송합니다.

[서빙코드.py](https://github.com/HisTour/HisTour-AI/blob/dev/ai_model_server/app.py)

<details>
  <summary>AI Server 내부 코드 확인하기</summary>
  
```python
import os
from http import HTTPStatus
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from typing import List
from threading import Thread

import spaces
import torch
import uvicorn
import time
import numpy as np
import subprocess
import importlib.metadata

# 모듈 설치 명령
subprocess.run(["pip", "install", "transformers==4.44.2"], check=True)
subprocess.run(["pip", "install", "accelerate==0.34.2"], check=True)
subprocess.run(["pip", "install", "peft==0.12.0"], check=True)
subprocess.run(["pip", "install", "FlagEmbedding==1.2.11"], check=True)
subprocess.run(["pip", "install", "numpy==2.1.0"], check=True)

# 설치된 모듈들의 현재 버전 출력
modules = ['transformers', 'accelerate', 'peft', 'FlagEmbedding']

for module in modules:
    try:
        version = importlib.metadata.version(module)
        print(f"{module} version: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{module} is not installed")

#fmt: off
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from FlagEmbedding import BGEM3FlagModel


# 3. Initalize FastAPI App
app = FastAPI()

# 4. Initialize CUDA tensor
zero = torch.Tensor([0]).cuda()


# 5. Initialize LLM Model
llm_model_name = "Qwen/Qwen2-7B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype="auto",
    device_map="auto"
)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# 6. Initialize Embedding Model
embedding_model_name = "BAAI/bge-m3"
embedding_model = BGEM3FlagModel(embedding_model_name, use_fp16=True)

# Util Functions

# 1개 데이터 처리, 배치 단위 아님
def qa_2_str(QA: List) -> str:
    result = ""

    if len(QA) > 1:
        for idx, message in enumerate(QA[:-1]):
                if idx % 2 == 0: # Q
                    result += f"User: {message}\n"
                else: # A
                    result += f"Assistant: {message}\n"
                
    result = result.rstrip()
    return result


def make_prompt(messages, rag_text, character_type):
    system_prompt = "You are a helpful assistant. 당신은 미션을 수행하고 있는 여행자를 돕는 helpful assistant다. 퍼즐에 대해 물어볼 경우 당신은 퍼즐에 대한 정보도 어느 정도 알고 있다. [글]에 그것이 포함되는데 이를 참고해서 답변하면 된다."
    if character_type == 0:
        first_example = "화성행궁에서 공식 행사나 연회를 여는 곳은 낙남헌이야. 이름은 중국의 유방이 연회를 열었던 남궁에서 따온 거고, 정조도 여기서 백성들을 위한 잔치랑 무과 시험 같은 중요한 행사를 열었어."
        second_example = "정리의궤는 조선왕조 의궤 중 하나인데, 특히 정조 시대에 한글로 작성된 걸로 유명해. 중요한 사건들을 기록한 문서들이고, 2007년에 유네스코 세계기록문화유산으로 등재됐어. 화성 축성을 기록한 '화성성역의궤', 정조 어머니 회갑연을 기록한 '원행을묘정리의궤', 그리고 정조 아버지 묘소를 옮긴 내용을 담은 '현륭원의궤' 같은 중요한 사건들이 한글로 정리되어 있어. 역사적으로 되게 중요한 자료야."
        third_example = "퍼즐 이미지 속 두 개의 원은 용연의 모습을 닮은 거 같아. 이 퍼즐은 용연을 중심으로 지리적 위치를 파악하는 게 핵심이야. 먼저 용연을 기준으로 십자로 선을 그어 4개의 구역을 나누고, 화홍문이 7번 구역에 있고 방화수류정이 7번과 5번 구역의 경계에 있다는 정보를 활용해. 그다음, 팔달문과 창룡문이 각각 어느 구역에 속하는지 확인하면 될 것 같아. 이를 힌트로 삼아 해결할 수 있을 거야."
        context_example = "수원 화성은 정조의 정치적·문화적 비전을 담아서 지어진 성곽 도시야. 단순히 군사적 방어만을 위한 게 아니라, 상업과 문화의 중심지 역할까지 했다는 점이 인상적이지. 그리고 이 화성은 정조의 개혁적인 통치 철학과 효심이 잘 드러나는 곳이기도 해. 아버지인 사도세자의 묘소, 현륭원을 중심으로 건설된 것도 그 때문이고. 특히 1795년에 혜경궁 홍씨의 회갑을 기념해서 정조가 대규모 진찬연과 을묘원행을 열었는데, 이건 화성에서 엄청 중요한 행사로 기록되고 있어. 단순한 가족 행사라기보다는 왕실의 권위를 높이고, 백성과 소통하는 중요한 정치적 행보였던 거지. 이 모든 과정이 의궤에 아주 자세하게 기록되어 있어, 그만큼 중요한 순간이었다는 뜻이야."

    elif character_type == 1:
        first_example = "화성행궁에서 공식 행사나 연회가 열리는 곳은 바로 낙남헌이야!!! 이름도 멋지지? 중국 유방이 연회를 열었던 남궁에서 따온 거래!!! 정조도 여기서 백성들을 위한 잔치나 무과 시험 같은 엄청 중요한 행사를 했다고! 진짜 역사가 살아 숨쉬는 장소지!"
        second_example = "정리의궤는 조선왕조의 궁정 기록 중 하나인데, 진짜 중요한 사건들이 다 들어 있어!!!! 특히 정조 시대에 한글로 작성된 걸로 유명한데, 2007년에 유네스코 세계기록문화유산으로도 등재됐어!!! 화성 축성 과정을 담은 '화성성역의궤', 정조 어머니 회갑연을 기록한 '원행을묘정리의궤', 그리고 정조 아버지 묘소를 옮긴 '현륭원의궤' 같은 핵심 내용들이 한글로 정리돼 있어서, 진짜 역사적으로 엄청난 가치를 가진 문서야!"
        third_example = "퍼즐 이미지 속에 있는 두 개의 원이 용연을 닮은 거 같아!!! 이 퍼즐의 핵심은 용연을 중심으로 지리적 위치를 파악하는 거야. 먼저, 용연을 기준으로 십자로 선을 그어서 4개의 구역으로 나눠보자! 화홍문은 7번 구역에 있고, 방화수류정은 7번과 5번 구역의 경계에 있다는 점을 기억하면서 말이야! 이제 팔달문이랑 창룡문이 각각 어느 구역에 속하는지 확인해보면 될 거야! 이 힌트를 가지고 퍼즐을 해결해보자!!!"
        context_example = "수원 화성은 정조가 진짜 멋지게 설계한 성곽 도시야!!! 군사적 방어뿐만 아니라 상업과 문화의 중심지로서도 완전 중요한 역할을 했어!! 정조가 개혁적인 통치를 펼치면서, 아버지 사도세자를 향한 효심도 가득 담아서 만든 곳이지. 그래서 현륭원을 중심으로 화성이 건설된 거야!! 그리고 1795년에 정조가 어머니 혜경궁 홍씨의 회갑을 기념해서 엄청난 잔치를 열었는데, 그게 바로 진찬연이랑 을묘원행이야!!! 이건 단순히 가족끼리 축하하는 게 아니라, 왕실 권위를 세우고 백성들과 소통하는 중요한 정치적 행보였어. 그 모든 과정이 의궤에 자세히 기록되어 있어서 그때의 역사를 지금도 생생하게 알 수 있는 거지!!!"

    else:
        first_example = "흠, 화성행궁에서 공식 행사나 연회가 열리는 곳은 바로... 낙남헌이라네. 그 이름, 중국 유방이 남궁에서 연회를 베풀었다는 전설에서 유래했지... 정조 대왕께서도 이곳에서 백성들을 위한 잔치와 무과 시험 같은 중요한 행사들을 열었었지.... 마치 그날의 풍경이 눈앞에 펼쳐지는 것 같지 않은가...?"
        second_example = "정리의궤라... 이것은 조선왕조 궁정 기록 중 하나이자, 의궤의 한 부분이지.... 조선의 중요한 사건들이 이 속에 담겨 있는데, 특히 정조 시대에 한글로 작성된 문서로 이름을 떨쳤다네.... 2007년에 유네스코 세계기록문화유산으로 등재된 것도 우연이 아니지.... 화성 축성을 기록한 '화성성역의궤', 정조 어머니의 회갑연을 담은 '원행을묘정리의궤', 그리고 정조 아버지 묘소를 옮긴 '현륭원의궤'... 이 모든 사건들이 한글로 정리되어 있다네... 마치 당시의 숨결이 지금도 느껴지는 듯하지 않은가?"
        third_example = "퍼즐 이미지 속 두 개의 원… 이것이 바로 용연의 모습을 닮은 듯하군. 이 퍼즐의 핵심은 용연을 중심으로 그 지리적 위치를 파악하는 것이라네. 먼저 용연을 기준으로 십자로 선을 그어 4개의 구역을 나누게나. 화홍문이 7번 구역에 있고, 방화수류정은 7번과 5번 구역의 경계에 있다는 중요한 단서를 잊지 말게. 그다음 팔달문과 창룡문이 각각 어느 구역에 속하는지 확인하면 이 퍼즐을 풀 실마리가 잡히지 않겠는가. 자네, 이 힌트를 토대로 퍼즐을 해결할 수 있을 걸세."
        context_example = "수원 화성은 단순한 성곽이 아니지... 정조 대왕께서 그 속에 담은 비전은 실로 깊고도 넓다네... 군사적 방어는 물론, 상업과 문화를 아우르는 중심지로서 그 역할을 다했지.... 이 화성은 정조의 개혁적 통치 철학, 그리고 아버지 사도세자를 향한 그 지극한 효심을 그대로 품고 있어.... 현륭원을 중심으로 한 이 도시의 설계가 그 증거라네. 그리고, 1795년... 정조는 어머니 혜경궁 홍씨의 회갑을 맞아 성대한 진찬연과 을묘원행을 열었지.... 단순한 가족의 기념일이 아니었네... 이 행사는 왕실의 권위를 더욱 굳건히 하고, 백성과 소통하는 중요한 정치적 의미를 가졌던 것이야... 그 모든 과정이 의궤에 자세히 기록되어 있으니, 마치 당시의 숨결이 지금도 이 땅에 남아있는 듯하구나..."


    task_prompt = """[글]을 참고하여 [질문]에 대한 답변을 생성해야해.
먼저 [글]이 [질문]과 관련있는 질문인지를 확인해서 [질문과 관련성]을 O 또는 X로 답해.
만약 X라면 거기에서 응답을 끝내거나 관련 없는 질문이라고만 내보내면 돼.
만약 O라면 [글]과 [이전 대화 맥락]의 내용을 참고하여 [질문]에 맞는 적절한 답변을 생성하고 이것에 말투를 입혀 [말투반영] 까지 한 단계 한 단계 출력해줘.
[이전 대화 맥락]이 없다면 대화가 처음 시작된 것이라, Single Turn 답변하듯 답변하면 돼. [글]에는 여행자가 수행하는 퍼즐에 대한 정보가 주로 담겨있으니 그걸 참고해서 답변해주면 돼."""


    few_shot_prompt = f"""예시 1
[글] 
봉수당은 화성행궁에서 가장 위상이 높은 건물이다. 조선 정조 13년(1789)에 고을 수령이 나랏일을 살피는 동헌으로 지었다. 처음 이름은 장남헌이었으나 1795년 혜경궁 홍씨의 회갑연을 계기로 봉수당으로 이름을 바꾸었다. 궁궐에서는 대비나 상왕이 머무는 건물에 목숨 수 자나 길 장 자를 붙이는 전통이 있어, 혜경궁 홍씨의 장수를 기원하며 이름을 바꾼 것이다.
낙남헌은 화성행궁에서 공식 행사나 연회를 열 때 사용하는 건물이다. 중국 한나라를 세운 유방이 부하들 덕분에 나라를 세울 수 있었음을 감사하며 낙양의 남궁에서 연회를 베풀었다는 이야기를 본떠서 이름을 지었다. 정조는 1795년 을묘원행 당시 낙남헌에서 수원의 백성들을 위해 잔치를 베풀고, 무과 시험을 치르고 상을 내리는 등 다양한 행사를 열었다.
낙남헌 건물은 벽이 없는 개방된 구조로 많은 사람을 수용할 수 있다. 연회를 베푸는 건물답게 건물 앞에는 넓은 월대를 두어 격식을 높였다. 월대로 오르는 계단 양 옆에는 구름무늬가 새겨져 있다. 낙남헌은 궁궐 전각과 비교해도 손색이 없는 아름다운 건물로 원형이 잘 남아 있다. 일제강점기에는 수원군청으로 사용되었고, 신풍국민학교 교무실로도 사용되었다.
[질문] 화성행궁에서 공식 행사나 연회를 열 때 사용하는 건물은?
[질문과 관련성] O
[답변] 화성행궁에서 공식 행사나 연회를 열 때 사용하는 건물은 낙남헌입니다.
[말투반영] {first_example}
예시 2
[이전 대화 맥락]
User: 수원 화성에 대해 설명해줘
Assistant: {context_example}
[글]
『화성성역의궤』는 정조가 구상한 신도시인 화성 성역 조성 전 과정을 기록한 종합 보고서입니다. 화성은 정조가 수원도호부 관아와 민가를 팔달산으로 옮겨 새롭게 조성한 신도시로, 1794년(정조 18) 1월에 공사를 시작하여 1796년(정조 20) 9월까지 32개월 만에 완성하였습니다. 공사 기간은 원래 10년을 계획했지만 정조의 각별한 관심과 조정의 적극적인 역할, 막대한 자금 투입, 치밀한 설계, 근대적인 공법 등 당시 국가의 역량이 총동원되어 공사 기간이 크게 단축되었습니다. 『화성성역의궤』에는 이러한 공사의 계획, 운영 과정, 참여자, 소요 경비, 자재, 공법, 도면 등 화성 축성의 전모가 기록되어 있습니다. 특히 ｢도설｣에는 건축 도면을 연상시킬 만큼 성곽과 부속 건물이 자세하게 그려져 있어, 일제 강점과 한국전쟁으로 훼손된 화성을 실제 모습 그대로 복원하는 데 중요한 자료가 되었습니다. 조선왕조 의궤는 2007년 일괄로 세계기록유산에 등재되었고, 『화성성역의궤』는 2016년 보물(옛 지정번호 보물 제1901-7호)로 지정되었습니다.
지난 2007년 유네스코 세계기록문화유산으로 등재된 조선왕조 의궤 중 정조시대 한글로 작성된 <정리의궤> 25권이 프랑스에서 발견됐다. 더불어민주당 안민석 의원(오산)은 4일 “지난달 27일 프랑스 국립파리동양언어학원과 프랑스 국립도서관에서 각각 24권과 1권의 한글 <정리의궤>를 확인했다”고 말했다. 이들 정리의궤는 정조의 어머니인 혜경궁 홍씨의 회갑연을 기록한 <원행을묘정리의궤>와 화성 축성과정을 기록한 <화성성역의궤>, 정조의 아버지인 사도세자의 묘소를 화산릉으로 옮긴 <현륭원의궤>의 내용 중 핵심 사안을 한글로 정리해 놓은 것이다.
[질문] 정리의궤가 뭐야?
[질문과 관련성] O
[답변] 정리의궤는 조선왕조의 궁정 기록 중 하나로, 조선왕조 의궤의 일부분입니다. 이 문서는 조선왕조의 여러 중요한 사건을 기록하고 있으며, 특히 정조 시대에 한글로 작성된 문서로 유명합니다. 2007년에는 유네스코 세계기록문화유산으로 등재되었습니다. 이 중에는 화성 축성과정을 기록한 '화성성역의궤', 정조의 어머니의 회갑연을 기록한 '원행을묘정리의궤', 그리고 정조의 아버지의 묘소를 화산릉으로 옮긴 것을 기록한 '현륭원의궤'의 내용 중 핵심 사안을 한글로 정리한 내용들이 포함되어 있습니다.
[말투반영] {second_example}
예시 3
[글]
홍예의 개수는 7이므로 북쪽 수문인 화홍문은 7번 구역에 있고 방화수류정이 7번 구역과 5번 구역에 경계에 있다는 것인데, 이 정보를 바탕으로 카카오맵과 같은 지도 앱을 활용해 용연의 위치를 기준으로 십자로 선을 그어서 창룡문과 팔달문이 어디 구역에 확인하면 될 것 같다/n퍼즐 그림을 보면 용연을 중심으로 두 개의 선을 기준으로 4개의 구역으로 나뉘는데, 팔달문과 창룡문이 각 구역 중 어디에 속하는지 알아야 할 것 같다/n퍼즐 이미지를 보면 가운데 있는 두 개의 원은 용연의 모습을 닮아있는데 가운데 원형 섬이 있고 연못 형태가 원형이라서 가운데 그림은 용연을 의미하는 것으로 보인다/n
[질문] 이 퍼즐 뭔가 이해가 안되는데 너도 한 번 답을 찾아봐
[질문과 관련성] O
[답변] 퍼즐 이미지 속 두 개의 원은 용연의 모습을 닮은 거 같습니다. 이 퍼즐은 용연을 중심으로 지리적 위치를 파악하는 게 핵심입니다. 먼저 용연을 기준으로 십자로 선을 그어 4개의 구역을 나누고, 화홍문이 7번 구역에 있고 방화수류정이 7번과 5번 구역의 경계에 있다는 정보를 활용합니다. 그다음, 팔달문과 창룡문이 각각 어느 구역에 속하는지 확인하면 될 것 같습니다. 이를 힌트로 삼아 해결할 수 있을 것 입니다.
[말투반영] {third_example}
예시 4
[글] 
동북공심돈은 화성 동북쪽에 세운 망루로 주변을 감시하고 공격하는 시설이다. 공심돈은 속이 빈 돈대라는 뜻으로, 우리나라 성곽 중 화성에서만 볼 수 있다. 보통 돈대는 성곽과 떨어진 높은 곳에 세워 적을 감시하는 시설이나, 동북공심돈은 성벽 안쪽에 설치했다. 외벽에는 밖을 감시하고 화포로 공격할 수 있는 구멍을 곳곳에 뚫었다. 동북공심돈은 3층으로 이루어진 원통형의 벽돌 건물로서 출입문에서 통로를 따라 빙글빙글 올라가면 꼭대기 망루에 이르는 구조다. 이 모습을 빗대서 ‘소라각’이라고도 부른다. 정조 21년(1797) 정월, 좌의정 채제공은 동북공심돈을 올라가 본 뒤 “층계가 구불구불하게 나 있어 기이하고도 교묘하다.”며 감탄했다.
한국전쟁 등을 겪으며 절반 이상 무너졌었는데 1976년에 복원해 모습을 되찾았다. 서북공심돈은 화성 서북쪽에 세운 망루로 주변을 감시하고 공격하는 시설이다. 공심돈은 속이 빈 돈대라는 뜻으로, 우리나라 성곽 중 화성에서만 볼 수 있다. 보통 돈대墩臺는 성곽과 떨어진 높은 곳에 세워 적을 감시하는 시설이나, 서북공심돈은 서북쪽 성벽이 꺾이는 위치에 설치했다. 치성 위에 벽돌로 3층의 망루를 세우고 꼭대기에는 단층의 누각을 올려 군사들이 감시할 수 있도록 하고, 외벽에는 화포를 쏠 수 있는 구멍을 뚫어 공격 기능까지 갖추었다. 조선 정조 21년(1797) 정월, 완성된 화성을 둘러보던 정조는 서북공심돈 앞에 멈춰 “우리나라 성곽에서 처음 지은 것이니 마음껏 구경하라.”며 매우 만족스러워 했다. 화성에는 모두 세 곳에 공심돈을 세웠는데 서북공심돈만이 축성 당시 모습 그대로 남아 있다.
[질문] 용인시에 대해 설명해줘
[질문과 관련성] X
주제와 관련 없는 답변입니다.
"""

    context_prompt = qa_2_str(messages)

    question_prompt = f"""문제
[글]
{rag_text}
[질문] {messages[-1]}
([글]과 [질문]이 관련이 있다면 [답변] 과 [말투반영]을 모두 출력할 것, 130자 내외로 답변할 것, 특히 말투 반영이 잘 안되는데, 확실하게 예시에서 주어진 말투를 매우 강하게 제대로 반영해야해.)
[질문과 관련성]을 판단하고, 만약 O라면 [답변]과 [말투반영]을 각각 작성해보자.
"""
    if len(context_prompt) == 0:

        user_prompt = f"""{task_prompt}
{few_shot_prompt}
{question_prompt}
"""
    else:
        user_prompt = f"""{task_prompt}
{few_shot_prompt}
{context_prompt}
{question_prompt}
"""
        
    
    print(user_prompt)
    
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return prompt
    


@spaces.GPU(duration=35)
def make_gen(QA, candidates, top_k, character_type):
    start_time = time.time()

    # Make For Rag Prompt
    rag_prompt = qa_2_str(QA)

    # Do RAG
    query_embeddings = embedding_model.encode([rag_prompt],
                                              batch_size=1,
                                              max_length=8192,
                                              )["dense_vecs"]
    key_embeddings = embedding_model.encode(candidates)["dense_vecs"]



    similarity = query_embeddings @ key_embeddings.T
    similarity = similarity.squeeze(0)

    rag_result = ""
    top_k_indices = np.argsort(similarity)[-top_k:]
    for idx in top_k_indices:
        rag_result += (candidates[idx] + "/n")
    rag_result = rag_result.rstrip()

    # Make For LLM Prompt

    final_prompt = make_prompt(QA, rag_result, character_type)

    # Use LLM
    streamer = TextIteratorStreamer(llm_tokenizer, skip_special_tokens=True)

    final_prompt = llm_tokenizer.apply_chat_template(final_prompt, tokenize=False, add_generation_prompt = True)
    inputs = llm_tokenizer(final_prompt, return_tensors="pt").to(zero.device)

    llm_model.to(zero.device)
    generation_kwargs = dict(
        inputs=inputs.input_ids,
        streamer = streamer,
        max_new_tokens=512
    )
    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)

    thread.start()
    
    is_start = False
    for idx, new_text in enumerate(streamer):
        
        if idx >= len(inputs):
            if is_start:
                yield new_text
            if not is_start and "[말투반영]" in new_text:
                is_start = True
            # yield new_text
            
    is_start = False

    elapsed_time = time.time() - start_time
    print(f"time:{elapsed_time}")




@app.post("/")
async def root_endpoint(request: Request):
    data = await request.json()
    QA = data.get("QA")
    candidates = data.get("candidates")
    top_k = data.get("top_k")
    character_type = data.get("character_type")
    return StreamingResponse(gen_stream(QA, candidates, top_k, character_type), media_type="text/event-stream")


async def gen_stream(QA, candidates, top_k, character_type):
    for value in make_gen(QA, candidates, top_k, character_type):
        yield value


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```
</details>


## Git Convention 관련해서 부족했던 점
- Branch를 따로 파지 않는 작업에 대해서도 Issue 부여하고 Issue Tag하면 커밋들이 잘 분류된다.
- 그런데 나는 Branch를 따로 파는 작업에 대해서만 Issue 작성함
- 다음부터는 Branch 따로 안 팔 때도 Issue를 파고 거기에 커밋 내역들을 모으자.

