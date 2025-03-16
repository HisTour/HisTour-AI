# NadeulAI Chatbot(구, HisTour)

<p align="center">
<img src="https://github.com/user-attachments/assets/cf82f8dc-749d-40cc-9da3-c96f1f3bb189">
</p>

## Objective

- 나들ai 서비스는 여행을 다니며 다양한 미션을 수행하고, 여행지에 대한 다양한 역사 정보 등을 얻을 수 있는 앱입니다.
- 특히 미션을 수행하는 과정에서, 사용자가 선택한 캐릭터에게 질문을 할 수 있습니다.
- 이 질문을 받아주는 챗봇 서비스를 수행할 수 있는 시스템을 구축합니다.
- 챗봇은 미션 관련된 질문을 받아줄 수 있어야 하며, 해당 캐릭터의 말투대로 출력할 수 있어야 합니다.

## How to make


<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/13fa176c-1e25-40e0-88ef-3b3af46d69c2" />
</p>

- 간단한 벤치마크를 돌려본 결과 성능이 제일 준수했던 Qwen2 7B Instruct 모델을 사용했습니다.
- RAG를 통해 답변에 필요한 정보를 가져와 프롬프트에 추가합니다.
- CoT 과정을 통해 해당 미션과 관련 없는 질문은 무시하도록 프롬프팅합니다.
- CoT 과정을 통해 생성된 답변을 캐릭터 말투가 입혀진 버전으로 바꾸어 재작성합니다.
- 미션별 예시 QA 쌍을 활용한 벤치마크를 통해 올바르게 출력되는지 테스트합니다.

## 1차 완성 버전 서빙 방식 (HF Zero의 성능 좋지 않음 -> 아래에서 고도화 작업)

<details>
<summary>모델 구조 그림</summary>
<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/4f50d68b-78c9-479e-b030-38f4a91579b8" />
</p>
</details>


- HuggingFace Spaces Zero를 활용해 6개 분산 서버로 서빙
- Proxy 서버 (위의 모델 구조 그림에서 Python Server)에서 로드밸런싱 및 스트리밍 관련 로직 처리 수행
- GPU를 필요한 순간에만 대여하는 방식이므로 높은 성능을 내기에는 문제가 있음
- 실제 프로젝트 제출 시에는 비용 문제로 인해 이 방법으로 서빙을 수행

# 챗봇 서빙 고도화 프로젝트

전체 계획 관련 이슈: https://github.com/HisTour/HisTour-AI/issues/16

## 1. 서빙 환경 고도화를 위한 Observability 확보

<p align="center">
<img width="1060" alt="image" src="https://github.com/user-attachments/assets/bf5715f2-7bfb-4566-bd2b-0069e2e42f1b" />
</p>

- Prometheous와 Grafana를 활용해 모니터링 환경 구축
- 지표는 아래와 같이 구성
  - Resources
    - CPU Memory Usage (사용량, 백분율)
    - CPU 사용량 (백분율)
    - GPU Memory Usage (사용량, 백분율)
    - GPU 사용량 (백분율)
  - Latency (P50, P90, P99)
    - TTFT (Time To First Token)
    - TPOT (Time Per Output Token)
    - TTFRT (Time To First Responsed Token) (커스텀 지표 - 본 모델은 CoT가 종료된 후 부터 Token을 전송함)
    - Request E2E Time
  - Throughput
    - TPS (Tokens Per Second)
- 상세한 내용은 아래 이슈를 확인해주세요.
  - https://github.com/HisTour/HisTour-AI/issues/17   

## 2. RAG 시 VectorDB (ChromaDB) 적용

- 기존에는 요청이 들어오면 그때그때 해당 미션에 맞는 후보군 벡터들을 임베딩화 시켜 내적하였음
- 당연히 성능이 좋지 않음
- 그래서 ChromaDB를 적용하여 미션 별로 임베딩 벡터로 저장하여 성능을 높이고 유지보수하기 편리하게 함
- VectorDB는 1차 완성본에서 Proxy 서버 위치에 배치함
- 상세한 내용은 아래 이슈를 확인해주세요.
  - https://github.com/HisTour/HisTour-AI/issues/18
  - 
## ⭐️3. vLLM 적용 및 파라미터 튜닝 : 🔥 Throughput 165배 향상 🔥

![vllm-logo-text-light](https://github.com/user-attachments/assets/8f541611-8e50-4627-b9ff-e21fa15984a1)


- vLLM은 Paged Attn, Continous Batching을 활용해 LLM 서빙 성능을 극대화함
- vLLM을 도입하여 TPS 4.8 -> 약 50으로 상승
- 서비스 특성상 중복 프롬프트가 많다는 점에서 Prefix Caching을 적용하여 TPS 50 -> 약 600으로 상승
- Prefill Phase가 Decoding Phase보다 우선순위가 높은데, 이로 인해 이미 Decoding 처리 중이던 요청이 Block 되는 것을 막기 위해 Chunked Prefill을 Enable함
- bf16 -> fp8로 Quantization, 출력 품질은 놀랍게도 큰 차이 없음, TPS 약 600 -> 약 800으로 상승
- 최종적으로 유효수준 Latency 유지하며 Throughput 165배 향상
- 상세한 내용은 아래 이슈를 확인해주세요.
  - https://github.com/HisTour/HisTour-AI/issues/21
  - https://github.com/HisTour/HisTour-AI/issues/22
  - https://github.com/HisTour/HisTour-AI/issues/23
  - https://github.com/HisTour/HisTour-AI/issues/24
  - https://github.com/HisTour/HisTour-AI/issues/27   

## 4. gRPC 도입

- 향후 K8S 기반 CloudNative 형태로 구조를 바꿔볼 예정이라 gRPC를 도입해보았음
- LLM에 대한 Proxy 서버와 LLM 서버 사이에 적용함
- LLM 작업이 워낙 오래 걸리는 작업이다보니 두드러지게 성능 향상이 있는 것은 아니나 분명히 네트워크 속도를 개선하기에 적용하였음
- 상세한 내용은 아래 이슈를 확인해주세요.
  - https://github.com/HisTour/HisTour-AI/issues/25 

# Work In Progress

## 5. Triton 서버로 Wrapping

- Triton 서버를 이용하면 전처리, 후처리, 모델 로직을 선언형으로 관리가 가능
- 이는 gitOPs와 궁합이 잘 맞음
- 또한 차후 TensorRT-LLM 기반 서빙 / vLLM 기반 서빙 모두 지원할 수 있게 대응이 가능

## 6. GitOPs, K8S 기반 클러스터 구성 및 Wrapping CRD 구성

- 현재 vLLM, TensorRT-LLM, SGLang, llama.cpp 등 다양한 서빙 환경이 존재함
- 또한 앞으로도 다양한 형태의 더 성능 좋은 서빙 환경이 등장할 가능성이 높음
- 이러한 환경에 빠르게 대응할 수 있도록 K8S Custom Resource를 만들어, 비즈니스 로직 부분의 코드 수정은 최소화하고 다양한 서빙 프레임워크를 쉽게 도입할 수 있게끔 의존성 역전을 실현함
- 이 모든 것은 선언형 Manifest로 관리되어 ArgoCD 기반 GitOPs 실현, Helm Operator 방식으로 관리
- 모델 레지스트리를 이용해 새로운 모델 업로드 시 자동 배포 로직 구현

## 7. 캐릭터 말투 입히는 과정을 LoRA Adapter로 변경

- 현재는 캐릭터 말투를 입히는 과정을 CoT로 처리하고 있음
- 그렇기 때문에 실질적으로 답변이 가는 시간이 느림
- 이 부분을 LoRA Adapter로 처리하여 CoT 과정을 줄여 성능을 높임
- 이 때 새로 생산된 모델은 구축된 자동화 배포 파이프라인을 잘 타는지 확인
