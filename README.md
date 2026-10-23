# LLM 파인튜닝 튜토리얼 (Step by Step)

기초적인 학습 개념부터 시작해서, NVIDIA GPU 또는 CPU 환경에서 LLM 파인튜닝을 단계별로 배우는 실습형 과정입니다.
각 단계는 한국어 설명이 담긴 Jupyter 노트북으로 되어 있고, 코드를 직접 돌려 보면서 개념을 이해하도록 구성했습니다.

**대상:** Python과 PyTorch는 알고, LLM inference는 대략 알지만 파인튜닝은 처음인 분

## 과정 구성

| 단계 | 노트북 | 내용 | 상태 |
|---|---|---|---|
| 1 | [01_basics_pretraining_vs_finetuning.ipynb](notebooks/01_basics_pretraining_vs_finetuning.ipynb) | 프리트레이닝 vs 파인튜닝, 토큰화, 다음 토큰 예측 loss, 순수 PyTorch로 작은 GPT를 직접 학습하고 파인튜닝, GPU 메모리 계산 | ✅ |
| 2 | 02_full_finetuning.ipynb | Hugging Face로 작은 모델(GPT-2 small 또는 Qwen2.5-0.5B) Full 파인튜닝. fp16, gradient accumulation 등 메모리 절약 팁 | 예정 |
| 3 | 03_lora_finetuning.ipynb | PEFT로 같은 작업을 LoRA로 파인튜닝하고 Full과 메모리/결과 비교 | 예정 |
| 4 | 04_evaluation_and_next_steps.ipynb | 학습 전후 출력 비교, 모델 저장/로드, 다음 단계(QLoRA, SFT 데이터 만들기) | 예정 |

## 환경 준비

```bash
git clone https://github.com/ldsljh-dot/llm-finetuning-tutorial.git
cd llm-finetuning-tutorial

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# NVIDIA GPU가 있다면 PyTorch를 먼저 CUDA 버전으로 설치 (https://pytorch.org/get-started/locally/ 참고)
# 예: pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

jupyter lab
```

CPU만 있어도 1단계는 1~2분 안에 끝납니다. 노트북은 CUDA가 있으면 자동으로 GPU를 사용합니다.

### GPU VRAM 확인하기

```bash
nvidia-smi
```

`Memory-Usage` 칸의 오른쪽 숫자가 전체 VRAM입니다. RTX 3060은 데스크톱이 보통 12GB(일부 8GB), 노트북이 6GB입니다.
VRAM 크기에 따라 2단계(Full)와 3단계(LoRA) 중 어떤 방식이 현실적인지가 달라지며, 1단계 노트북 마지막에서 직접 계산해 봅니다.

## 폴더 구조

```
.
├── README.md
├── requirements.txt
└── notebooks/      # 단계별 학습 노트북
```
