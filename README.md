# AI Tutorial Plus — Alzheimer MRI Severity Classification (Team 5)

뇌 **MRI** 영상으로 알츠하이머 **중증도 4단계**(Non-Demented, Very Mild, Mild, Moderate)를 분류하는 딥러닝 프로젝트입니다.  
**CNN의 로컬 특징**과 **ViT의 전역 문맥**을 결합한 **CNN-ViT 하이브리드**로, 정상↔초경증 경계 판별을 개선하는 것이 핵심 목표입니다.

## Dataset
- Original Alzheimer MRI Dataset (OASIS 기반)
- 총 6,400장 / 4 classes (불균형 고려, 증강·샘플링 적용)

## Methods
- Baseline: ViT (timm)
- Distilled: DeiT-S (ResNet-50 Teacher → ViT Student)
- Proposed: **CNN-ViT Hybrid**
  - ResNet-50 feature ↔ ViT classifier 조합
  - Class-Balanced Sampler, Label Smoothing, Dropout
  - Attention Rollout 근거 시각화, Calibration(T-scaling)

## Key Findings
- 정상↔초경증 오분류 **유의미 감소(≈47%↓)**  
- 소수 클래스(Moderate) 예측 안정성 향상  
- 의료영상에서 **데이터 불균형/저량** 조건에 강건

> Stack: PyTorch, timm · Metrics: Confusion Matrix, Calibration, Attention Rollout
