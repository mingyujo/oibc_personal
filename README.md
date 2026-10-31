# OIBC portable models

원본 데이터 폴더를 지정하면 다른 Windows/Linux 컴퓨터에서도 입력 생성, 모델 학습, 시간순 평가와 예측 파일 생성을 실행할 수 있는 독립 패키지입니다. 과거 실험 캐시나 특정 PC의 경로가 필요하지 않습니다.

지원 모델은 **LightGBM-MAE, CatBoost-MAE, CatBoost-RMSE, 24시간 MLP, 3시간 패치 Attention**입니다. 기존 모델 검토에서 확인한 내부 검증 시간 경계를 보완하고, 모든 모델의 조기 종료를 최종0하한 출력의 용량 가중 MAE로 맞췄습니다. 이 변경은 새 학습 레시피입니다. 기존 실험 점수나 보존된 제출 모델을 그대로 재현했다고 주장하지 않습니다.

지금까지의 시도와 결과는 [실험 기록·주요 결과·EXP-001~026 색인](docs/HISTORY.md)에서 볼 수 있습니다. [이전 전체 기록](learning_history.md)과 [최신 전체 기록](learning_history_02.md)을 함께 보관합니다.

## 설치

Python3.12를 권장합니다. 기본 설치는 CPU 트리 모델용이며 시스템 PATH나 전역 Python을 바꾸지 않습니다.

ZIP으로 받았다면 압축을 푼 프로젝트 폴더에서 시작하고 아래 `git clone` 단계는 생략합니다.

Windows PowerShell:

```powershell
git clone https://github.com/mingyujo/oibc_personal.git
cd oibc_personal
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Linux:

```bash
git clone https://github.com/mingyujo/oibc_personal.git
cd oibc_personal
sh setup.sh
.venv/bin/python -m unittest discover -s tests -q
```

직접 설치할 수도 있습니다.

```bash
python -m venv .venv
# 가상환경의 Python을 사용합니다.
python -m pip install -e .
python -m oibc_portable doctor
```

MLP/Patch는 `pip install -e ".[neural]"` 또는 `setup.ps1 -Neural`로 선택 설치합니다. PyTorch CUDA 설치 명령은 [공식 설치 안내](https://pytorch.org/get-started/locally/)에서 운영체제와 GPU에 맞게 선택하세요. 기본 CPU 경로는 합성 데이터로 통합 검증했습니다. GPU 경로와 Linux 실행은 해당 환경에서 별도 확인이 필요합니다.

## 데이터

원본 데이터와 학습 가중치는 저장소에 포함하지 않습니다. 제공받은 데이터를 별도로 옮기고, 다음 파일이 바로 있는 폴더를 `--data-dir`로 지정합니다.

```text
01_participant_dataset/
  sites.csv
  smp.csv
  train/
    generation.parquet
    forecast_wa.parquet
    forecast_wb.parquet
    smp.csv
  test/
    forecast_wa.parquet
    forecast_wb.parquet
```

기상 실측 파일은 이 기본 발전량 학습 경로에서 사용하지 않습니다. 파일을 원본에 덮어쓰지 않으며 학습 준비 전후 데이터 SHA256을 확인합니다. 없는 발전량 정답과 음수 발전량은 마스킹하고, 기록된0은 유지합니다. 비정시 예보는 반올림하지 않고 제외하며 건수만 기록합니다.

## 한 번 준비하고 반복 학습

아래 예시는 Windows이며 Linux에서는 가상환경 Python 경로만 바꾸면 됩니다.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m oibc_portable prepare --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-runs' --start 2021-01-01 --end 2025-05-31

& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model lgbm_mae --recipe raw6 --cutoff 2024-05-31 --threads 6
```

실제 작업 경로는 지정 폴더 아래 `.local-only/`입니다. 학습 이력이 있는 발전소-날짜의24시간 입력만 월별로 보관해 전체 빈 격자 캐시를 줄입니다. 과거 관측이 전혀 없는 날의 정답을 만들어 넣지 않습니다.

전체 이력 트리 학습은 **RAM32GB 이상, 작업 드라이브 여유20GB 이상**을 권장합니다. 캐시와 학습 준비 배열이 몇GB가 될 수 있으며 모델별 메모리는 자료에 따라 달라집니다. 디스크 부족은 검사해서 중단합니다. 메모리가 부족한 PC에서는 `--history-days 365`처럼 학습 창을 명시적으로 줄일 수 있지만, 전체 이력 실험과 다른 설정으로 기록해야 합니다. 한 번에 한 모델을 실행하는 것이 기본입니다.

CatBoost와 선택적 신경망:

```powershell
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model catboost_mae --cutoff 2024-05-31 --threads 6
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model catboost_rmse --cutoff 2024-05-31 --threads 6
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model mlp24 --cutoff 2024-05-31 --device gpu
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model patch3 --cutoff 2024-05-31 --device gpu
```

CatBoost는 `--device gpu`를 선택할 수 있습니다. GPU 미지원 환경에서 CPU로 몰래 바꾸지 않고 오류 로그를 남깁니다. LightGBM은 일반 pip 설치의 이식성을 위해 CPU만 지원합니다.

## 중단 후 재개

같은 명령에 `--resume`를 붙입니다. 캐시의 완료된 월, 학습의 완료된 selection/refit 단계를 재사용합니다. **미완료 단계 내부의 optimizer 상태를 이어가는 기능은 아닙니다.** 중단된 단계는 다시 시작합니다.

```powershell
& $py -m oibc_portable prepare --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-runs' --start 2021-01-01 --end 2025-05-31 --resume
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model lgbm_mae --cutoff 2024-05-31 --threads 6 --resume
```

실행 중인 PID와 생성 시각으로 중복 실행을 막습니다. 설정·코드·캐시 계약이 바뀌면 재개하지 않습니다. 다른 설정은 새 `--run-name`이나 새 작업 폴더를 사용합니다.

단계 시작과 성공 완료를 구분하고 `.local-only/logs/*.events.jsonl`에 진행 이벤트를 저장합니다. 긴 트리 적합은30초마다 생존시간·PID·메모리를 기록합니다. LightGBM은500회마다 실제 반복 수도 남깁니다.

## 평가

```powershell
& $py -m oibc_portable evaluate --work-dir 'D:\OIBC-runs' --model-dir 'D:\OIBC-runs\.local-only\models\lgbm_mae-raw6-2024-05-31-20261024' --start 2024-06-01 --end 2025-05-31
```

마감 이후 고정 모델의 정규화 MAE, kWh MAE, WAPE, 주간/야간 지표와 월별 결과를 저장합니다. 정산W는 **모든 발전소의 정답이 같은 시각에 있을 때만** 계산하며, 없으면0이 아닌null입니다. 관측이 일부 있는253개소 지표와 고정205개소 연간 지표는 같은 대상이 아닙니다. 이 패키지 기본 평가는 해당 기간의 모든 유효 정답을 사용합니다.

이미 살펴본 기간의 결과는 개발 평가입니다. 시드·마감을 늘려도 새 독립 최종시험이 되지는 않습니다. 입력 중요도 gain으로 변수 제거 여부를 결정하지 않습니다.

## 예보 정보의 추가 기여 비교

| recipe | 입력 | 목적 |
|---|---:|---|
| `raw6` | 262 | 원래 예보+결측/품질+시간 맥락+태양 기하 |
| `base137` | 137 | 원래 메타·달력·예보·결측/품질 |
| `no_weather` | 15 | 메타5+달력4+태양6. 예보 원변수와 모든 기상 파생값을 제외 |

동일한 모델·시점·시드로 `raw6`와 `no_weather`를 **각각 재학습**하고 이후 기간의 MAE와 정산을 비교합니다. 원본 예보만 삭제하고 기상 파생값을 남기는 것과 구분합니다. 변수 제거 효과도 인과효과는 아닙니다.

```powershell
# 1마감·1시드·3트리·2입력 = 모델6개, 최대12개 적합 단계
& $py -m oibc_portable experiment --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-runs' --config configs/quick_cpu.json --resume

# 3마감·3시드·2트리·2입력 = 모델36개, 최대72개 적합 단계
& $py -m oibc_portable experiment --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-runs' --config configs/confirmation_cpu.json --resume
```

이 설정은 장시간 학습입니다. 실행 전 JSON의 마감·시드·학습량을 확인하세요. 기존 완료 모델의 계약이 다르면 조용히 덮어쓰지 않고 중단합니다.

## 공식 예측 파일

공식 예측 전에는 선택한 설정으로 적절한 최종 마감에서 새로 학습합니다. 아래 명령이 모델을 새로 고르지는 않습니다.

```powershell
& $py -m oibc_portable train --work-dir 'D:\OIBC-runs' --model lgbm_mae --cutoff 2025-05-31 --threads 6
& $py -m oibc_portable predict --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-runs' --model-dir 'D:\OIBC-runs\.local-only\models\lgbm_mae-raw6-2025-05-31-20261024'
```

기본 배정은 원본의 참조A/B를 G1/G2로 매핑합니다. 별도 배정은 `--composition`으로 지정할 수 있습니다. 각 집합20MW 이상, 원본 전체 발전소 일대일 배정, 제공된8760시각과 일치,17520개 유한·비음수 예측을 검사합니다. CSV는 `.local-only/details/<실행이름>/`에 저장하고 자동 제출하지 않습니다. 공식 구간 발전량 정답은 없어 성능 점수를 산출하지 않습니다.

## 코드 검토·검증

- [모델 검토와 변경 이유](docs/MODEL_REVIEW.md)
- [검증 범위](docs/VALIDATION.md)
- 공개 대상 검사: `python tools/check_release.py`
- 에러 상세는 `.local-only/logs/`에만 기록합니다. 터미널에는 단계·건수·집계 지표만 출력합니다.

기존 실험 코드·모델·제출 파일과 별도의 새 패키지입니다. 기존 점수, 모델 승격 기준 또는 진행 중인 실험을 소급 변경하지 않습니다.

## GitHub에 코드만 게시

연결된 앱에서 기존 저장소에 업로드하거나, GitHub CLI로 직접 게시할 수 있습니다. 원본·캐시·모델·상세출력은 게시 허용 목록에 없습니다.

```bash
gh auth login
python tools/publish_github.py --repo mingyujo/oibc_personal --create --dry-run
python tools/publish_github.py --repo mingyujo/oibc_personal --create
```

직접 게시 도구는 코드 허용 목록 검사 후 새 비공개 저장소를 만들고 원격 커밋SHA를 대조합니다. 기존 비공개 저장소라면 `--create`를 빼면 새 브랜치에 게시합니다. 기존 원격 브랜치를 덮어쓰거나 강제 push하지 않습니다. 토큰을 코드에 넣지 않고 이미 인증된GitHub CLI의 정상 credential helper를 사용합니다.
