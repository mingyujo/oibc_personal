# OIBC 고정 1년 테스트베드

결과 예측 CSV가 있으면 학습을 다시 실행하지 않고 [CSV 채점 화면](CSV_TESTBED.md)에서 바로 W·S를 계산할 수 있습니다. W·S 등 금액은 원 단위 정수로 표시하고 계산 정밀도는 유지합니다.

테스트 기간은 사용자가 지정한 **마지막 1년, 2024-06-01~2025-05-31 대상일**입니다. 마지막 종료시각은 2025-06-01 00:00 KST입니다. 모델과 발전소 배정을 2024-05-31 00:00 KST 이전 정보로 고정하고 1년 전체를 평가합니다. 테스트 정답으로 월별 재학습하거나 조기 종료하지 않습니다.

대회 목적·정보 제한은 [공식 대회 안내](https://dataen.ai/challenge), 제공된 대회 개요·규정·데이터 README와 설명회 기록을 확인했습니다. 정산식·시간·입출력 계약과 각 근거의 SHA256은 [COMPETITION_CONTRACT.json](COMPETITION_CONTRACT.json)에 있습니다. 대회 요구사항과 이 프로젝트가 선택한 학습·통계 절차를 구분해 적었습니다.

## 먼저 합성 데모

설치를 마친 프로젝트 폴더에서 실행합니다. 대회 원본 없이 만들어진 가상 자료로 데이터 준비, 작은 모델 학습, 정산 비교, HTML 보고서, 제출 CSV 생성·검사를 한 번에 확인합니다. 짧은 데모 기간은 동작 검증용이며 실제 마지막 1년 평가 점수가 아닙니다.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m oibc_portable testbed demo --work-dir 'D:\OIBC-testbed-demo'
```

마지막 이벤트의 `report` 경로를 브라우저에서 열면 후보 비교표와 월별 결과가 보입니다. 결과는 지정 폴더의 `.local-only/` 아래에만 저장됩니다. 행별 예측·원본 식별자·좌표는 보고서에 넣지 않습니다.

## 실제 마지막 1년 평가

계획 확인은 학습을 시작하지 않습니다.

```powershell
& $py -m oibc_portable testbed plan --config configs/testbed_last_year.json
```

기본값은 LightGBM-MAE RAW6, CatBoost-MAE RAW6, LightGBM-MAE no_weather의 3후보 × 3시드입니다. 모델9개, 최대18개 selection/refit 단계이며 CPU2스레드로 순차 실행합니다. 학습량·입력 기간은 대회 필수 조건이 아닌 프로젝트 설정입니다. 장시간 실행이며 RAM32GB 이상과 충분한 작업 디스크를 권장합니다.

```powershell
& $py -m oibc_portable testbed run --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-testbed' --config configs/testbed_last_year.json --run-name last-year-v1
```

중단 후에는 동일 명령에 `--resume`를 붙입니다. 실행 중 프로세스 잠금, 설정·코드·패키지·데이터·가격·모델 해시를 검사합니다. 완료된 모델과 평가 작업을 재사용하고, 미완료 학습 단계는 다시 시작합니다. 기존 설정이 바뀌면 새 실행 이름을 사용합니다. 설정이 다른 데이터 캐시는 새 작업 폴더가 필요합니다.

```powershell
& $py -m oibc_portable testbed run --data-dir 'D:\OIBC\01_participant_dataset' --work-dir 'D:\OIBC-testbed' --config configs/testbed_last_year.json --run-name last-year-v1 --resume
& $py -m oibc_portable testbed report --work-dir 'D:\OIBC-testbed' --run-name last-year-v1
```

Linux에서는 Python 경로만 `.venv/bin/python`으로 바꿉니다. 실행기는 실측 기상이나 외부 사전학습 가중치를 모델 입력으로 받지 않습니다. SMP는 정산 계산에서 사용하며 발전량 모델의 미래 입력으로 넣지 않습니다.

## 보고서 읽기

- **S·W·R**: 모든253개소 정답이 동시에 있는 시간만 정산합니다. G1·G2 내부에서는 오차를 먼저 합산하고 절댓값을 취합니다. 서로 다른 집합의 오차는 상쇄하지 않습니다. SMP0원도 유효한 시간입니다.
- **공통 정답 시간**: 후보·시드마다 같은 달력과 정답 지원을 강제합니다. 정답이 전혀 없으면 S/W/R은0 대신null입니다. 기본48시간 미만이면 정산 순위 근거가 부족하다고 표시합니다. 이는 대회 규칙이 아닌 최소 지원 진단값입니다.
- **전체 관측 MAE와 완비 코호트 MAE**: 관측이 있는 전체 표본과, 1년8,760개 정답이 모두 있는 발전소 코호트를 따로 계산합니다. 원자료 기준 완비 코호트는 이전 감사에서205개소였지만 실행 시 다시 계산합니다. 이205개소만으로253개소 정산을 대체하지 않습니다.
- **시드·월·주야간**: 각 시드의 점수 평균을 비교합니다. 예측을 먼저 평균하는 앙상블이 아닙니다. 월별 S/R/W/MAE와 주야간 오차는 `summary.json`에도 보존됩니다.
- **불확실성**: 같은 날짜의 기준·후보 차이를 묶고, 시드별 차이의 일별 평균을7일 이동 블록으로1,000회 재표집합니다. 탐색 구간이며 다중 비교 보정을 하지 않았습니다. 테스트 연도는 이미 여러 번 확인한 과거 구간이므로 새 독립 최종시험의 보장으로 해석하지 않습니다.

HTML은 후보 이름 검색과 월별 상세 펼치기를 지원합니다. 설정과 해시는 `contract.json`, 진행·재개 상태는 `state.json`, 후보·시드별 결과는 개별 JSON에 기록됩니다. 자동 모델 승격·집합 교체·대회 제출은 하지 않습니다.

## 다른 모델·발전소 배정 비교

후보의 `model`은 `lgbm_mae`, `catboost_mae`, `catboost_rmse`, `mlp24`, `patch3`, 점검용 `zero`를 지원합니다. 신경망은 선택적 PyTorch 설치가 필요합니다. `recipe`는 `raw6`, `base137`, `no_weather` 중 하나입니다. `zero`는 약한 동작 점검 대조군이며 실제 비교 기준 모델을 대신하지 않습니다.

`composition: "reference"`는 제공된 A/B 배정을 G1/G2로 매핑합니다. 최적 배정이라고 가정하지 않습니다. 외부에서 만든 배정을 평가하려면 후보에 다음 객체를 넣습니다. 파일은 제공253개소를 정확히 한 번씩 배정해야 하며 각 집합20,000kW 이상입니다.

```json
{"composition": {"file": "../private/composition.csv", "fitted_through": "2024-05-31T00:00:00+09:00"}}
```

파일 경로는 설정 파일 기준이며 고정 시점 선언과 내용 해시를 계약에 기록합니다. 선언은 실제 제작 이력의 증명이 아니므로 제작 코드에서도 테스트 정답을 사용하지 않아야 합니다. 테스트1년에서 배정을 탐색하고 같은 기간을 배정 선택 전 독립 평가라고 부르면 안 됩니다. 같은 모델·레시피·시드로 배정만 바꾼 후보는 동일한 학습 모델을 재사용합니다. 최적 배정 탐색 알고리즘 자체는 이 테스트베드에 포함하지 않습니다.

## 대회 제출 파일 검사

공식 제출 구간은 **2025-06-01 01:00~2026-06-01 00:00 KST**로, 과거 테스트1년과 다릅니다. 그 구간의 실제 발전량은 없으므로 제출 형식·배정·시간 계약만 검사합니다.

```powershell
& $py -m oibc_portable testbed validate-submission --data-dir 'D:\OIBC\01_participant_dataset' --composition 'D:\OIBC-runs\.local-only\details\final\composition.csv' --forecast 'D:\OIBC-runs\.local-only\details\final\forecast.csv' --work-dir 'D:\OIBC-testbed'
```

검사는253개소의 일대일 고정 배정, 두 집합 각각20MW, 정확한 열과17,520행, 집합별8,760시각, 명시적 타임존, 중복·누락·음수·NaN·무한대, 제공 SMP의0이상 가격과종료시각 기준k를 확인합니다. 결과는 `.local-only/submission-checks/`의 JSON으로 저장합니다. 원본 CSV를 수정하거나 서버에 제출하지 않습니다. 공식 서버의 숨겨진 정답 점수·정규화 점수·동점 처리를 재현한다고 주장하지 않습니다.
