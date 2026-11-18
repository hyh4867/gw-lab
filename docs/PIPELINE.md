# 파이프라인 (PIPELINE) — gw-lab

절차의 근거와 고정 값은 [PREREGISTRATION.md](PREREGISTRATION.md) 4, 6, 7, 9절에 있다. 이 문서는 무엇을 어떤 순서로 실행하고 어디에 무엇이 남는지를 적는다.

## 환경

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-optional.txt
```

- 필수: numpy, scipy, matplotlib, h5py, emcee, pytest.
- 선택(`requirements-optional.txt`): lalsuite. IMRPhenomD 템플릿을 만들고, 직접 구현한 TaylorF2 위상과 안테나 패턴을 교차 확인하는 데 쓴다. macOS arm64용 wheel이 있어 설치에 문제가 없었다(가상환경 전체 315 MB). pycbc는 쓰지 않는다. matched filter 자체를 직접 구현하는 것이 목표라서 필요하지 않았다.
- lalsuite가 없으면 TaylorF2 템플릿과 matched filter 테스트만 돌 수 있다(IMRPhenomD를 쓰는 테스트와 탐색은 실패한다).

## 실행 순서

| 순서 | 명령 | 하는 일 | 산출물 | 걸린 시간 (8 GB M-시리즈 Mac) |
|---|---|---|---|---|
| 1 | `python scripts/download_data.py` | GWOSC strain 4개 받기, SHA-256 출력 | `data/raw/*.hdf5` (git 제외) | 약 1분 |
| 2 | `python -m pytest` | 단위 테스트 (자료 불필요) | — | 3초 |
| 3 | `python scripts/run_search.py` | 4096초 전체, 검출기별로 거친 뱅크 496개 탐색. 구간마다 trigger 저장(중간에 끊겨도 이어서 실행) | `results/triggers/imr/<det>/seg_*.npy` (git 제외), 요약은 `results/triggers_imr_<det>.csv` | 약 10분 |
| 4 | `python scripts/analyze_event.py` | zero-lag에서 가장 큰 동시 발생 후보를 찾고, 그 구간에서 거친·세밀한 뱅크(IMRPhenomD, TaylorF2), merger·ringdown 기여, f(t) chirp mass, 그림 1~6 | `results/event.json`, `docs/figures/01~06_*.png` | 약 1분 |
| 5 | `python scripts/background.py` | time slide 배경과 거짓 경보율, 그림 7 | `results/background.json`, `docs/figures/07_background.png` | 수 초 |
| 6 | `python scripts/run_injections.py` | 주입 900개(20개마다 저장, 이어서 실행 가능) | `results/injections.npz` (git 제외) | 약 35분 |
| 7 | `python scripts/analyze_injections.py` | 효율 곡선, 그림 8 | `results/injections.json`, `results/injections.csv`, `docs/figures/08_injections.png` | 수 초 |
| 8 | `python scripts/bank_match.py` | 거친 뱅크의 match 분포 | `results/bank_match.json` | 약 1분 |
| 9 | `python scripts/run_pe.py` | emcee로 (ℳ, q), 그림 9 | `results/pe.json`, `docs/figures/09_pe.png` | 수 분 |

오래 걸리는 3, 6은 `screen -dmS <이름> zsh -c "caffeinate -i .venv/bin/python scripts/... > logs/....log 2>&1"`로 분리 세션에서 하나씩 돌렸다.

## 자료 흐름

```
strain (4096 s, 4096 Hz)
  └─ 32초 구간 j (16초 간격, 가운데 16초만 유효)
       ├─ PSD: 구간 중심 ±256초, Welch 4초 조각·Hann·50 % 겹침·중앙값 → 1/32 Hz로 선형 보간
       │        → 1/S의 시간 응답을 ±2초로 자름                         [psd.psd_for_segment]
       ├─ Tukey 창(1초 taper) → rfft                                     [mf.to_fd]
       └─ 템플릿마다 z(t) = 4∫ s̃ h̃*/S e^{2πift} df (ifft 한 번), ρ = |z|/σ   [search.bank_filter]
            └─ 표본마다 뱅크 최댓값 → ρ ≥ 5.5, ±0.5초 안 최대 → trigger
                 └─ χ²(p = 16) → ρ̂ (재가중), ρ̂ ≥ 5                       [search.triggers_from_series, mf.chisq]
H1 trigger × L1 trigger
  ├─ |Δt| ≤ 15 ms → zero-lag 후보, ρ̂c = √(ρ̂_H² + ρ̂_L²)                 [coinc.zero_lag]
  └─ L1을 0.1초 × k (k = 1…40799) 원형 이동 → 배경                       [coinc.slide_coincidences]
```

## 모듈

| 파일 | 내용 |
|---|---|
| `src/gwlab/const.py` | 물리 상수와 사전 등록에서 고정한 설정값 |
| `src/gwlab/io.py` | GWOSC hdf5 읽기, SHA-256 |
| `src/gwlab/psd.py` | 중앙값 Welch, 보간, inverse spectrum truncation, whitening, 대역 통과 |
| `src/gwlab/mf.py` | 내적, σ², 복소 필터 출력, SNR 시계열, 포물선 정점, χ², 재가중 SNR |
| `src/gwlab/waveforms.py` | 직접 구현한 TaylorF2(3.5PN), lalsimulation IMRPhenomD 래퍼, f_ISCO, chirp mass |
| `src/gwlab/detectors.py` | H1·L1 검출기 텐서, 안테나 패턴, 도착 시간 지연 |
| `src/gwlab/search.py` | 뱅크 생성, 구간 준비, 뱅크 필터, trigger, clustering |
| `src/gwlab/event.py` | 두 검출기 network SNR 스캔, 최적 신호 복원 |
| `src/gwlab/coinc.py` | 동시 발생, time slide |
| `src/gwlab/chirpfit.py` | zero-crossing으로 f(t) 읽기, f^(−8/3) 직선 fit |
| `src/gwlab/qtransform.py` | constant-Q 변환 |
| `src/gwlab/plotstyle.py` | 그림 공통 양식(H1 파랑, L1 주황 고정) |

## 구현에서 정한 세부 사항 (사전 등록에 없던 것)

판정 기준이 아닌 구현 선택이며, 결과를 보고 고른 것이 아니다(코드 커밋 67b4896은 자료를 받기 전이다).

- 1/S를 자르기 전에 15 Hz 아래를 0으로 둔다(내적 범위는 20 Hz부터라 결과에 들어가지 않는다).
- 템플릿을 complex64로 저장한다(메모리 128 MB). 필터 계산은 complex128로 한다.
- trigger 시각은 표본 시각이다(0.244 ms 격자). 사건 분석에서만 포물선 보간으로 표본 사이 정점을 구한다.
- 구간 안에서 ±0.5초 clustering을 한 뒤, 구간을 합쳐 한 번 더 clustering한다(구간 경계 중복 제거).
- 주입 신호가 든 구간의 PSD는 주입 전 자료로 추정한다(512초 중앙값이라 차이는 무시할 만하다).
- 주입 파형은 20 Hz에서 갑자기 시작하는 주파수 영역 파형을 그대로 시간 영역으로 바꿔 더한다. 20 Hz 근처의 작은 ringing이 생기지만 필터 대역의 가장자리다.
- Q-transform은 Q = 6, 25~400 Hz 로그 간격 120개 주파수.

---

# 5단계 (2026-11-16)

절차의 근거와 고정 값은 [PREREGISTRATION_STAGE5.md](PREREGISTRATION_STAGE5.md)에 있다. 0~4단계의 스크립트와 결과 파일은 그대로 두었다.

## 환경 추가

- bilby 2.8.2 (+ dynesty 3.2.0): `uv pip install bilby`. 가상환경 323 MB → 379 MB. lalsimulation의 IMRPhenomD를 그대로 쓴다.
- PyTorch 2.15.0 (CPU): CNN 비교(선택)에만 쓴다.

## 실행 순서

| 순서 | 명령 | 하는 일 | 산출물 | 걸린 시간 |
|---|---|---|---|---|
| 1 | `python scripts/select_background_files.py` | GWOSC 메타데이터로 배경용 파일 선택 | `data/meta/background_files.json` | 수 초 |
| 2 | `python scripts/make_lowmass_bank.py` | 저질량·정렬 spin 확률적 뱅크 (GW150914 파일의 PSD만 사용) | `results/bank_lowmass.csv`, `.json` | 4분 |
| 3 | `python scripts/download_stage5.py` | 자료 32개 받기, SHA-256 | `data/raw/*` (git 제외), `data/meta/stage5_files.json` | 약 8분 |
| 4 | `python scripts/run_search5.py --bank lowmass --start 1135134303` | GW151226 사건 파일 탐색 (1,318개 템플릿) | `results/triggers/lowmass/<gps>/<det>/` (git 제외) | 약 11분 |
| 5 | `python scripts/analyze_gw151226.py` | zero-lag 후보, 세밀한 뱅크, 무회전 비교, 그림 12 | `results/event_gw151226.json` | 약 2분 |
| 6 | `python scripts/chirp_newton.py GW150914` (와 `GW151226`) | Newton chirp 필터: 주입 1,000개로 검증·보정 후 사건에 적용, 그림 14·15 | `results/chirp_newton_<event>.json` | 7분, 5분 |
| 7 | `python scripts/compare_events.py` | 두 사건 비교 수치, 그림 13 | `results/compare_events.json` | 7초 |
| 8 | `python scripts/run_pe_spin.py GW150914` (와 `GW151226`) | 보조 PE: 직접 쓴 가능도 + emcee, (ℳ, q, χ) | `results/pe_spin_<event>.json` | 2분, 5분 |
| 9 | `python scripts/run_search5.py --bank coarse --start <8개>` | GW150914 배경용 8개 파일 탐색 (496개 템플릿) | `results/triggers/coarse/<gps>/<det>/` | 73분 |
| 10 | `python scripts/background5.py GW150914` | 고리 time slide, 모든 쌍 추정, jackknife, 그림 10 | `results/background_ext_GW150914.json` | 수 초 |
| 11 | `python scripts/run_search5.py --bank lowmass --start <4개>` | GW151226 배경용 4개 파일 탐색 | `results/triggers/lowmass/<gps>/<det>/` | 50분 |
| 12 | `python scripts/background5.py GW151226` | 위와 같음, 그림 11 | `results/background_ext_GW151226.json` | 수 초 |
| 13 | `python scripts/run_pe_bilby.py GW150914` (와 `GW151226`) | bilby + dynesty, 정렬 spin IMRPhenomD (`--dry`는 구성만 확인) | `results/pe_bilby_<event>.json` (sampler 출력 `results/pe_bilby/`는 git 제외) | 43분, 78분 |
| 14 | `python scripts/compare_pe.py` | 문헌·GWTC-1 표본과 비교, 판정 F·G8~G11, 그림 16 | `results/pe_compare.json` | 수 초 |
| 15 | `python scripts/diag_stage5.py` | whitening RMS 원인, 일괄 파일과 사건 파일 비교 | `results/diag_stage5.json` | 약 20초 |
| 16 | `python scripts/cnn_train.py`, `python scripts/cnn_eval.py` | (선택) CNN 학습과 같은 주입에서의 비교, 그림 17. `cnn_eval.py wide`는 변경 2의 참고 결과 | `results/cnn_train.json`, `results/cnn_eval.json`, `results/cnn_eval_wide.json` | 14분, 8분, 6분 |

4~16은 screen 분리 세션 하나에서 `caffeinate -i`로 차례대로 돌렸다(한 번에 하나). 모든 긴 작업은 구간·주입·체크포인트 단위로 이어서 실행할 수 있다.

## 5단계에서 달라진 자료 흐름

```
저질량 탐색 (GW151226)
  64초 구간 (32초 간격, 가운데 32초 유효)  ← 템플릿이 최대 약 11초
  ρ(t)는 2048 Hz 간격 (길이 절반의 복소 ifft)                         [search.LAYOUT_LOWMASS, search.bank_filter(decim=2)]
  NaN이 있는 구간은 버리고, PSD 창은 NaN이 없는 연속 구간 안에 둔다     [psd.valid_stretches, search.prepare_segment]

배경 확장
  파일마다 trigger → 두 검출기가 함께 유효한 구간만 이어 붙여 고리로   [coinc.to_ring]
  L1을 0.1초 × k 이동 → time slide                                    [coinc.slide_coincidences]
  모든 (H1, L1) 쌍 → R(x) = n(≥x) · 2w / T²                           [coinc.all_pairs_rate]
  각 trigger를 한 번만 쓰는 분포 (낮게 치우침, 진단용)                 [coinc.no_reuse]

f(t) chirp mass
  h̃_N(f; ℳ) ∝ f^(−7/6) exp[−i (3/128)(πℳf)^(−5/3)], 30~150 Hz, ℳ 격자 0.25 %
  → SNR 최대인 ℳ_raw → 주입으로 만든 보정 곡선의 역함수 → ℳ_cal     [chirpfilter.py, scripts/chirp_newton.py]
```

## 5단계 모듈

| 파일 | 내용 |
|---|---|
| `src/gwlab/bankgen.py` | 확률적 뱅크 배치, match, fitting factor |
| `src/gwlab/chirpfilter.py` | Newton 차수 chirp 필터 |
| `src/gwlab/cosmo.py` | 평탄 ΛCDM의 d_L ↔ z |
| `src/gwlab/cnn.py` | CNN 구조와 전처리 (선택) |
| `src/gwlab/search.py` | `Layout`(구간 배치), 정렬 spin 뱅크, 간격을 줄인 ρ(t) 추가 |
| `src/gwlab/coinc.py` | 고리 사상, 모든 쌍 추정, 재사용 없는 선택 추가 |
| `src/gwlab/waveforms.py` | 정렬 spin 인자, (ℳ, η) 변환, χ_eff 추가 |
| `scripts/evcommon.py` | 사건 구간을 두 검출기에서 읽는 공통 함수 |

## 5단계 구현에서 정한 세부 사항 (사전 등록에 없던 것)

- Newton chirp 필터의 출력은 1024 Hz 간격이고(대역이 150 Hz에서 끝남), 템플릿마다 |z|²의 포물선 보간으로 표본 사이 정점을 구한다.
- Newton chirp 주입은 주파수 영역에서 FFT한 자료에 더한다(주입 시각은 구간의 15~17초로 Tukey 창이 평평한 곳이다). 위상은 무작위다.
- GW151226의 H1 사건 구간은 자료 시작(1925초)에서 가까워, Newton chirp 필터의 32초 구간이 사건 중심에서 벗어나지 않게 연속 구간 안으로 밀어 넣는 분기를 두었다(실제로는 사건 중심 구간이 연속 구간 안에 들어가 쓰이지 않았다).
- bilby에 넘기는 자료는 앞뒤 2초를 더 붙여 2048 Hz로 낮춘 뒤 잘라 낸다(재표본화 필터의 가장자리 효과를 피함).
- bilby는 4개 프로세스(`npool=4`)로 돌렸다.
- CNN의 단일 검출기 trigger는 ±0.5초 안 최대인 창 가운데 logit이 큰 1,000개만 남긴다(문턱 값을 따로 정하지 않기 위해).
- `compare_events.py`의 사이클 수는 최적 템플릿의 해석 신호(analytic signal) 위상에서, 순간 주파수가 35 Hz 아래로 내려가는 마지막 시각부터 진폭 정점까지로 센다.

# 6단계 (GW170817) 실행 순서

기준은 [PREREGISTRATION_STAGE6.md](PREREGISTRATION_STAGE6.md)에 있다. 1~3은 자료를 받기 전에 실행했다.

| 순서 | 명령 | 하는 일 | 출력 | 걸린 시간 |
|---|---|---|---|---|
| 1 | `python scripts/make_bns_bank.py` | 무회전 TaylorF2 확률적 뱅크(1~2 M☉), fitting factor 시험 | `results/bank_bns.csv`, `results/bank_bns.json` (1차 배치는 `bank_bns_pass1.*`) | 15분 + 7분 |
| 2 | `python scripts/select_background_files.py GW170817` | 메타데이터로 배경 파일 선택 | `data/meta/background_files_GW170817.json` | 수 초 |
| 3 | (사전 등록 커밋) | | | |
| 4 | `python scripts/download_stage6.py` | 자료 A, B, 배경 파일, GWTC-1 표본 | `data/raw/`, `data/meta/stage6_files.json` | 약 8분 |
| 5 | `python scripts/run_search6.py --start 1187006835` | 자동 gating 뒤 사건 파일 탐색 (H1, L1) | `results/triggers/bns_gated/<gps>/` (git 제외) | 13분 |
| 6 | `python scripts/run_search6.py --start 1187006835 --variant raw` | gating 없이 같은 탐색 | `results/triggers/bns_raw/<gps>/` | 12분 |
| 7 | `python scripts/analyze_gw170817.py` | zero-lag 후보, glitch 처리 전후, 세밀한 뱅크, Virgo, TaylorF2 대 IMRPhenomD, 그림 18·19 | `results/event_gw170817.json` | 3.4분 |
| 8 | `python scripts/run_search6.py --start 1187012608 1186996224` | 배경 파일 2쌍 탐색 | `results/triggers/bns_gated/<gps>/` | 29분 |
| 9 | `python scripts/background6.py` (와 `raw`) | time slide 배경, 그림 21 | `results/background_GW170817.json` | 수 초 |
| 10 | `python scripts/chirp_newton6.py` | Newton chirp 필터: 주입 500개로 보정 후 사건에 적용, 그림 20 | `results/chirp_newton_GW170817.json` | 12분 |
| 11 | `python scripts/sky_gw170817.py` | 세 검출기 결맞음 통계량으로 하늘 위치, 그림 22 | `results/sky_gw170817.json` | 26초 |
| 12 | `python scripts/run_pe_bns.py --dry`, `python scripts/run_pe_bns.py`, `python scripts/run_pe_bns.py --check` | relative binning 점검, bilby + dynesty, 끝난 뒤 사후표본 40개에서 정확한 가능도와 비교 | `results/pe_bilby_GW170817_dry.json`, `results/pe_bilby_GW170817.json`, `results/pe_bilby_GW170817_rbcheck.json` | 1.5분, 2.1시간 (npool 3), 2분 |
| 13 | `python scripts/compare_pe6.py` | 문헌·GWTC-1 표본과 비교, 판정 M3~M8, 그림 23 | `results/pe_compare_GW170817.json` | 수 초 |
| 14 | `python scripts/compare_three.py` | 세 사건 비교, GW151226의 TaylorF2 측정, 그림 24 | `results/compare_three.json` | 1.5분 |

모두 screen 분리 세션에서 `caffeinate -i`로 하나씩 돌렸다. 탐색은 템플릿 250개마다, Newton 주입은 25개마다, PE는 10분마다 체크포인트를 남긴다.

## 6단계의 자료 흐름

```
strain (4096 s 파일)
  → 자동 gating: 64초 블록 whitening → |x| > 50σ인 곳 ±0.1초를 0으로, 양쪽 0.5초 roll-off   [gate.py]
  → 256초 구간 (128초 간격, 구간의 112~240초가 유효)                                      [bns.LAYOUT]
  → 구간마다 FFT, PSD(512초 중앙값 Welch), w = s̃/S (30~1024 Hz, complex64)               [bns.reduce_prep]
  → 템플릿마다: 기저 9개에서 TaylorF2 위상 재구성 → 모든 구간에 ifft(길이 1/4) → 최대 갱신  [f2fast.py, bns.filter_bank]
  → 구간마다 clustering, χ², 재가중 SNR → trigger                                          [bns.triggers]
  → H1–L1 동시 발생(15 ms), time slide                                                     [coinc.py, background6.py]
```

## 6단계 모듈

| 파일 | 내용 |
|---|---|
| `src/gwlab/f2fast.py` | TaylorF2 위상을 고정 기저의 선형 결합으로 다시 만드는 클래스 |
| `src/gwlab/bns.py` | 긴 템플릿용 탐색 루프(템플릿 바깥, 구간 안쪽), χ², trigger |
| `src/gwlab/gate.py` | glitch 탐지와 inverse Tukey gating |
| `src/gwlab/detectors.py` | Virgo, 항성시, 적경·적위 → 지구 고정 좌표 추가 |
| `src/gwlab/bankgen.py` | TaylorF2 match, 보충 배치(`top_up`) 추가 |
| `scripts/ev6common.py` | 자료 A·B 읽기, 사건 구간, 템플릿 scan, 기하학적 지연 |
