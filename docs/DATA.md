# 자료 (DATA) — gw-lab

원자료는 git에 넣지 않는다. 아래 방법으로 받고 SHA-256을 비교한다.

## 출처와 라이선스

- Gravitational Wave Open Science Center (GWOSC), https://gwosc.org
- 릴리스: GWTC-1-confident / GW150914 / v3 (event API: `https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/`, 자료 DOI https://doi.org/10.7935/82H3-HH23)
- 라이선스: Creative Commons Attribution 4.0 International (CC BY 4.0). 2026-11-15에 https://gwosc.org/acknowledgement/ 에서 확인.

## 받는 방법

```
python scripts/download_data.py
```

`data/raw/`에 4개 파일을 받고 SHA-256과 크기를 출력한다.

| 파일 | 크기 (byte) | SHA-256 |
|---|---|---|
| `H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5` | 1,040,781 | `c5ea87beced5094b56b4d694f3a36e6d59f74d3a46e546b980a6a199d1f252a9` |
| `L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5` | 1,007,262 | `9d65bde82f7ec035bea4c02228bce4acb3dcaae774c047edce71ef2ad64f1cc4` |
| `H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5` | 130,158,262 | `8ef8b762cff53b422429c474b1e35a6d9bfa86f3ed0c0d3e855f599781390a27` |
| `L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5` | 125,865,927 | `bdd2dca270ad5205355b35d1a0da764c3927650917d971a10ecc6113efe4ebf4` |

받은 날: 2026-11-15. 해시는 이 날 받은 파일에서 `gwlab.io.sha256`으로 계산한 값이다. GWOSC가 공식 해시를 따로 싣지 않아 외부 값과 대조하지는 못했다.

## 파일 내용 (읽어서 확인한 것)

- `strain/Strain`: float64, 표본 간격 1/4096초(`Xspacing` = 0.000244140625), 시작 GPS(`Xstart`) 1126257415(4096초 파일), 1126259447(32초 파일). 4096초 파일은 16,777,216 표본.
- `quality/simple/DQmask`: 1초마다 비트마스크. 비트 이름은 DATA, CBC_CAT1, CBC_CAT2, CBC_CAT3, BURST_CAT1, BURST_CAT2, BURST_CAT3.
- 4096초 전체에서 H1, L1 모두 NaN 0개, 7개 비트가 4096초 모두 켜져 있다(품질 플래그로 빠지는 구간 없음).
- 32초 파일과 4096초 파일의 겹치는 구간은 최대 절대 차이 0.0으로 같다(H1, L1).
- 사건 시각(GWOSC) GPS 1126259462.4 = 4096초 파일 시작에서 2047.4초.

## 5단계에서 추가한 자료 (2026-11-16)

```
python scripts/select_background_files.py   # 메타데이터만 읽어 배경용 파일을 고른다 -> data/meta/background_files.json
python scripts/download_stage5.py           # 아래 32개 파일을 data/raw/에 받고 SHA-256을 data/meta/stage5_files.json에 적는다
```

- 합계 3.55 GB. git에 넣지 않는다.
- **GW151226 사건 파일**: GWOSC GWTC-1-confident / GW151226 / v2 (`https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/`). CC BY 4.0, 인용 문구는 아래와 같다.
- **배경용 일괄 파일**: O1 4 kHz 자료 (`https://gwosc.org/archive/data/O1/…`, 자료 DOI https://doi.org/10.7935/K57P8W9D). CC BY 4.0. 고르는 규칙은 PREREGISTRATION_STAGE5.md 5절(두 검출기 모두 가동률 100 %, 사건 파일과 겹치지 않음, 사건에서 가까운 순).
- **GWTC-1 사후분포 표본**: LIGO DCC P1800370 (`https://dcc.ligo.org/LIGO-P1800370/public`). 비교용으로만 쓴다. 인용: B. P. Abbott et al., PRX 9, 031040 (2019).

| 용도 | 파일 | 크기 (byte) | SHA-256 |
|---|---|---|---|
| GW151226 사건 | `H-H1_GWOSC_4KHZ_R1-1135134303-4096.hdf5` | 69,367,295 | `e42fee9452e34d22d96c11cbaa96c7a545133b72d53ad30637769158ac1d066d` |
| GW151226 사건 | `H-H1_GWOSC_4KHZ_R1-1135136335-32.hdf5` | 1,040,482 | `e47c44aef95d7929e2a5a7723787acaa82545b55e7b6ac998e8acf6ccc65f264` |
| 겹침 확인(배경에 쓰지 않음) | `H-H1_LOSC_4_V1-1126256640-4096.hdf5` | 130,159,479 | `30ad150a40c911e24652f2ffc2f8e7a92bb60d579531dde3160f475bb306c41a` |
| GW151226 사건 | `L-L1_GWOSC_4KHZ_R1-1135134303-4096.hdf5` | 128,039,935 | `651f64f33599388115f044acc0e873df8e8864da98b95a4d218bededbe36ff1a` |
| GW151226 사건 | `L-L1_GWOSC_4KHZ_R1-1135136335-32.hdf5` | 1,022,103 | `90b6538f635d6c795bc3923139fcf2f1382346360997107b5a7c45ba293c48fc` |
| 겹침 확인(배경에 쓰지 않음) | `L-L1_LOSC_4_V1-1126256640-4096.hdf5` | 120,033,701 | `b6896cb749931cfc0c35fb01e750f1c4e6847830122f9afa5640ebc78e97f384` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126309888-4096.hdf5` | 130,181,364 | `e010d118c00b2b9438003f44b044f31236563d8417b05a672662302eef6d0e73` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126309888-4096.hdf5` | 125,941,264 | `7f5317a3645e891e021db458dee67d121a1f1c3cbd0fe2004bc7287de8db8892` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126313984-4096.hdf5` | 130,178,341 | `7ffee2efdb7e4d4909fc51fdb77037b85edfd8a9e601dba6e6d36b330ee26c4a` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126313984-4096.hdf5` | 125,893,360 | `2a4b87e93c628ad8d369c897d6b4476ab36a31dbbb2e507b215a2936873406ee` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126318080-4096.hdf5` | 130,171,133 | `5c29608477d527646a1bba9cf70acd44ff6f964b042d228bffa611a658a5fb7d` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126318080-4096.hdf5` | 125,820,553 | `4df140036c6fb73e2bcda6c4e11cb68fb85c671045cef8c1d84fd350c7cd6373` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126322176-4096.hdf5` | 130,173,566 | `9ea6b08e80eaf02a87c6d22cd989b3a1c7a408b55edf5d3e0263ff64a727cf53` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126322176-4096.hdf5` | 125,841,353 | `73bf86ae25df7a27e9c0829fd2fbaf4eb1bf148310f43ed3ac593b029182668d` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126404096-4096.hdf5` | 130,188,567 | `3ebcef0f07c332b550c2610b2ca1de0ed97e85344eb51721b0fc35bf8fce937e` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126404096-4096.hdf5` | 128,543,880 | `2992b60a698e2e1ca327c986a1f4adb669f1d66bc33be8b3995b30237d343018` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126105088-4096.hdf5` | 130,171,386 | `475dc24380e6b6ebdd4bd709d1836ad2f904f00f98048c89786c8f5e448d4ede` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126105088-4096.hdf5` | 126,774,593 | `d68825df1cb1e83440c1e94bab9e02ac6768084ed27ecbf81397ca96ca1cd2c9` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126100992-4096.hdf5` | 130,170,442 | `8ca06790a285100067489de735e2ed4c48807ef958ea7d263fb505d13f4a9c5c` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126100992-4096.hdf5` | 126,751,658 | `aee3cae8eca6b9b58d4ee623509b123c333fc34fd4299f0c42c13bfeb94ff442` |
| GW150914 배경 | `H-H1_LOSC_4_V1-1126092800-4096.hdf5` | 130,165,016 | `785bc5dac65d1b4b96ce0be8090f7eef60fea012b8c5cf0e62f4acc91f9f0fd7` |
| GW150914 배경 | `L-L1_LOSC_4_V1-1126092800-4096.hdf5` | 126,681,538 | `74101a6f420c41659814f74f9adc28303ecb3dc4f24aa4d3f145565a8462460e` |
| GW151226 배경 | `H-H1_LOSC_4_V1-1135140864-4096.hdf5` | 130,074,527 | `45296a0e71119d77e78d0de0c0877d7cc025f03afb033df19087538dcd756d51` |
| GW151226 배경 | `L-L1_LOSC_4_V1-1135140864-4096.hdf5` | 127,956,979 | `e563b7b9e0844178190a21e25e5e4f3cecb4aa3f071afbc6277dd26225325d4a` |
| GW151226 배경 | `H-H1_LOSC_4_V1-1135144960-4096.hdf5` | 130,074,623 | `49eb9b9c3bc7d4a3df446093a5c7917761a9152a2c08f86be9174c4fe0f67e77` |
| GW151226 배경 | `L-L1_LOSC_4_V1-1135144960-4096.hdf5` | 127,952,343 | `913bdb308ae493f2ae178b3a6cdf3e259366df9a485cc38119ce33c0f12fc017` |
| GW151226 배경 | `H-H1_LOSC_4_V1-1135153152-4096.hdf5` | 130,064,645 | `25b2fa11e8fb211763250833f772becc555876ddcaafd6e5daaef95d8803b21f` |
| GW151226 배경 | `L-L1_LOSC_4_V1-1135153152-4096.hdf5` | 128,090,436 | `7e2c039affd2ddf13fe495f3875821b08111446b65ef168196f5a662de88677f` |
| GW151226 배경 | `H-H1_LOSC_4_V1-1135157248-4096.hdf5` | 130,081,606 | `e9402851ca1e34925dad463f827a50af35131d0e84e8ba050eeb657bef303bd3` |
| GW151226 배경 | `L-L1_LOSC_4_V1-1135157248-4096.hdf5` | 128,196,479 | `22d094fceac66cf3da441c84f1c031b4a4d7099f1d14cd36fafdbf5a5d12cc84` |
| GWTC-1 사후분포 표본 | `GW150914_GWTC-1.hdf5` | 7,026,464 | `21fc54453e73df01340fcfe11fa69972699e68f4196a04fbb04e5e8907e874cb` |
| GWTC-1 사후분포 표본 | `GW151226_GWTC-1.hdf5` | 10,603,104 | `03aa4afaba1ce0de82d5f32b4f4e2f48129da797990cb5ae70fe98d09b3e4a23` |

읽어서 확인한 것 (`results/diag_stage5.json`, `results/event_gw151226.json`):

- GW151226 H1 4096초 파일은 **앞 1,925초가 NaN**이다(DATA 비트 꺼짐, H1이 관측 상태가 아니었다). L1은 전체가 유효하다. 사건(파일 시각 2047.6초)은 H1 자료가 시작되고 약 123초 뒤다. 두 검출기가 함께 유효한 분석 시간은 2,112초다.
- 일괄 파일(`LOSC_4_V1`)과 사건 파일(`GWOSC_4KHZ_R1`)은 겹치는 3,321초에서 비트 단위로 같지는 않다. 최대 절대 차이 1.5 × 10⁻²⁰(H1), 1.7 × 10⁻²⁰(L1). 차이의 스펙트럼은 20~1024 Hz에서 자료 스펙트럼의 10⁻⁹(H1)~10⁻⁸(L1) 수준이라 분석 대역에서는 같은 자료로 볼 수 있다. 차이가 생긴 이유(저장 정밀도 등)는 확인하지 못했다.
- 일괄 파일의 구조(`strain/Strain`, `quality/simple/DQmask`)는 사건 파일과 같다.

## 문헌

사전 등록에 쓴 문헌은 arXiv에서 PDF로 받아 읽었다(저장소에 넣지 않음): 1602.03837, 1602.03839, 1602.03840, 1811.12907, 1608.01940, 0907.0700, 1606.04855, 1606.04856(뒤의 둘은 5단계).

## 인용 문구

GWOSC 안내에 따라 아래 문헌 중 하나를 인용하고 문구를 붙인다.

- R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13 (2021) 100658.

> This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. This material is based upon work supported by NSF's LIGO Laboratory which is a major facility fully funded by the National Science Foundation, as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan.

## 소프트웨어

- 템플릿 IMRPhenomD와 교차 확인에 LALSuite(lalsuite 8.0.2, GPL-2.0-or-later)를 쓴다.
- 5단계의 매개변수 추정에 bilby 2.8.2(MIT)와 dynesty 2.1.5(MIT)를 쓴다(dynesty 3.2.0은 bilby 2.8.2와 맞지 않았다). CNN 비교(선택)에는 PyTorch를 쓴다. matched filter, PSD, TaylorF2, 안테나 패턴, 동시 발생·time slide는 이 저장소의 코드다.
