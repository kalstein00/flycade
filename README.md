# flycade

초파리 커넥톰 기반 레트로 게임 학습 실험실.

현재 구현은 **T01 환경 연결, T02 그래프 준비, T03 실제 커넥톰 정책의 짧은 PPO 학습**이다. CLI는 JSON 결과를 출력하며 성공은 exit 0, 준비/검증 실패는 exit 2, Ctrl+C는 exit 130이다.

## 설치와 확인

검증한 조합은 Ubuntu 22.04.5 / WSL2, CPython 3.12.9, Stable-Retro 1.0.1이다. 다른 배포판의 성공이나 학습 라이브러리 호환성을 뜻하지 않는다. Python은 `.python-version`, 의존성 및 다운로드 해시는 `uv.lock`으로 고정한다. `uv`가 설치된 WSL Linux 디렉터리에서 실행한다.

```bash
uv sync --locked
uv run flycade diagnose
uv run flycade catalog
uv run mypy src
uv run pytest
```

진단은 CPU, Windows 물리 RAM, Linux/WSL RAM과 cgroup 제한, 디스크, GPU별 전체/가용 VRAM, 드라이버, Python/패키지를 구분한다. Windows RAM은 PowerShell interop가 없으면 `null`과 오류 사유를 남긴다. GPU 인식은 sparse backward나 optimizer 검증이 아니다. 학습 후보는 선택적 `train` extra의 PyTorch 2.10.0이며 SB3는 사용하지 않는다. NES 데모 자체는 CPU로 동작한다.

WSL CUDA는 **Windows NVIDIA 드라이버**를 사용한다. WSL 안에 Linux 디스플레이 드라이버를 설치하지 않는다. 드라이버를 갱신해야 한다면 Windows에서 갱신하고 WSL을 다시 시작한다. [NVIDIA WSL 안내](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [Microsoft WSL 메모리 설정](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)을 참고한다.

## 사용자 ROM 등록과 데모

```bash
uv run flycade register-rom 'roms/Super Mario Bros. (Japan, USA).nes'
uv run flycade inspect
uv run flycade demo --output reports/demo-001 --steps 120 --actions 0,1,2,3,4,5,6
```

ROM은 직접 준비한 압축되지 않은 iNES 파일을 사용한다. 프로그램은 ROM을 다운로드하지 않는다. 선택한 integration의 SHA-1과 비교한 뒤 `.flycade/`에 로컬 복사한다. 원본과 복사본, save-state, 실행 보고서는 Git에서 제외한다. `--home`으로 등록 위치를 지정할 수 있다. 기존 등록과 보고서는 덮어쓰지 않는다.

`catalog`는 설치한 릴리스의 실제 게임/state/버튼 목록 및 코어를 확인한다. `inspect`는 등록된 ROM, state와 integration 파일을 재검사한다. SHA-1은 Stable-Retro와 동일하게 NES의 16바이트 헤더를 제외하고, SHA-256은 전체 ROM을 검사한다. 실행마다 ROM/state/integration/코어 해시를 보고한다.

| 행동 ID | 버튼 |
| --- | --- |
| 0 | 대기 |
| 1 | RIGHT |
| 2 | RIGHT + A (점프) |
| 3 | RIGHT + B (달리기) |
| 4 | RIGHT + B + A |
| 5 | A |
| 6 | LEFT |

`--actions` 목록을 전이마다 순환한다. 긴 재현 입력은 `--actions-file path.json`의 정수 배열로 전달한다. 기본 action repeat는 4프레임이다. `--steps`는 전이 수 상한이며 게임 사건이 먼저 발생하면 즉시 끝난다. 실행 중 모든 내부 프레임에서 사건을 확인한다. 한 프로세스에 에뮬레이터 하나만 열며 정상 종료·예외·Ctrl+C에서 닫는다.

출력 디렉터리에는 다음을 남긴다.

- `report.json`: 실제 환경, 해시, 버튼 순서, 전처리/보상 설정, reset과 마지막 전이, 검증 상태.
- `transitions.jsonl`: 행동·버튼·보상·위치/스크롤·종료 사유·관측 해시. RAM 값은 이 진단 정보에만 포함한다.
- `reset-screen.png`, `last-screen.png`: 실제 게임 화면.
- `reset-policy.npy`, `last-policy.npy` 및 `*-policy.png`: 정책에 전달하는 픽셀 배열과 마지막 누적 프레임의 미리보기.

설정 JSON은 `--config path.json`으로 지정한다. 생략한 항목은 기본값을 사용하며 최종 해석값을 보고서에 기록한다.

```json
{"width":84,"height":84,"color":"RGB","frame_stack":4,"action_repeat":4,"progress_reward":1.0,"survival_reward":0.0,"death_reward":-10.0,"completion_reward":100.0,"no_progress_frames":600,"max_frames":18000}
```

정책 관측은 uint8 `(누적 프레임, 높이, 너비, 채널)`이다. RGB 또는 L 회색조(1채널), Pillow bilinear resize를 사용한다. 누적은 오래된 순서이며 reset에서는 같은 시작 프레임을 채운다. 각 전이의 마지막 프레임을 추가하며 RAM은 관측에 섞지 않는다. 학습/평가는 동일한 `GameEnv` 공개 reset/step 계약을 재사용할 수 있다.

## 보상과 종료

World 1-1의 절대 위치는 플레이어 page × 256 + x이다. 화면 스크롤은 별도로 보고하며 더하지 않는다. low byte의 255→0 순환을 page가 보완한다. reset 위치를 기준점으로 삼아 이전 최고 위치를 넘어선 거리만 보상하므로 왕복과 reset 자체에 가짜 진행 보상이 없다.

보상은 에뮬레이터 프레임 단위로 누적한다. 정상 프레임의 새로운 진행 1픽셀당 +1, 생존 0, 사망/게임 시간 초과 -10, 완료 +100이 기본값이다. 사건이 발생한 프레임에는 진행 보상을 추가하지 않는다. 이 계수는 T01 실험 기본값이며 학습 최적값이 아니다.

| 사유 | terminated | truncated | 의미 |
| --- | --- | --- | --- |
| `death` | true | false | 사망 루틴 또는 목숨 감소; respawn을 기다리지 않음 |
| `completion` | true | false | 1-1 깃발 슬라이드가 끝나 종료 루틴(engine 5)에 진입 |
| `game_timeout` | true | false | 게임 timer-expired 플래그; 단순 시간 숫자 0과 구분 |
| `no_progress` | false | true | 기본 600프레임 동안 새로운 최고 위치 없음 |
| `external_limit` | false | true | 설정한 프레임/CLI 전이 예산 종료 |
| `position_discontinuity` | false | true | 한 프레임 16픽셀 초과 이동; 보상 없이 검토 요구 |
| `unexpected_game_state` | false | true | 예상 밖 월드/레벨/게임 모드 |
| `backend_end_unclassified` | false | true | 분류하지 못한 코어 종료; 성공으로 집계하지 않음 |

무진행과 외부 상한은 게임 내부 사건이 아닌 수집 제한이다. learner는 종료와 truncation을 구분하고, 계약 위반 trace는 학습에 사용하기 전에 검토해야 한다. 완료 판정은 깃발 통과 시점이며 성 입장 영상 전체 재생을 뜻하지 않는다.

## 오류 해결과 검증

- `rom_missing`: 원본 경로 또는 `--home`이 맞는지 확인한다.
- `rom_format` / `rom_hash_mismatch`: 압축을 해제하고 ROM 판본을 확인한다. 헤더만 바뀌어도 등록 후 전체 해시 검사는 실패한다.
- `integration_missing` / `release_mismatch`: `uv sync --locked`로 선택 릴리스를 복원한다.
- `artifact_missing` / `artifact_hash_mismatch`: 기존 파일을 임의 변경하지 말고 새로운 `--home`에 재등록한다.
- `already_registered` / `output_exists`: 검증에는 `inspect`, 새 실험에는 새로운 출력 경로를 사용한다.
- `start_state_mismatch` / `game_contract_violation`: 보고서와 마지막 trace를 확인한다. 다른 게임 state나 순간이동을 진행 보상으로 처리하지 않는다.
- GPU probe 실패: Windows 드라이버와 `/usr/lib/wsl/lib/nvidia-smi`를 확인한다. CPU NES 실행 범위는 별도로 보고한다.

기본 테스트는 합성 전이 계약과 CLI를 검사한다. 실제 NES 검증은 사용자가 준비한 ROM으로 명시적으로 활성화한다.

```bash
FLYCADE_TEST_ROM='roms/Super Mario Bros. (Japan, USA).nes' uv run pytest
```

설치와 ROM 등록 이후 `uv run --offline --locked flycade ...`로 오프라인 실행할 수 있다. 최초 설치에는 패키지 접근이 필요하다.

실측 결과와 A1/AC-01 판정은 [검증 기록](docs/validation/A1.md), RAM 필드 출처와 한계는 [integration 설명](src/flycade/integration/README.md)을 참고한다.

## 고정 커넥톰 데이터 준비 (A2)

FlyWire v783·annotation v2.1.0의 실제 부분 그래프를 준비한다. 출처 해시, ID 대응, 영역 합산 후 임계값, 방향성 입출력 도달성, 전처리 RAM·시간을 함께 기록한다.

```bash
uv sync --locked --extra graph
uv run --extra graph flycade fetch-graph-data
uv run --offline --locked --extra graph flycade prepare-graph --config configs/graph-visual.json --output .flycade/graphs/visual-001
uv run --offline --locked --extra graph flycade inspect-graph .flycade/graphs/visual-001 --cache .flycade/data/v783
uv run --extra graph pytest
```

설정·출력 형식·출처·라이선스·오류 대응은 [데이터 준비 안내](docs/graph-preparation.md), fixture와 실제 데이터의 구분된 검증 결과는 [A2 기록](docs/validation/A2.md)을 참고한다. `graph` extra를 설치하지 않은 환경에서는 그래프 통합 테스트를 건너뛴다.

## 새 Run과 짧은 PPO 학습 (A3)

```bash
uv sync --locked --extra graph --extra train
uv run --offline --locked --extra train flycade train \
  --graph .flycade/graphs/visual-001 --output reports/train-001 \
  --device cuda --updates 16 --rollout-steps 64 --seed 7
```

등록된 NES 1-1과 실제 그래프를 사용하는 새 Run을 만든다. `--output`은 새 디렉터리여야 한다. CUDA가 없으면 오류로 끝나며 CPU로 자동 전환하지 않는다. `--device cpu --fixture`는 합성 에뮬레이터를 명시적으로 사용하는 자동 검사 옵션이며 실제 NES/GPU 검증으로 기록하지 않는다. 그래프 준비 이후 학습에는 `graph` extra가 필요하지 않다.

픽셀 인코더 → 고정 방향성 COO 그래프 → 출력 노드 readout → 7개 행동의 확률 분포와 가치 추정 순서다. 기본 노드 상태 차원 8, 관측마다 2회 전파, 입력군에 반복 주입, tanh 활성화이며 시간 순환 상태는 없다. 연결은 도착 뉴런별 synapse count 합으로 나눠 고정하고 양수 구조 가중치로 취급한다. 생물학적 흥분/억제 부호를 추론하지 않는다. 인코더·행동 및 가치 readout만 학습하며 사전학습·교사 정책은 없다. 출력군과 입력군은 분리하고 주어진 전파 횟수 안에 방향성 경로가 있어야 한다.

`--training-config path.json`은 학습 설정, `--config path.json`은 앞서 설명한 게임 설정이다. 명시한 CLI의 `--updates`, `--rollout-steps`, `--seed`가 JSON보다 우선한다. 학습 기본값은 다음과 같으며 최종 값은 Run에 기록한다.

```json
{"updates":2,"rollout_steps":32,"epochs":2,"state_dim":8,"propagation_steps":2,"seed":7,"learning_rate":0.0003,"gamma":0.99,"gae_lambda":0.95,"clip_ratio":0.2,"entropy_coefficient":0.01,"value_coefficient":0.5,"max_grad_norm":0.5}
```

PPO-Clip은 전체 rollout을 한 batch로 epoch마다 Adam 갱신한다. GAE는 실제 종료의 bootstrap을 0으로 하고 truncation은 마지막 관측 가치로 bootstrap하되 reset 사이의 advantage를 연결하지 않는다. GPU 메모리 사용량은 rollout과 그래프 크기에 따라 늘어난다. 이 작은 A3 후보의 규모는 장기 학습 프리셋 확정이 아니다.

Run 출력은 다음과 같다.

- `run.json`, `graph/`: Run/세션 UUID, seed, 게임·ROM·state·그래프·원본 출처, 모델·학습 설정, 코드 파일 해시·Git 기준점·lock 해시·패키지 버전, 사용 그래프 사본.
- `initial.pt`: 첫 행동 이전의 무학습 모델. 읽기 전용으로 보존하고 종료 시 해시를 다시 검사한다.
- `control-pixels.npy`, `report.json`: 고정 입력에서 동일 정책의 연결 제거 전후 행동 확률, 유효 갱신·전체 파라미터 변화·시간·자원·환경 종료 결과.
- `transitions.jsonl`: 행동 확률·선택 행동·보상·관측 해시·종료 사유와 내부 프레임 수. RAM 진단은 로그에만 있고 정책 입력에 섞지 않는다.
- `updates.jsonl`: 완료된 PPO 갱신별 누적 전이·프레임·완료 에피소드·성과·loss·entropy·gradient·비정상 수치·전이/초. `updates`는 rollout 학습 완료 횟수, `optimizer_steps`는 Adam 갱신 수다.
- `final.pt`: rollout과 해당 optimizer epoch가 모두 끝난 안전 경계의 모델·optimizer·RNG·진도·manifest. 아직 새 프로세스 resume·원자 저장·손상 복구 계약을 제공하는 체크포인트 형식은 아니다.

예산은 `updates × rollout_steps` 환경 전이다. 예산 종료 시 진행 중인 에피소드는 완료로 세지 않는다. 에뮬레이터는 같은 프로세스에 하나만 있으며 종료·예외·Ctrl+C에서 닫는다. Ctrl+C는 즉시 중단하고 실패/부분 보고서를 남기므로 `final.pt` 저장을 보장하지 않는다. 안전한 저장 요청과 재개는 후속 T04 범위다.

작은 CPU fixture와 실제 NES·RTX 5090의 분리된 결과, 라이브러리 선택 근거와 남은 불확실성은 [A3 검증 기록](docs/validation/A3.md)을 참고한다. 이 결과는 플랫폼 갱신 성공이며 초기 정책 대비 행동 성능 개선을 뜻하지 않는다.
