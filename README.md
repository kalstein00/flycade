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
- `checkpoints/`, `latest.json`, `final.pt`: 안전 경계의 전체 학습 상태와 정상본 포인터. `final.pt`는 마지막 정상본의 편의 링크다. 새 프로세스 재개는 아래 절을 따른다.

예산은 `updates × rollout_steps` 환경 전이다. 예산 종료 시 진행 중인 에피소드는 완료로 세지 않는다. 에뮬레이터는 같은 프로세스에 하나만 있으며 종료·예외·Ctrl+C에서 닫는다. Ctrl+C는 현재 rollout과 optimizer 갱신 완료 후 저장·종료를 요청한다. 예외로 갱신이 실패하면 기존 정상 체크포인트를 보존한다.

작은 CPU fixture와 실제 NES·RTX 5090의 분리된 결과, 라이브러리 선택 근거와 남은 불확실성은 [A3 검증 기록](docs/validation/A3.md)을 참고한다. 이 결과는 플랫폼 갱신 성공이며 초기 정책 대비 행동 성능 개선을 뜻하지 않는다.

## 저장 후 종료, 재개, 플레이 영상

새 Run의 `--updates`는 **전체 학습 예산**이다. `--stop-after-updates`는 이번 세션에서 실행할 갱신 수만 제한한다. 학습 도중 **Ctrl+C**를 눌러도 현재 rollout과 optimizer 갱신을 끝낸 뒤 안전하게 저장·종료한다. 완료 JSON의 `status: saved` 또는 `completed`, `environment_closed: true`를 확인한 뒤 PC를 종료한다.

```bash
uv run --offline --locked --extra graph --extra train flycade train \
  --graph .flycade/graphs/visual-a2-final-001 --output reports/my-run \
  --updates 1000 --stop-after-updates 10
uv run --offline --locked --extra graph --extra train flycade resume reports/my-run
```

재개는 같은 Run에 새 세션을 만들며 모델·optimizer·난수·누적 진도를 이어받는다. **게임은 새 에피소드**로 시작한다. 전체 예산이 끝난 Run의 재개는 거부한다. 예산 연장·과거 분기는 후속 범위다. 자동 저장은 기본 누적 학습600초마다 수행한다. `latest.json`이 가리키는 정상 체크포인트를 사용하며, 이전 정상본도 `checkpoints/`에 남긴다. 기존 A3의 단순 `final.pt` snapshot은 전체 재개 형식과 달라 지원하지 않는다.

학습 플레이는 기본으로 **VP9 WebM, 12fps, CRF 45, 원본 게임 크기, 무음**으로 녹화한다. `ffmpeg`의 `libvpx-vp9` 인코더가 필요하다(Ubuntu: `sudo apt install ffmpeg`). 실제 게임 프레임 5장마다 1장을 스트리밍하고, 최대 60초씩 나눠 저장한다. 녹화 오류는 보고서의 `recording_error`로 알리고 학습 저장은 계속한다. 완성된 영상은 자동 삭제하지 않으므로 디스크 사용량을 확인한다.

```bash
# JSON 내역
uv run --offline --locked --extra graph --extra train flycade history reports/my-run
# Windows 브라우저에서 http://127.0.0.1:8765/ 열기
uv run --offline --locked --extra graph --extra train flycade history reports/my-run --serve
```

목록에서 영상을 선택하고 재생·일시정지·탐색한다. 화면을 닫아도 저장된 파일은 유지된다. 새 녹화는 새로고침하면 나타난다. 이 화면은 학습 기록 영상이며 실시간 관찰·정책 평가·회로 활동 재생은 아니다. 서버는 loopback에만 바인딩하고 학습 상태를 쓰지 않는다.

재개 호환성은 Python 소스 전체의 SHA-256, lock·패키지·Python/CUDA 버전, Run 설정 및 graph/ROM/state/초기 모델 해시가 동일한 경우로 한정한다. Git 커밋 ID는 출처 기록이며 문서만 바뀐 커밋은 허용된다. 코드 변경 후 부분 가중치 로드로 우회하지 않는다. 체크포인트는 Python/NumPy RNG를 포함한 로컬 신뢰 파일만 로드한다. 상세 계약과 실제 검증은 [A4 기록](docs/validation/A4.md)을 참조한다.

브라우저 테스트 준비: `uv run playwright install chromium`. 자동 검증: `uv run --offline --locked --extra graph --extra train pytest -q`.

## 고정 정책 평가와 초기 기록

본 학습 전 비교 기준을 남기려면 새 Run에서 `--initial-evaluation-config configs/evaluation-small.json`을 지정한다. 초기 정책을 먼저 고정하고 **별도 CPU 프로세스의 평가가 끝난 뒤 첫 rollout**을 시작한다. 그동안 CLI와 Run 보고서에 학습 대기를 표시한다. 초기 평가 실패 시 학습을 시작하지 않는다.

```bash
uv run --offline --locked --extra graph --extra train flycade train \
  --graph .flycade/graphs/visual-a2-final-001 --output reports/evaluated-run \
  --updates 100 --initial-evaluation-config configs/evaluation-small.json

uv run --offline --locked --extra graph --extra train flycade evaluate reports/evaluated-run \
  --snapshot latest --protocol reports/evaluated-run/initial-evaluation-protocol.json
```

기본 예시는 seed 11·22·33의 **확률적 행동 선택**, 에피소드당 최대 1,800 emulator frame, 첫 에피소드 영상 최대 15초다. 학습 게임의 더 짧은 종료 제한은 유지한다. 결과에는 개별 거리(pixels)·분포 요약·완료 횟수/평가 횟수·사망/무진행 등 종료 사유·정규화하지 않은 게임 보상과 Run/snapshot/protocol 식별자를 기록한다. 짧은 표본의 차이를 학습 성공으로 단정하지 않는다.

평가 protocol을 별도로 생성할 수도 있다. 게임·ROM/state·전처리·행동·횟수·seed·행동 선택 방식을 고정한다. 결과 JSON의 `evaluation_id` 두 개를 사용해 CLI에서 같은 protocol의 초기/후속 결과를 비교한다. 다른 protocol이면 비교를 거부한다.

```bash
uv run --offline --locked --extra graph --extra train flycade evaluation-protocol reports/evaluated-run \
  --output reports/evaluation-protocol.json --seeds 11,22,33 --max-frames 1800 --video-seconds 15
uv run --offline --locked --extra graph --extra train flycade evaluate reports/evaluated-run \
  --snapshot initial --protocol reports/evaluation-protocol.json
uv run --offline --locked --extra graph --extra train flycade compare-evaluations reports/evaluated-run \
  <초기-evaluation-ID> <후속-evaluation-ID>
```

`--snapshot initial`, `latest`, 또는 snapshot UUID를 선택한다. `snapshots/`의 정책 전용 불변 파일을 사용하므로 optimizer나 전체 resume 파일 없이도 평가할 수 있다. 처음부터 평가 형식으로 생성한 Run이 필요하며 구형 Run snapshot의 자동 변환은 제공하지 않는다. 평가 프로그램은 학습 상태를 쓰지 않고 결과를 `evaluations/<ID>/`에 추가한다. 같은 모델을 다시 평가해도 기존 결과를 덮어쓰지 않는다.

대표 **결정론적 관전**은 별도 protocol로 `--mode deterministic --seeds 11`을 지정한다. 가장 확률이 큰 행동을 고르며 seed 하나만 허용한다. seed만 바꾼 결정론적 반복을 다양한 평가로 세지 않는다. `evaluate --realtime`은 이 모드에서 가능한 범위의 게임 속도(약 60 emulator fps)에 맞춘다. 화면 관전은 `history --serve`에서 완성된 짧은 영상을 재생한다. 확률적 평가·결정론적 관전·학습 기록은 화면에서 구별되며 평가 영상에 snapshot/protocol을 표시한다.

평가는 기본 CPU·환경 1개·에피소드 순차 실행이다. `--device cuda`는 여유가 있을 때 명시한다. RAM/VRAM이 부족하면 학습을 Ctrl+C로 안전 저장·종료한 다음 평가하고 `resume`한다. 평가 명령 자체가 실행 중인 trainer를 중단하지는 않는다. 소요 시간·프로세스 peak RSS·인코더 peak RSS·PyTorch peak GPU allocation을 평가 보고서에 기록한다. GPU 전체 VRAM이나 동시 프로세스 합계의 측정값은 아니다. 평가용 녹화에 실패하면 결과를 실패로 표시하고 비교에서 제외한다.

## 실시간 게임·회로 관찰

학습은 기본으로 최대 **3 Hz**의 동기화 표본을 내보낸다. 다른 터미널에서 같은 Run의 관찰 서버를 열고 Windows 브라우저로 표시된 주소에 접속한다.

```bash
uv run --offline --locked --extra graph --extra train flycade live reports/my-run
# 기본 주소: http://localhost:8766 · 포트 변경: --port 8770
```

게임·전처리 프레임 묶음·실제 회로 활성·행동 분포는 **실제 행동을 선택한 같은 forward**의 표본이다. 회로는 해부학 좌표가 아닌 고정 구조도이며 색은 최종 노드 상태 벡터의 산술평균이다. 표시 범위와 버전은 회로 아래에서 펼쳐 확인한다. 선택 행동과 확률 최댓값을 구별하며, 보상과 RAM 진행 거리는 **행동 적용 후 전이 결과**로 따로 표시한다.

관찰 중에 브라우저를 닫거나 서버를 종료해도 trainer는 계속 실행된다. 다시 접속하면 최신 표본으로 복구한다. 표본이 오래되거나 연결이 끊기면 마지막 화면을 보존하고 지연·연결 상태를 표시한다. 학습 종료 후에는 종료·저장 상태를 표시한다. 관측이 없는 Run에는 데이터 없음이 표시된다. 서비스는 loopback 읽기 전용이며 외부 CDN이나 인터넷 연결이 필요 없다.

`train`과 `resume`의 `--observe-hz 2|3|4|5`로 최대 표본 빈도를 선택하고 `--observe-hz 0`으로 끈다. 이것은 게임 속도나 학습 예산을 바꾸지 않는다. `live/latest.json`은 최신 완성 표본 하나만 유지하고 전송 대기도 1개로 제한한다. 느린 소비자에게 누적 재생하지 않는다. 학습 기록 영상은 `history --serve`에서 본다. 새 평가 영상은 **평가 관찰·이력**에서 저장된 회로 표본과 함께 재생한다.

실제 NES·RTX 5090 측정과 브라우저 검증 범위는 [T06 기록](docs/validation/T06.md)을 참고한다. 기존 Run은 코드 해시가 다르면 재개가 거부되므로 이번 소스로 새 Run을 만들어 사용한다.

## 실측된 작은 기본 프리셋 (A5)

실제 NES·RTX5090·WSL에서 관찰/평가/저장/새 프로세스 재개를 포함해10분25초 검증했다. 설정과 수치 예산은 `configs/preset-small.json`, 근거와 재실행 절차는 [A5 기록](docs/validation/A5.md)에 있다. 본 학습은 다음 고정 예산과 초기 평가로 시작한다.

```bash
uv run --offline --locked --extra train flycade train \
  --graph .flycade/graphs/visual-a2-final-001 --output reports/daily-001 \
  --training-config configs/training-small.json \
  --initial-evaluation-config configs/evaluation-small.json --observe-hz 3
```

이 자원 측정은1시간 안정성이나 게임 클리어를 보장하지 않는다. 자동 저장은 기본600초이며, 초기 평가를 활성화한 Run은 기본1,000 updates마다 같은 조건으로 순차 CPU 평가한다.


## 매일 저장하고 이어 학습하기

```bash
# 별도 터미널: 저장 후 계속 / 저장 후 종료 / 상태 조회
uv run --offline --locked --extra train flycade save reports/daily-001
uv run --offline --locked --extra train flycade save reports/daily-001 --stop
uv run --offline --locked --extra train flycade status reports/daily-001
# 다음 실행: 같은 Run, 새 세션과 새 게임 에피소드
uv run --offline --locked --extra train flycade resume reports/daily-001
```

학습 JSON의 `autosave_seconds`(기본600)로 주기를 설정한다. 요청은 다음 rollout/optimizer 완료 경계에서 저장하며 CLI·브라우저에서 접수/대기/저장/완료와 마지막 복구 시점을 확인한다. `save --stop`은 환경과 인코더 종료까지 기다린다. 자동 저장 주기와 누적 예산은 재개할 때 보존한다. `--wait-seconds 0`으로 요청만 보내거나 대기 시간을 바꿀 수 있다. 실제 실행·실패 처리·매일 절차는 [T08 운영 기록](docs/validation/T08.md)을 참조한다.

## 손상·강제 종료 후 복구

`resume RUN`은 최신 정상본을 검증하고, 손상됐으면 공개 확인 기록이 있는 이전 정상본을 차례로 검사한다. 같은 Run의 새 세션에서 이어가며 되돌린 update·전이와 후보별 오류를 CLI/관찰 화면에 표시한다. 이전 로그는 보존한다. 검증 가능한 정상본이 없으면 일치하는 로컬 백업을 복원하거나 새 Run을 만들어야 한다.

기본 최근3개 정상본 외에 원하는 전체 학습 상태는 `pin-checkpoint RUN UUID`로 보호한다. `checkpoints RUN`에서 목록을 확인하며 pin 변경은 저장 후 종료한 상태에서 수행한다. 초기 정책과 최고본 참조는 보호한다. 공개 순서·오류 주입·보존 범위·실제 복구 증거는 [T09 기록](docs/validation/T09.md)을 참조한다.

과거 저장본의 전체 학습 상태를 새 Run으로 이어가려면 `branch PARENT --checkpoint UUID --output CHILD` 후 `resume CHILD`를 실행합니다. `warm-start PARENT --checkpoint UUID --output CHILD --training-config CONFIG.json`은 가중치만 가져와 새 optimizer·진도·스케줄로 학습합니다. `extend-budget RUN --updates TOTAL`은 정지한 같은 Run의 총 update 예산을 명시적으로 늘립니다. 브라우저 ‘Run 정보’에서 부모 계보와 예산 변경을 확인할 수 있습니다. [계약·검증·실행 안내](docs/validation/T10.md).

라이브 회로에서 노드(Enter/Space 가능)나 뉴런 표를 선택하면 원본 ID·알려진 종류·모델 활성 평균을 볼 수 있습니다. 표시군/연결 필터·확대·입력 프레임·짧은 활동 이력은 브라우저 안에서만 바뀝니다. `live RUN_A --run RUN_B`로 명시한 Run 사이를 전환합니다. [관찰 범위와 이력 상한·검증](docs/validation/T11.md).

주기 평가와 초기·최근·최고본 비교는 [T12 운영·검증](docs/validation/T12.md)을 참고하세요. `live RUN` 화면의 **평가 비교**에서 저장된 짧은 영상과 시드별 성과를 확인합니다.

평가 영상 탐색과 실제 회로·정책 입력·행동의 동기 재생, 고정 평가 최신 관측은 [T13 시간 대응·상한·검증](docs/validation/T13.md)을 참고하세요.

동일 조건의 별도 CNN Run은 `train --model-kind cnn`으로 시작합니다. `compare-runs RUN_A RUN_B`는 모델별 초기 대비 거리·완료율·사망 변화와 예산/설정 차이를 보고합니다. [T14 비교 조건과 실제 미개선 결과](docs/validation/T14.md)를 참고하세요.

Run별 저장 공간과 보존 정책은 `storage RUN`으로 확인합니다. 최근 평가·영상·로그를 제한하고 초기·최고·사용자 보존본을 보호하는 방법과 오프라인 절차는 [T15 안내](docs/validation/T15.md)를 참고하세요.
