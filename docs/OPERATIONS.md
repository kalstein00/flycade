# Flycade 운영 안내

이 도구는 실제 NES를 실행하면서 커넥톰 정책을 PPO로 학습하고, 로컬 브라우저에서 플레이·실제 정책 입력·부분 회로 활동·저장 영상과 평가 결과를 관찰한다. 정책은 픽셀만 받는다. 게임 내부 상태는 보상과 결과 판정에만 쓰며 화면의 활성값은 생물학적 전압이 아니다. 게임 클리어 시간이나 특정 성과는 보장하지 않는다.

## 설치와 데이터 준비

저장 위치는 WSL Linux 파일시스템을 사용한다. 프로젝트 README의 환경 설치, ROM 등록, 그래프 데이터 준비 절차를 먼저 완료한다. ROM은 사용자가 확보한 파일을 등록하며 프로젝트가 배포하지 않는다.

```bash
uv sync --locked --extra graph --extra train
uv run flycade diagnose
uv run flycade catalog
uv run flycade register-rom /path/to/rom.nes
uv run flycade inspect
uv run flycade inspect-graph .flycade/graphs/visual-a2-final-001
```

`diagnose`에서 WSL·CUDA·여유 메모리·ffmpeg 상태를 확인한다. CPU fixture는 자동 검사용으로, 실제 NES·GPU 검증을 대신하지 않는다. 그래프 출처·고정 데이터 버전·ID 대응과 준비 명령은 [README](../README.md), [A2 검증](validation/A2.md)에 있다. 최초 다운로드와 준비를 마친 뒤에는 `uv run --offline --locked --extra graph --extra train`을 사용해 네트워크 없이 실행할 수 있다.

## 새 Run과 관찰

```bash
uv run --offline --locked --extra graph --extra train flycade train \
  --graph .flycade/graphs/visual-a2-final-001 \
  --output reports/my-run \
  --training-config configs/training-small.json \
  --initial-evaluation-config configs/evaluation-small.json
```

선택 프리셋은 `configs/preset-small.json`이다. 100,000 updates의 명시적 예산, rollout32, epochs2, 관측 최대3Hz, 초기 평가와1,000 updates마다 고정 CPU 평가, 누적 학습시간600초마다 자동 저장을 사용한다. 평가 중 학습은 안전 경계에서 기다리고 GUI에 모드를 표시한다. 최적화가 다시 시작되면 누적 진도를 이어간다. `--output`은 새 경로여야 한다.

다른 WSL 터미널에서:

```bash
uv run --offline --locked --extra graph --extra train flycade live reports/my-run
```

출력한 localhost 주소를 Windows 브라우저에서 연다. 게임(45%)·회로(30%)·입력/행동(25%)과 학습·저장 상태를 함께 볼 수 있다. 뉴런 클릭 또는 키보드 Enter로 상세를 선택하고, 연결 필터·확대·입력 프레임을 바꿀 수 있다. 회로는 제한된 부분 그래프이며 전체 신경망의 해부학적 위치를 보여 주는 그림이 아니다. 표본과 이력은 제한된 관찰 정보이며 학습 상태가 아니다. 브라우저를 닫아도 학습은 계속된다. 다시 접속하면 최신 표본을 받는다.

## 저장하고 종료하기

```bash
uv run --offline --locked --extra graph --extra train flycade status reports/my-run
uv run --offline --locked --extra graph --extra train flycade save reports/my-run --stop
```

`save --stop`의 완료와 학습 프로세스 종료를 확인한 뒤 WSL 또는 PC를 종료한다. 영상 인코더 정리 중에는 “저장 완료 · 종료 정리 중”이 표시될 수 있다. 정상 종료 후에는 “종료 완료”가 된다. 학습 터미널에서 Ctrl+C도 안전 경계 저장·종료 요청이다. 강제 종료는 마지막 정상 저장 이후의 진도를 잃을 수 있다. 요청이 시간 초과되면 프로세스를 강제로 끄기 전에 `status`에서 대기·저장·완료·실패를 확인한다.

## 다음 세션에서 재개와 복구

```bash
uv run --offline --locked --extra graph --extra train flycade resume reports/my-run
```

새 프로세스는 모델·optimizer·난수·누적 진도·저장/평가 스케줄을 복원한다. 게임 에피소드는 시작 상태에서 새로 시작하며, 게임 중간의 에뮬레이터 프레임을 복원하는 기능은 아니다. Run은 유지되고 session ID가 바뀐다. 기본 최근 정상 체크포인트3개와 보호본을 유지한다. 최신본이 손상되면 검증 가능한 이전 정상본으로 복구하며 유실된 update·전이 수와 거부한 후보를 CLI/GUI에서 보여 준다.

소스·패키지·Python/CUDA·ROM·state·그래프·설정 식별자가 다르면 재개를 거부한다. 부분 로드나 최신 온라인 다운로드로 우회하지 않는다. 해당 Run을 만든 코드와 로컬 환경·데이터 백업을 복원한다. 문서만 바뀐 커밋은 Python 소스 해시를 바꾸지 않는다. [상세 복구 계약](validation/T09.md).

## 평가와 영상 내역

라이브 화면의 **평가 비교**에서 동일 protocol의 초기·최신·최고 평가를 고른다. 짧은 압축 VP9 영상은 기본12fps·CRF45·무음이며 원본 고화질 영상을 저장하지 않는다. **평가 관찰·이력**에서는 영상 재생·일시정지·탐색에 맞는 실제 입력·회로·행동 표본을 확인한다. 표본이 없거나 정리된 구간은 데이터 없음으로 표시한다. 학습 영상 전체 목록은 다음 명령으로 연다.

```bash
uv run --offline --locked --extra graph --extra train flycade history reports/my-run --serve
uv run --offline --locked --extra graph --extra train flycade evaluations reports/my-run
uv run --offline --locked --extra graph --extra train flycade evaluate reports/my-run \
  --snapshot latest --protocol reports/my-run/initial-evaluation-protocol.json
```

동일 조건은 게임·ROM/state·입력/행동/보상·평가 seed와 횟수를 protocol ID로 구별한다. 서로 다른 protocol을 동일 조건으로 합치지 않는다. 최고본은 완료율, 다음 평균 거리 순으로 선택하고 동률이면 기존 최고본을 유지한다. 단독 평가도 별도 프로세스의 고정 가중치로 동작하며 학습 상태를 갱신하지 않는다. 작은 고정 seed 결과는 일반화나 클리어 능력의 증명이 아니다.

## 실험 분기와 예산 변경

정지한 Run에서:

```bash
uv run --offline --locked --extra graph --extra train flycade checkpoints reports/my-run
uv run --offline --locked --extra graph --extra train flycade pin-checkpoint reports/my-run <checkpoint-id>
uv run --offline --locked --extra graph --extra train flycade branch reports/my-run \
  --checkpoint <checkpoint-id> --output reports/branch
uv run --offline --locked --extra graph --extra train flycade resume reports/branch
uv run --offline --locked --extra graph --extra train flycade warm-start reports/my-run \
  --checkpoint <checkpoint-id> --output reports/new-experiment \
  --training-config configs/training-small.json
uv run --offline --locked --extra graph --extra train flycade extend-budget reports/my-run --updates 200000
```

`branch`는 전체 상태를 이어받는 독립 Run이며 부모 원본을 복사해 보호한다. `warm-start`는 호환되는 전체 가중치만 가져오고 optimizer·난수·진도는 새로 시작한다. 기본 새 Run에는 사전학습·교사가 없다. `extend-budget`은 같은 Run의 총 예산만 늘리고 누적 카운터·고정 학습률 스케줄은 유지한다. 같은 Run에 두 학습 writer를 실행할 수 없다.

CNN 비교는 새 Run에 `--model-kind cnn`을 추가한다. CNN에는 “회로 해당 없음”을 표시하지만 실제 입력·행동과 영상 내역은 유지한다. `compare-runs RUN_A RUN_B`로 초기 대비 거리·사망·완료율, 예산·설정 차이와 파라미터 수를 확인한다. [실제 작은 비교와 미개선 결과](validation/T14.md).

## 공간 관리와 오프라인 사용

```bash
uv run --offline --locked --extra graph --extra train flycade storage reports/my-run
uv run --offline --locked --extra graph --extra train flycade storage reports/my-run --apply
```

기본 최근 평가3개·학습 영상10개·로그별8MiB를 유지하며 초기·최고·최신·수동 pin의 보호 규칙을 함께 적용한다. 저장 경계와 종료 시 자동 정리된다. 보호본이 많으면 공간이 계속 필요하므로 여유 공간을 확인한다. GUI **저장 공간**에 설정·결과·오류가 표시된다. 삭제 실패 시 쓰기 권한과 공간을 복구한 뒤 다시 정리한다. 전체 Run 백업은 정상 종료 후 수행한다. Windows 파일 접근, 보존 예외와 문제 해결은 [T15 안내](validation/T15.md)에 있다.

로컬 파일만 준비되어 있으면 재개·평가·영상/회로 이력은 인터넷 없이 동작한다. 브라우저는 로컬 서버에 연결해야 한다. “관측 없음”은 학습의 첫 표본을 기다리는 상태인지, 중지한 Run인지 확인한다. “영상 없음”은 파일이 보존 정책으로 정리되었거나 누락되었는지 확인한다. 실시간 회로와 고정 평가/과거 영상의 모드 표기를 구분한다.

## 실제 재시작 인수

[최종 인수 기록](validation/T16.md)에서 자동 검증과 사용자 수동 확인을 구분한다. 실제 WSL 종료/PC 재시작 및 Windows 화면 확인은 사용자 확인 전까지 미완료이다. 정상 저장 후 Windows PowerShell에서 `wsl --shutdown`을 수행하거나 PC를 재시작하고, WSL을 다시 연 뒤 동일 프로젝트·Run에서 `resume`과 `live`를 실행한다. 재개 전후 checkpoint ID·session ID·누적 updates·추가 갱신, Windows 브라우저의 재접속·저장 상태·뉴런 선택·평가 영상 탐색을 기록한다.

최종 인수 Run의 재시작에는 `scripts/verify_restart.py <restart-handoff.json>`을 사용할 수 있다. 이 도구는 boot ID 변경과 저장해 둔 정상본을 먼저 확인한 뒤3 updates만 재개·저장한다. 실제 재시작이 없으면 작업을 시작하지 않는다. Windows 화면 확인 결과는 별도로 남긴다.
