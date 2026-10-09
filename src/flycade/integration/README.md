# flycade-smb-1

Stable-Retro 1.0.1의 `SuperMarioBros-Nes-v0` / `Level1-1`을 선택한다. `release.json`은 설치 배포물에서 직접 조회한 버튼 순서, ROM payload SHA-1, 기본 integration/state SHA-256이다. 시작 state 자체는 vendoring하지 않으며 해당 릴리스에서 검증 후 로컬 등록에 복사한다. 코어 바이너리 SHA-256도 매 실행 기록한다.

상류 `scenario.json`의 xscrollLo 보상은 하위 바이트 순환과 왕복을 충분히 처리하지 않으며 모든 목숨을 잃어야 끝난다. 여기서는 빈 scenario와 명시적 RAM 필드 `data.json`을 사용하고, `GameEnv`가 매 프레임 진행·종료를 계산한다. Stable-Retro reset이 빈 info를 반환하므로 어댑터는 reset 직후 공개 `GameData.lookup_all()`을 조회한다.

주소와 루틴 의미는 [SMB disassembly](https://github.com/MrWint/smb-dis/blob/master/smbdis.asm)의 심볼에 대응한다. 이는 해당 ROM의 1-1용 계약이며 다른 게임/월드에 일반화하지 않는다.

| 필드 | 주소 | 의미 |
| --- | --- | --- |
| player_page / player_x | 0x6d / 0x86 | 플레이어 절대 수평 좌표의 페이지/하위 바이트 |
| screen_page / screen_x | 0x71a / 0x71c | 화면 왼쪽 경계; 플레이어 위치에 다시 더하지 않음 |
| engine | 0x0e | GameEngineSubroutine: 5 종료, 6 목숨 상실, 8 제어, 11 사망 |
| lives | 0x75a | 현재 플레이어의 여분 목숨 |
| world / level | 0x75f / 0x75c | 0부터 세는 월드/레벨 |
| time | 0x7f8 | 3자리 게임 시간 |
| timer_expired | 0x759 | 시간 만료 플래그; 완료 후 시간 점수 환산과 구분 |
| mode | 0x770 | 게임 운영 모드; 1은 게임 |
| player_y_high | 0xb5 | 수직 페이지; 진단용 |

page 경계를 넘는 이동과 스크롤, 왕복, reset은 합성 trace와 실제 실행으로 구분해서 확인한다. 깃발 종료, 사망, 시간 만료 등의 실제 증거는 `docs/validation/A1.md`에 기록한다. 합성 fixture만으로 주소 의미가 검증됐다고 주장하지 않는다.

참고: [공식 integration 형식](https://stable-retro.farama.org/integration/), [설치와 ROM import](https://stable-retro.farama.org/getting_started/). ROM과 게임 state는 저장소에 배포하지 않는다.
