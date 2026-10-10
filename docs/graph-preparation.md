# 고정 FlyWire 그래프 준비

#3 / 단계 A2 / AC-02의 공개 CLI 경계다. NES·ROM·GPU 없이 기존 Python 3.12 환경에서 실행한다.

```bash
uv sync --locked --extra graph
uv run --locked --extra graph flycade fetch-graph-data
uv run --offline --locked --extra graph flycade prepare-graph \
  --config configs/graph-visual.json --output .flycade/graphs/visual-001
uv run --offline --locked --extra graph flycade inspect-graph \
  .flycade/graphs/visual-001 --cache .flycade/data/v783
```

`fetch-graph-data`만 네트워크를 사용한다. 최초 약 880MB를 받으며 연결 속도에 따라 오래 걸릴 수 있다. 검증된 캐시는 네트워크 접근 없이 재사용한다. 원본과 임시 집계 DB를 위해 수 GB 이상의 여유 디스크를 확보한다. `--cache`로 캐시 위치를 바꿀 수 있다. 다운로드는 임시 파일에서 완료·검증 후 게시하며 손상된 기존 캐시는 자동으로 덮어쓰지 않는다. 오류에 표시된 파일을 별도로 이동하고 다시 다운로드한다.

`prepare-graph`는 매번 로컬 원본 해시를 확인하며 네트워크를 사용하지 않는다. 출력 디렉터리가 존재하면 거부한다. 새 경로로 다시 준비한 결과의 `graph_sha256`을 비교해 재현성을 확인한다. 이전 결과를 확인할 때는 `inspect-graph`를 사용한다. `--cache`를 주면 원본까지 검증하며 생략하면 완성된 그래프 artifact만 검증한다. 외부 다운로드 파일을 이미 보유한 경우에도 같은 이름으로 캐시에 놓고 위 명령을 사용할 수 있다.

## 고정 출처와 선택 근거

기본 원본 명세는 `src/flycade/data/flywire-v783.json`에 있다. 각 파일의 URL, 실제 확인한 SHA-256, 크기, 필요한 열, 라이선스 근거와 인용을 담는다.

| 파일 | 용도와 필요한 열 | 고정 근거 |
| --- | --- | --- |
| `proofread_connections_783.feather` | `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`; 방향 뉴런 쌍×영역별 집계 | [Zenodo 783.0](https://zenodo.org/records/10676866), 게시 MD5 `f48f972d262323a102aed49af1396b8a`도 대조 |
| `proofread_root_ids_783.npy` | 1차원 정수 배열; 고립 뉴런까지 포함하는 proofread 모집단 | 같은 archive, 게시 MD5 `e0e6c19732fd8c7a4e39a2d170105421`도 대조 |
| `Supplemental_file1_neuron_annotations.tsv` | `root_id`, `cell_type`, `super_class`; 유형별 선택과 원본 대응 | [annotation v2.1.0](https://github.com/flyconnectome/flywire_annotations/releases/tag/v2.1.0), 커밋 `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0` |

집계 파일을 선택해 전체 시냅스 원본·영상·mesh를 내려받지 않는다. v2.1.0은 v783 기반 릴리스이며 현재 서비스의 최신 annotation을 섞지 않는다. 공개 HTTPS 접근을 사용하고 계정·토큰은 필요 없다. GitHub annotation 파일은 upstream의 별도 SHA-256 배포가 없어 고정 커밋에서 받은 바이트의 SHA-256을 기록했다.

연결과 뉴런 목록의 라이선스는 [Zenodo metadata](https://zenodo.org/api/records/10676866)의 CC BY 4.0이다. annotation v2.1.0 저장소에는 별도 LICENSE 파일이 없다. 연관 논문·보충 자료의 [CC BY 4.0 명시](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446831/)를 근거로 기록하며 저장소 자체에 라이선스 파일이 있는 것으로 표현하지 않는다. 파생 그래프를 공유할 때 출처, 저자, [라이선스](https://creativecommons.org/licenses/by/4.0/), 아래 전처리 변경 내용을 함께 보존한다.

인용: FlyWire Consortium, *FlyWire Whole-brain Connectome Connectivity Data*, 783.0, [doi:10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866); Dorkenwald et al. (2024), [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y); Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5).

## 선택과 계산 의미

`configs/graph-visual.json`은 L1·Mi1·T4a 유형의 실제 뉴런을 선택하는 시작 실험이다. 입력은 L1, 출력은 T4a이며 수천 개 규모를 탐색하는 출발점일 뿐 최종 학습 크기나 학습 성능을 확정하지 않는다. 선택된 뉴런 사이의 유도 부분 그래프를 보존하며 노드를 축약하지 않는다. 이 출력군에 NES 행동을 연결하는 것은 후속 정책의 공학적 readout이고 초파리 운동 뉴런 재현이 아니다.

각 `selection`, `inputs`, `outputs`는 `{"cell_types": ["L1"]}` 또는 `{"root_ids": ["720575940..."]}` 중 하나다. ID는 항상 문자열로 지정한다. 입력·출력은 선택된 proofread 뉴런의 비어 있지 않은 서로 겹치지 않는 부분집합이어야 한다. 존재하지 않는 ID·유형, 모호한 숫자 ID, 알 수 없는 설정은 실패한다. annotation과 proofread 목록 차이는 보고한다.

전처리 순서:

1. 원본 SHA-256과 스키마·정수 ID·양수 시냅스 수·proofread endpoint를 검증한다.
2. **전체 영역에서 동일 방향 뉴런 쌍을 먼저 합산**한다. 그 뒤 `min_synapses`(기본 5)를 적용한다. A→B와 B→A는 별개다.
3. `self_loops`는 `drop`(기본) 또는 `keep`이다. 설정에 따른 전체 eligible 연결에서 선택한 뉴런 간 연결을 추출한다.
4. ID 숫자 오름차순으로 내부 인덱스를 만들고, `(pre_index, post_index)` 순서로 edge를 정렬한다. seed를 기록하지만 무작위 선택은 없다.
5. weight는 시냅스 수의 float64 변환이며 정규화하지 않는다. 부호는 비음수 구조 가중치이고, 신경전달물질만으로 흥분·억제를 확정하지 않는다.
6. 방향성 BFS로 각 입력의 출력 도달 여부, 각 출력의 입력으로부터 도달 여부와 한 예시 경로를 보고한다. 비율은 각각 전체 지정 입력 수·출력 수를 분모로 한다. 경로가 전혀 없으면 `graph_disconnected`로 실패한다. 일부만 도달하는 경우 해당 비율을 명시한다.

전체 연결은 Arrow record batch로 읽고 DuckDB 디스크 DB에서 집계한다. DuckDB 메모리 예산은 512MB, thread는 1개다. 전체 N×N 배열이나 전체 Python 객체 그래프를 만들지 않는다. 선택된 부분 그래프에 대해서만 NumPy edge 배열과 BFS adjacency를 만든다. 따라서 매우 큰 선택은 더 많은 RAM을 사용한다. DuckDB 예산은 프로세스 RSS 상한이 아니며 Arrow, Python과 OS 매핑 비용은 별도다.

## 출력 계약

| artifact | 의미 |
| --- | --- |
| `nodes.json` | 내부 `index` ↔ 원본 `root_id` 문자열의 무손실 대응, cell type·super class, input/output 표시 |
| `edge_index.npy` | int64, shape `(2, E)`, 첫 행 pre, 둘째 행 post; 방향 보존 |
| `syn_count.npy` | int64, shape `(E,)`, 영역 합산 후 시냅스 수 |
| `weight.npy` | float64, shape `(E,)`, 현재 정규화 없는 구조 가중치 |
| `config.json` | 기본값까지 해석한 정규화 설정 |
| `sources.json` | 고정 원본 명세와 세 파일의 SHA-256·출처·권리·인용 |
| `report.json` | 원본·집계 후·임계값/self-loop 처리 후·사용 통계, 제외량과 비율, 연결성, 가정, 패키지·Python·구현 해시 |
| `manifest.json` | 형식 버전, artifact별 SHA-256, 그래프·설정·출처 명세 해시 |
| `measurement.json` | 다운로드를 제외한 전처리 시간, 해당 Linux CLI 프로세스의 peak RSS, 측정 범위 |

그래프 해시는 정해진 순서의 `nodes.json`, `edge_index.npy`, `syn_count.npy`, `weight.npy` 파일명과 SHA-256을 결합한 문자열의 SHA-256이다. 파일 형식·도구 버전은 lock으로 고정한다. 실행 시간과 peak RSS는 매 실행 달라지므로 계산용 그래프 해시에서 제외하되 별도 해시로 무결성을 확인한다. 원본 파일의 해시는 `sources.json`, 해석 설정의 해시는 `manifest.json`에 있다.

오류는 JSON `error.code`와 종료 코드 2로 반환한다. `artifact_missing`은 파일 누락, `artifact_corrupt`는 해시 불일치, `download_failed`는 다운로드 실패, `output_exists`는 덮어쓰기 방지, `graph_disconnected`는 입출력 경로 부재다. 스키마·설정 오류는 `preparation_failed`이며 구체적인 메시지를 제공한다. 그래프는 임시 디렉터리에서 완성한 뒤 게시한다.

## 검증

```bash
uv run --extra graph mypy src
uv run --extra graph pytest tests/test_graph_cli.py
uv run --extra graph pytest
```

fixture는 공개 `--source-manifest` 옵션으로 로컬 테스트 원본을 지정한다. 이 옵션으로 대체한 파일은 기본 FlyWire 데이터라는 보장이 없으므로 그 명세의 `dataset`·`release`·출처를 확인한다. 테스트는 CLI 프로세스와 문서화된 출력 artifact만 관찰한다. 실제 데이터의 통계·재구성 해시·자원 측정은 [A2 검증 기록](validation/A2.md)에 별도로 남긴다. 이 검증은 정책 학습·NES 성공·AC-12의 1시간 운용을 증명하지 않는다.
