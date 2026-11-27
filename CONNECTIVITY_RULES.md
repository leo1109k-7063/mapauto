# 도로·강 연결 패치 v2

center_v2 기반. `map_auto/build.py`, `map_auto/ai_agent.py`, `map_auto/pbf_source.py`를 교체하고 `map_auto/connectivity.py`를 추가한다. center_v2의 `center_policy.py`도 포함했으므로 이전 패치를 적용하지 않았다면 함께 교체한다. 기존 워크플로 및 API 키 유지.

- 상하좌우 4방향 연결 기준. 대각선 래스터화 틈을 최소 보정한다.
- AI가 선정한 주요 도로/하천을 실제 PBF 원자료의 도로·하천 경로로 연결한다. 주요 도로의 연결 램프와 실제 교량도 사용한다. 추가 경로는 1칸 폭으로 연결하며 전체 원자료를 그대로 표시하지 않는다.
- 도로가 강과 교차하면 어두운 노란색 `#B8860B` 코드 81 우선. 일반 도로는 코드 8, 노란색 `#FFFF00` 유지. 코드 81은 도로 및 강 연결 검사 모두에 포함한다. 물이 가려진 칸은 JSON `river_under_road`와 HTML `riverUnderRoad`에 0부터 시작하는 `y*500+x` 인덱스로 저장한다. 기존 `gridData` 형식은 그대로이다. 81 자체가 교차 타일이다. 기존 에디터/게임에서 81을 색상표 및 통행 가능한 도로 목록에 추가해야 한다.
- 도로가 바다, 산, 시설을 덮지 않는다. 실제 연결 경로가 없는 고립 컴포넌트는 임의로 연결하지 않고 최대 연결망 외의 부분을 초안에서 제외하고 수치를 기록한다. 상당한 누락(최대 고립 부분 40칸 이상 또는 전체 선택의 5% 이상, 최소 10칸)이 있으면 AI 승인 금지 및 재검토. 실제로 분리된 수계 또는 섬 도로가 있는 도시에서는 이 정책이 지형을 생략할 수 있으므로 검토해야 한다.
- `metrics.connectivity`의 roads/rivers 및 logical_river_components, visible_road_components를 확인한다. 도로 및 강이 있는 경우 각각 1이어야 한다. 도로 아래 강의 연결은 시각적으로 끊어져 보이더라도 허용한다.
- 중심 최대 5km, 시설 자체 최대 7km 및 주요 강 건너 이동 금지 규칙은 유지한다.

테스트: `cd map_auto && python -m unittest test_connectivity test_center_policy test_facility_relocation test_model_fallback test_quality`. 실제 Gemini/PBF 재실행은 하지 않았다.
