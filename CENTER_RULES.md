# 중심·시설 이동 규칙 v2

이전 center_v1을 대체한다. map_auto/ai_agent.py, build.py, center_policy.py를 교체한다. 테스트 파일은 선택. 기존 PBF 소스와 Actions는 유지한다.

- 기본 중심: 기존 도시 대표 좌표. 직접 지정한 map_lat/map_lon은 고정.
- 검색 근거가 있는 필수 유명 지역이 범위 밖일 때만 최초 중심으로부터 최대 5km 이동. 주요 강 통과 금지, 기존 필수 지역 보존. 지도 크기와 축척 유지.
- 공항·항구는 중심 이동 사유가 아니다. 시설 자체가 중요하고 주변이 교외 저중요도라는 근거가 있는 경우, 최종 지도 범위에 맞지 않는 실제 시설 자체를 원래 좌표에서 최대 7km 이동할 수 있다. 실제 시설이 없으면 생성하지 않는다. 이동 경로가 주요 강을 교차하면 거부한다.
- 공항은 전체 40×16 또는 16×40 직사각형이 평지에 들어가는 가장 가까운 후보를 탐색한다. 항구는 실제 바다에 접한 가까운 해안 후보 중 터미널·부두 연결이 가능한 곳을 탐색한다. 행정구역 일치 여부는 자동 보장하지 않으며 현 단계는 거리와 지형을 우선한다. 배치 불가 시 미해결 기록.
- 중심 판단은 center-policy.json, 시설 실제 좌표·배치 좌표·이동 거리(m)는 각 결과 JSON의 metrics.facility_relocations에 기록한다.
- 지도 조사와 검수는 Gemini 및 Wikipedia, 지형은 PBF/DEM에 의존한다. 실제 네트워크 실행은 미검증. 기존 공항 선택은 대표 여객공항 하나를 유지한다.

테스트: `cd map_auto && python -m unittest test_center_policy test_facility_relocation test_quality test_model_fallback`
