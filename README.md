# mapaditer 자동 도시 맵 생성기 (초안 v1)

## 설치 (GitHub)
기존 mapaditer 저장소의 최상위에 이 ZIP의 `.github/workflows/auto-maps.yml`과 `map_auto/` 폴더를 추가한다. 기존 `ctlist.html`, `edit.html`은 유지한다. `.github`은 숨김 폴더일 수 있다. GitHub Actions에서 **Generate city maps (review drafts)** → Run workflow → 도시 코드/범위/도시 수 입력. 첫 시험은 1개 권장. 작업 결과 Artifacts의 `city-map-drafts`를 다운로드한다. 자동 커밋/배포하지 않으므로 기존 지도는 보존된다. 다음 실행은 다른 도시 코드를 지정한다. 한 번에 모든 도시를 공용 API로 대량 처리하는 용도가 아니다.

## 로컬 실행 / 기존 에디터 패치
저장소 최상위에서 Python 3.10 이상으로:
```
python -m pip install -r map_auto/requirements.txt
python map_auto/patch_editor.py edit.html
python map_auto/test_build.py
python map_auto/build.py --codes 도시코드 --limit 1
```
패치 스크립트는 확인한 원본 문자열이 맞지 않으면 파일을 바꾸지 않고 종료한다. 백업은 `edit.html.bak`에 저장. 패치한 HTML을 확인 후 GitHub에 업로드. 기존 게임 엔진에서도 51(부두)과 8(도로)을 색상 및 통행 가능 타일에 추가해야 한다. 자동 제작 파일은 기존 숫자 배열 형식을 유지한다.

## 좌표 및 범위
`ctlist.html`의 citiesData 배열을 읽는다. 실제 배열 내용 전체를 이 환경에서 실행 검증하지 못했다. 읽기 실패/좌표 모호성/공항·해안이 잘리는 경우에는 `map_auto/cities.example.json`을 참고해 도시별 좌표와 km 범위를 직접 정한 `cities.json`을 만들고 `--cities cities.json`으로 실행. 기본 지도는 북쪽 위, 실제 25km 정사각형(1칸 50m), 게임에서는 1칸 32px. 도시에 따라 km 값을 변경할 수 있다. 북극 가상 중앙마을과 세계맵 도시 연결은 이 프로그램의 생성 대상이 아니며, 날짜변경선 및 극지 경계는 별도 처리 필요.

## 데이터 / 검증
- OSM Overpass 벡터 + AWS Terrain Tiles의 고도 자료. 공용 서버는 소규모 작업용이므로 대량 처리 시 지역 OSM 파일 또는 별도 제공자 이용. 도시별 lat/lon을 지정하면 지오코딩을 생략한다. 공용 Nominatim은 소규모 일회성 요청만 캐시하여 순차 수행. 서비스 오류는 강제로 재시도하지 않는다.
- `CODE.html`: 기존 에디터가 읽을 수 있는 정수 배열, `CODE.png`: 미리보기, `CODE.json`: 검증 리포트, `CODE-compare.png`: 원자료 기반 지형(왼쪽)과 게임 맵(오른쪽) 비교. 실패는 `.error.json`.
- 산은 실제 지형의 경사/상대 고도 기준의 근사치. 도로는 지정된 주요 도로와 일부 생활도로이며, 교량/터널 미제작으로 단절 가능. 호수는 강으로 표현. 항구와 부두 연결/터미널 공간 검사, 공항 40x16 검사, 수역 및 산의 무단 덮어쓰기 검사. 실제 부두 도형이 없으면 알려진 항구 위치에 표준 부두를 두고 리포트에 기록. 모든 항구/공항을 완벽히 식별하는 것은 아직 보장되지 않는다.
- 이 버전은 규칙 기반 제작기이지 AI 모델을 호출해 지도 이미지를 검토하거나 반복 학습하는 에이전트는 아니다. 독립 지도 비교, 정확한 도시 경계 및 공항/항구 전수 검증은 추가 필요. 생성물은 검토용으로만 저장한다.
- 부두 표기 코드 5-1 / 저장 51, 도로 8. 시청 4는 기존 호환용이며 자동 생성하지 않는다.
- OSM attribution: https://www.openstreetmap.org/copyright (ODbL 및 파생 데이터 관련 조건 확인). DEM: https://registry.opendata.aws/terrain-tiles/ (각 원자료의 출처/라이선스 확인). 지도 게시 전에 해당 출처와 라이선스 적용 확인.
- 실제 API 호출 및 브라우저/GitHub Actions 실행은 이 대화 환경에서 검증하지 못했다. `test_example`은 합성 지형 테스트일 뿐 실제 도시 지도 아님.
