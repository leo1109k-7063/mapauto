#!/usr/bin/env python3
"""OSM vector + Terrarium DEM -> 500x500 game map. No API key needed."""
import argparse, json, math, re, time, io, sys, os
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage as nd
from PIL import Image, ImageDraw
import requests, json5
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union, polygonize
from shapely import make_valid

N=500; LAND,SEA,RIVER,MOUNTAIN,PORT,AIRPORT,ROAD,PIER=0,1,2,3,5,6,8,51
COLORS={0:'#4CAF50',1:'#1A237E',2:'#00BCD4',3:'#1B5E20',4:'#FFFFFF',5:'#BDBDBD',6:'#424242',8:'#FFFF00',51:'#795548'}
ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'.map-cache'; CACHE.mkdir(exist_ok=True)
SESSION=requests.Session(); SESSION.headers['User-Agent']='mapaditer-game-map/1.0 (https://github.com/leo1109k-7063/mapaditer)'
CROSS=nd.generate_binary_structure(2,1)

def cached_json(key,url,params=None,post=None):
    p=CACHE/(key+'.json')
    if p.exists(): return json.loads(p.read_text())
    r=SESSION.post(url,data=post,timeout=180) if post else SESSION.get(url,params=params,timeout=90)
    r.raise_for_status(); data=r.json()
    if isinstance(data,dict) and data.get('remark'): raise RuntimeError(data['remark'])
    p.write_text(json.dumps(data,ensure_ascii=False)); return data

def geocode(city):
    if city.get('lat') is not None and city.get('lon') is not None: return float(city['lat']),float(city['lon'])
    q=f"{city['name']}, {city.get('country','')}"; key='geo-'+re.sub(r'[^\w-]','_',q)
    # Respect public Nominatim limit; coordinates in cities.json are strongly preferred for batches.
    time.sleep(1.2)
    result=cached_json(key,'https://nominatim.openstreetmap.org/search',{'q':q,'format':'jsonv2','limit':5,'accept-language':'en'})
    candidates=[x for x in result if x.get('addresstype') in ('city','town','village','municipality','borough','administrative','hamlet') or x.get('category')=='place']
    if not candidates: raise RuntimeError('도시 좌표 확인 필요: '+q)
    if len(candidates)>1 and float(candidates[0].get('importance',0))-float(candidates[1].get('importance',0))<.04: raise RuntimeError('동명 도시 가능성: cities.json에 lat/lon 지정: '+q)
    return float(candidates[0]['lat']),float(candidates[0]['lon'])

class Frame:
    def __init__(self,lat,lon,km):
        self.lat,self.lon,self.km=lat,lon,km; self.m=km*1000/N; self.h=km*500
        crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs')
        self.forward=Transformer.from_crs(4326,crs,always_xy=True); self.back=Transformer.from_crs(crs,4326,always_xy=True)
        self.domain=box(0,0,N,N)
    def xy(self,lon,lat):
        x,y=self.forward.transform(lon,lat); return x/self.m+N/2,N/2-y/self.m
    def bbox(self):
        a=np.linspace(-self.h-3000,self.h+3000,21); x=np.r_[a,a,np.full(21,a[0]),np.full(21,a[-1])]; y=np.r_[np.full(21,a[0]),np.full(21,a[-1]),a,a]; lon,lat=self.back.transform(x,y)
        if max(lon)-min(lon)>180 or max(abs(lat))>85: raise RuntimeError('극지/날짜변경선 경계 별도 처리 필요')
        return min(lat),min(lon),max(lat),max(lon)

def osm(frame,key):
    bbox=','.join(f'{v:.6f}' for v in frame.bbox())
    selectors=['way[natural=coastline]','nwr[natural=water]','nwr[waterway=riverbank]','way[waterway~"^(river|stream|canal)$"]','nwr[water=sea]','way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]','nwr[aeroway=aerodrome]','way[aeroway=runway]','nwr[harbour][harbour!=no]','nwr[landuse=port]','nwr[industrial=port]','nwr[amenity=ferry_terminal]','nwr[man_made=pier]','nwr[man_made=quay]']
    query='[out:json][timeout:150];('+''.join(s+'('+bbox+');' for s in selectors)+');out geom;'
    return cached_json('osm-'+key,os.getenv('OVERPASS_URL','https://overpass-api.de/api/interpreter'),post={'data':query})['elements']

def geom(e,f):
    def line(g): return LineString([f.xy(p['lon'],p['lat']) for p in g]) if len(g)>=2 else None
    if e['type']=='node': return Point(f.xy(e['lon'],e['lat']))
    if e['type']=='way':
        g=line(e.get('geometry',[])); return g
    outer=[]; inner=[]
    for m in e.get('members',[]):
        l=line(m.get('geometry',[]))
        if l is not None: (inner if m.get('role')=='inner' else outer).append(l)
    if not outer: return None
    p=list(polygonize(unary_union(outer)))
    if p:
        a=unary_union(p); holes=list(polygonize(unary_union(inner))) if inner else []
        return make_valid(a.difference(unary_union(holes))) if holes else make_valid(a)
    return unary_union(outer)

def asarea(g):
    if g.geom_type=='LineString' and g.is_ring: return make_valid(Polygon(g))
    return g

def mask(g):
    im=Image.new('1',(N,N)); d=ImageDraw.Draw(im)
    def draw(p):
        if p.is_empty: return
        if p.geom_type=='Polygon':
            d.polygon(list(p.exterior.coords),fill=1)
            for h in p.interiors: d.polygon(list(h.coords),fill=0)
        elif p.geom_type in ('LineString','LinearRing'): d.line(list(p.coords),fill=1,width=1)
        elif p.geom_type=='Point': d.point((p.x,p.y),fill=1)
        elif hasattr(p,'geoms'):
            for x in p.geoms: draw(x)
    draw(g); return np.asarray(im,dtype=bool)

def coast_mask(lines,notes):
    # OSM coastline direction: land left, sea right. Y is inverted in the game grid.
    domain=box(0,0,N,N); pieces=[]; segments=[]
    for l in lines:
        if l.geom_type!='LineString': continue
        for a,b in zip(l.coords,list(l.coords)[1:]):
            if a!=b: segments.append((a,b,LineString([a,b])))
        clip=l.intersection(domain)
        if not clip.is_empty: pieces.append(clip)
    if not pieces: return np.zeros((N,N),bool)
    from shapely.strtree import STRtree
    tree=STRtree([s[2] for s in segments]); faces=list(polygonize(unary_union([domain.boundary,*pieces])))
    if len(faces)<=1 and any(p.length>1 for p in pieces): notes.append('해안선이 닫히지 않음: 원자료/경계 검토 필요')
    sea=np.zeros((N,N),bool)
    for p in faces:
        pt=p.representative_point(); a,b,_=segments[int(tree.nearest(pt))]
        cross=(b[0]-a[0])*(pt.y-a[1])-(b[1]-a[1])*(pt.x-a[0])
        if cross>0: sea|=mask(p)
    return sea

def dem(frame):
    # AWS public elevation tiles, Terrarium encoding. Resample to local north-up grid.
    xx,yy=np.meshgrid(np.arange(N)+.5,np.arange(N)+.5); lon,lat=frame.back.transform((xx-N/2)*frame.m,(N/2-yy)*frame.m)
    z=max(8,min(13,round(math.log2(156543*math.cos(math.radians(frame.lat))/frame.m))))
    s=256*2**z; px=(lon+180)/360*s; py=(1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*s
    tx=np.floor(px/256).astype(int); ty=np.floor(py/256).astype(int); result=np.empty((N,N),np.float32)
    for x,y in set(zip(tx.ravel(),ty.ravel())):
        p=CACHE/f'dem-{z}-{x}-{y}.png'
        if not p.exists():
            r=SESSION.get(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',timeout=60); r.raise_for_status(); p.write_bytes(r.content)
        a=np.asarray(Image.open(p).convert('RGB'),np.float32); h=a[:,:,0]*256+a[:,:,1]+a[:,:,2]/256-32768; k=(tx==x)&(ty==y)
        result[k]=h[(py[k]%256).astype(int),(px[k]%256).astype(int)]
    return nd.gaussian_filter(result,1.1)

def line_width(t,m):
    try: width=float(re.match(r'[\d.]+',t.get('width','')).group())
    except (AttributeError,ValueError): width={'river':max(9*m,80),'canal':max(4*m,25),'stream':max(m,5)}.get(t.get('waterway'),m)
    return max(1,min(35,width/m))

def airport(grid,center,horizontal):
    w,h=(40,16) if horizontal else (16,40); x,y=center; best=None
    for dy in range(-20,21,2):
        for dx in range(-20,21,2):
            l=int(round(x-w/2))+dx; t=int(round(y-h/2))+dy
            if l<0 or t<0 or l+w>N or t+h>N: continue
            v=grid[t:t+h,l:l+w]; bad=np.count_nonzero(v!=LAND)
            if bad: continue
            score=dx*dx+dy*dy
            if best is None or score<best[0]: best=(score,l,t)
    if best is None: return None
    _,l,t=best; grid[t:t+h,l:l+w]=AIRPORT; return [l,t,w,h]

def port(grid,center,piergeom,notes):
    # Candidate shore within 25 cells of mapped port/pier. All edits rollback if no valid terminal/connection.
    sea=grid==SEA; land=grid==LAND; coast=land & nd.binary_dilation(sea,structure=CROSS)
    ys,xs=np.where(coast)
    if not len(xs): return False
    cx,cy=center; distance=(xs-cx)**2+(ys-cy)**2; idx=np.argmin(distance)
    if distance[idx]>25**2: return False
    x,y=int(xs[idx]),int(ys[idx]); old=grid.copy()
    # Choose a contiguous 10x5 or 5x10 land footprint at the shore; smaller terminal-compatible fallback.
    choices=[]
    for w,h in ((10,5),(5,10),(6,4),(4,6),(4,2),(2,4)):
        for l in range(x-w+1,x+1):
            for t in range(y-h+1,y+1):
                if l<0 or t<0 or l+w>N or t+h>N: continue
                if np.all(grid[t:t+h,l:l+w]==LAND):
                    choices.append((-(w*h),abs(l+w/2-cx)+abs(t+h/2-cy),l,t,w,h))
        if choices: break
    if not choices: return False
    _,_,l,t,w,h=min(choices); grid[t:t+h,l:l+w]=PORT
    pm=np.zeros_like(sea)
    if piergeom is not None:
        p=asarea(piergeom); pm=mask(p if p.geom_type in ('Polygon','MultiPolygon') else p.buffer(1.25,cap_style=2)) & sea
    adjacent=sea & nd.binary_dilation(grid==PORT,structure=CROSS)
    if not adjacent.any(): grid[:]=old; return False
    ay,ax=np.where(adjacent); j=np.argmin((ax-cx)**2+(ay-cy)**2); sx,sy=int(ax[j]),int(ay[j]); pm[sy,sx]=True
    # Retain mapped pier shape and join its nearest point to the harbor, if possible.
    py,px=np.where(pm); dist=(px-sx)**2+(py-sy)**2; far=int(np.argmax(dist))
    if dist[far]>0:
        nearx,neary=int(px[far]),int(py[far]); pm|=mask(LineString([(sx,sy),(nearx,neary)]).buffer(1.25,cap_style=2)) & sea
    else:
        # Existing mapped harbor without pier geometry: reserve a short abstract pier into sea.
        dy,dx=sy-y,sx-x
        if abs(dx)+abs(dy)!=1:
            opts=[(0,1),(0,-1),(1,0),(-1,0)]; dy,dx=max(opts,key=lambda q: sum(0<=sy+q[0]*k<N and 0<=sx+q[1]*k<N and sea[sy+q[0]*k,sx+q[1]*k] for k in range(1,7)))
        pm|=mask(LineString([(sx,sy),(sx+dx*6,sy+dy*6)]).buffer(1.15,cap_style=2)) & sea
        notes.append('실제 항구 위치에 부두 자료 부족: 표준 부두로 단순화')
    labels,n=nd.label(pm,structure=CROSS); keep=np.unique(labels[adjacent & pm]); keep=keep[keep>0]; pm=np.isin(labels,keep)
    if not pm.any() or not (nd.binary_dilation(pm,structure=CROSS)&sea&~pm).any() or not (nd.binary_dilation(grid==PORT,structure=CROSS)&(grid==LAND)).any(): grid[:]=old; return False
    grid[pm]=PIER; return True

def validate(grid,airports):
    errors=[]; allowed={LAND,SEA,RIVER,MOUNTAIN,PORT,AIRPORT,ROAD,PIER}
    if grid.shape!=(N,N) or not set(np.unique(grid)).issubset(allowed): errors.append('크기 또는 타일 코드 오류')
    for tile,other in ((PORT,PIER),(PIER,PORT)):
        lab,n=nd.label(grid==tile,structure=CROSS)
        for i in range(1,n+1):
            m=lab==i; adj=nd.binary_dilation(m,structure=CROSS)
            if not (adj & (grid==other)).any(): errors.append(f'{tile} 연결 오류')
            if tile==PIER and not (adj & (grid==SEA)).any(): errors.append('부두 해상 공간 없음')
            if tile==PORT:
                if not (adj & np.isin(grid,[LAND,ROAD])).any(): errors.append('항구 육지 연결 오류')
                if not any(nd.binary_erosion(m,structure=np.ones(shape,bool)).any() for shape in [(2,4),(4,2)]): errors.append('터미널 부지 부족')
    a=grid==AIRPORT; count=0
    for x,y,w,h in airports:
        if (w,h) not in [(40,16),(16,40)] or not a[y:y+h,x:x+w].all(): errors.append('공항 규격 오류')
        count+=w*h
    if a.sum()!=count: errors.append('공항 타일 수 오류')
    return errors

def build(frame,elements,elev):
    grid=np.zeros((N,N),np.uint8); notes=[]; objects=[]
    for e in elements:
        try:
            g=geom(e,frame)
            if g is not None and not g.is_empty and g.intersects(box(-30,-30,N+30,N+30)): objects.append((e.get('tags',{}),g,e.get('id')))
        except Exception as ex: notes.append(f'원자료 객체 {e.get("id")} 처리 실패: {ex}')
    grid[coast_mask([g for t,g,_ in objects if t.get('natural')=='coastline'],notes)]=SEA
    for t,g,_ in objects:
        if t.get('natural')=='water' or t.get('waterway')=='riverbank' or t.get('water')=='sea':
            p=asarea(g)
            if p.geom_type in ('Polygon','MultiPolygon'):
                w=mask(p); grid[w & (grid!=SEA)]=SEA if t.get('water')=='sea' else RIVER
        elif t.get('waterway') in ('river','stream','canal'):
            w=mask(g.buffer(line_width(t,frame.m)/2,cap_style=1)); grid[w & (grid!=SEA)]=RIVER
    # Adjustable mountain proxy: both relative relief and slope; preserve actual lowland.
    if elev is None: notes.append('고도 자료 없음: 산 검증 불가')
    else:
        low=nd.minimum_filter(elev,size=61); relief=elev-low; gy,gx=np.gradient(elev,frame.m); slope=np.hypot(gx,gy)
        mountain=((relief>100)&(slope>.12))|((relief>180)&(elev>150)); mountain=nd.binary_opening(nd.binary_closing(mountain,iterations=2),iterations=1)
        grid[mountain & (grid==LAND)]=MOUNTAIN
    original_terrain=grid.copy()
    runways=[g for t,g,_ in objects if t.get('aeroway')=='runway' and g.geom_type=='LineString']; airports=[]
    facilities=[g for t,g,_ in objects if t.get('aeroway')=='aerodrome']
    # Each mapped aerodrome, or unmatched mapped runway cluster. No invented airport.
    anchors=[]
    for g in sorted(facilities,key=lambda g:g.area,reverse=True):
        g=asarea(g); c=g.centroid
        if any(c.distance(p)<25 for p,_ in anchors): continue
        nearby=[r for r in runways if g.distance(r)<15]
        anchors.append((c,max(nearby,key=lambda r:r.length) if nearby else None))
    for r in sorted(runways,key=lambda r:r.length,reverse=True):
        c=r.centroid
        if not any(c.distance(p)<35 for p,_ in anchors): anchors.append((c,r))
    for c,r in anchors:
        if not frame.domain.covers(c): continue
        horizontal=True
        if r is not None:
            a,b=r.coords[0],r.coords[-1]; horizontal=abs(b[0]-a[0])>=abs(b[1]-a[1])
        else: notes.append('활주로 방향 자료 없음: 공항 가로 방향 임시 적용')
        a=airport(grid,(c.x,c.y),horizontal)
        if a: airports.append(a)
        else: notes.append(f'공항 배치 실패: {c.x:.0f},{c.y:.0f}')
    candidates=[]
    port_anchors=[asarea(g).centroid for t,g,_ in objects if t.get('harbour') not in (None,'no') or t.get('landuse')=='port' or t.get('industrial')=='port' or t.get('amenity')=='ferry_terminal']
    for t,g,_ in objects:
        pier=t.get('man_made') in ('pier','quay'); other=t.get('harbour') not in (None,'no') or t.get('landuse')=='port' or t.get('industrial')=='port' or t.get('amenity')=='ferry_terminal'
        if pier or other:
            c=asarea(g).centroid
            if pier and not other and not any(c.distance(a)<30 for a in port_anchors): continue
            if frame.domain.covers(c): candidates.append((not pier,c,g if pier else None))
    ports=[]; unresolved=[]
    for _,c,g in sorted(candidates,key=lambda a:a[0]):
        if any(c.distance(p)<12 for p in ports): continue
        if port(grid,(c.x,c.y),g,notes): ports.append(c)
        else: unresolved.append(c)
    for c in unresolved:
        if not any(c.distance(p)<25 for p in ports): notes.append(f'항구/부두 배치 실패: {c.x:.0f},{c.y:.0f}')
    widths={'motorway':4,'trunk':3,'primary':3,'secondary':2,'tertiary':2,'residential':1,'unclassified':1}
    for t,g,_ in objects:
        if t.get('highway') not in widths or t.get('tunnel') not in (None,'no') or t.get('bridge') not in (None,'no'): continue
        # Omit short residential fragments; do not paint over water, mountain, or facilities.
        if t['highway'] in ('residential','unclassified') and g.length<6: continue
        w=mask(g.buffer(widths[t['highway']]/2,cap_style=1)); grid[w & (grid==LAND)]=ROAD
    errors=validate(grid,airports)
    # Compare against terrain derived directly from the source before gameplay simplification.
    # A pier is the only permitted change to sea; rivers and mountains must remain intact.
    invalid_water=((original_terrain==SEA)&~np.isin(grid,[SEA,PIER]))|((original_terrain==RIVER)&(grid!=RIVER))
    if invalid_water.any(): errors.append('원자료 수역과 불일치')
    if ((original_terrain==MOUNTAIN)&(grid!=MOUNTAIN)).any(): errors.append('원자료 산지와 불일치')
    land=grid==LAND; labels,n=nd.label(land,structure=CROSS); sizes=np.bincount(labels.ravel()); largest=int(sizes[1:].max()) if n else 0
    if land.sum()<N*N*.12 or largest<2500: notes.append('건축용 연결 평지 부족 가능성: 맵 범위/산 기준 검토')
    notes.append('실제 도로의 교량/터널 및 산·물 위 도로는 현 단계에서 생략; 도로망이 끊길 수 있음')
    return grid,airports,len(ports),notes,errors,original_terrain

def save(out,city,grid,report):
    out.mkdir(parents=True,exist_ok=True); name=str(city['code']); colors=json.dumps(COLORS); data=','.join(map(str,grid.ravel())); safe=json.dumps(city['name'],ensure_ascii=False).replace('<','\\u003c')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>500×500 city map</title><style>body{{margin:0;background:#111;color:#ddd;text-align:center;font:14px sans-serif}}canvas{{width:min(94vw,85vh);height:auto;image-rendering:pixelated}}</style><h2 id="name"></h2><canvas id="mapCanvas" width="500" height="500"></canvas><p>© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · <a href="https://registry.opendata.aws/terrain-tiles/">AWS Terrain Tiles / elevation sources</a></p><script>const gridData = new Uint8Array([{data}]);const colors={colors};const tileCodes={{51:'5-1',8:'8'}};document.getElementById('name').textContent={safe};const ctx=document.getElementById('mapCanvas').getContext('2d');for(let i=0;i<gridData.length;i++){{ctx.fillStyle=colors[gridData[i]];ctx.fillRect(i%500,Math.floor(i/500),1,1);}}</script></html>'''
    (out/(name+'.html')).write_text(html,encoding='utf-8'); (out/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    palette=np.zeros((256,3),np.uint8)
    for k,v in COLORS.items(): palette[k]=[int(v[i:i+2],16) for i in (1,3,5)]
    Image.fromarray(palette[grid]).save(out/(name+'.png'))

def load_cities(path):
    s=path.read_text(encoding='utf-8-sig')
    if path.suffix=='.json': return json5.loads(s)
    m=re.search(r'\b(?:const|let|var)\s+citiesData\s*=\s*(\[.*?\])\s*;',s,re.S)
    if not m: raise RuntimeError('ctlist.html의 citiesData 배열을 읽을 수 없음')
    return json5.loads(m.group(1))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cities',type=Path,default=ROOT/'ctlist.html'); p.add_argument('--codes',nargs='*'); p.add_argument('--limit',type=int,default=3); p.add_argument('--km',type=float,default=25); p.add_argument('--out',type=Path,default=ROOT/'generated_maps'); p.add_argument('--retry',action='store_true'); p.add_argument('--allow-no-dem',action='store_true'); a=p.parse_args()
    cities=load_cities(a.cities); failures=0; count=0
    for city in cities:
        code=str(city['code']);
        if not re.fullmatch(r'[\w-]+',code): print('unsafe code skipped:',code); continue
        if a.codes and code not in a.codes: continue
        if (a.out/(code+'.html')).exists() and not a.retry: continue
        if count>=a.limit: break
        count+=1; print('BUILD',code,city['name'],flush=True)
        try:
            lat,lon=geocode(city); frame=Frame(lat,lon,float(city.get('km',a.km))); key=f'{code}-{lat:.5f}-{lon:.5f}-{frame.km}'; elements=osm(frame,key)
            if not elements: raise RuntimeError('OSM 데이터가 비어 있음')
            try: elev=dem(frame)
            except Exception:
                if not a.allow_no_dem: raise
                elev=None
            grid,airports,ports,notes,errors,reference=build(frame,elements,elev)
            report={'city':city,'center':[lat,lon],'km':frame.km,'tile_m':frame.m,'tile_px':32,'orientation':'north-up','counts':{str(k):int(v) for k,v in Counter(grid.ravel()).items()},'airports':airports,'ports':ports,'status':'REVIEW_REQUIRED' if notes or errors else 'STRUCTURE_OK','structural_errors':errors,'notes':notes,'sources':['OpenStreetMap contributors (ODbL)','AWS Terrain Tiles (see source attribution)'],'geographic_validation':'Source terrain checked for sea/river/mountain changes; independent map or human review still needed','comparison_image':'left: source-derived terrain; right: game map'}
            if errors: raise RuntimeError('; '.join(errors))
            save(a.out,city,grid,report)
            palette=np.zeros((256,3),np.uint8)
            for k,v in COLORS.items(): palette[k]=[int(v[i:i+2],16) for i in (1,3,5)]
            Image.fromarray(np.hstack([palette[reference],palette[grid]])).save(a.out/(code+'-compare.png'))
            errfile=a.out/(code+'.error.json')
            if errfile.exists(): errfile.unlink()
            print('SAVED (review required)',code,flush=True)
        except Exception as ex:
            failures+=1; a.out.mkdir(parents=True,exist_ok=True); (a.out/(code+'.error.json')).write_text(json.dumps({'city':city,'error':str(ex)},ensure_ascii=False,indent=2)); print('FAILED',code,ex,flush=True)
        time.sleep(3)
    print('attempted',count,'failed',failures)
    return 1 if failures else 0

if __name__=='__main__': sys.exit(main())
