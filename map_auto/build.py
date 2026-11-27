#!/usr/bin/env python3
"""OSM vector + Terrarium DEM -> 500x500 game map. No API key needed."""
import argparse, json, math, re, time, io, sys, os, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage as nd
from PIL import Image, ImageDraw
import requests, json5
import pbf_source
from connectivity import connected_network, components
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union, polygonize
from shapely import make_valid

N=500; LAND,SEA,RIVER,MOUNTAIN,PORT,AIRPORT,ROAD,PIER,ROAD_RIVER=0,1,2,3,5,6,8,51,81
COLORS={0:'#4CAF50',1:'#1A237E',2:'#00BCD4',3:'#1B5E20',4:'#FFFFFF',5:'#BDBDBD',6:'#424242',8:'#FFFF00',51:'#795548',81:'#B8860B'}
ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'.map-cache'; CACHE.mkdir(exist_ok=True)
SESSION=requests.Session(); SESSION.headers['User-Agent']='mapaditer-game-map/1.0 (https://github.com/leo1109k-7063/mapaditer)'
CROSS=nd.generate_binary_structure(2,1)

def _valid_json(data, overpass=False):
    if isinstance(data,dict) and data.get('remark'):
        raise ValueError(str(data['remark']))
    if overpass and (not isinstance(data,dict) or not isinstance(data.get('elements'),list)):
        raise ValueError('Invalid Overpass response: elements missing')
    return data

def _read_cache(p,overpass=False):
    if p.exists():
        try: return _valid_json(json.loads(p.read_text()),overpass)
        except (ValueError,OSError): p.unlink(missing_ok=True)
    return None

def _write_cache(p,data):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False)); tmp.replace(p)

def cached_json(key,url,params=None,post=None):
    overpass=bool(post and 'data' in post); p=CACHE/(('pbf-query-v1-' if overpass else '')+key+'.json')
    data=_read_cache(p,overpass)
    if data is not None: return data
    if overpass:
        # Legacy caller (airport search) uses Overpass selector syntax only; read local PBF instead.
        query=post['data']; matches=list(re.finditer(r'((?:nwr|way|node|rel)(?:\[[^\]]+\])+?)\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\);',query))
        if not matches or len({m.groups()[1:] for m in matches})!=1:
            raise RuntimeError('Unsupported local PBF query')
        bbox=tuple(map(float,matches[0].groups()[1:])); selectors=[m.group(1) for m in matches]
        data={'elements':pbf_source.elements(CACHE,bbox,selectors)}
    else:
        r=SESSION.post(url,data=post,timeout=(15,90)) if post else SESSION.get(url,params=params,timeout=(15,90))
        r.raise_for_status(); data=_valid_json(r.json())
    _write_cache(p,data); return data

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
    return pbf_source.elements(CACHE,frame.bbox(),pbf_source.SELECTORS)

def geom(e,f):
    def line(g): return LineString([f.xy(p['lon'],p['lat']) for p in g]) if len(g)>=2 else None
    if e['type']=='node': return Point(f.xy(e['lon'],e['lat']))
    if e['type']=='way':
        g=line(e.get('geometry',[])); return g
    # Only actual polygon boundaries, never route/member centre lines.
    if e.get('tags',{}).get('type') not in ('multipolygon','boundary'): return None
    outer=[]; inner=[]
    for m in e.get('members',[]):
        if m.get('role','') not in ('outer','inner',''): continue
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

# Seoul's actual civil airport: official Korean AIP ARP, not an arbitrary aerodrome.
GIMPO=(37+33/60+25/3600,126+47/60+51/3600)
SEOUL_CENTER=(37.5665,126.92)  # 25 km: retain scale, include Gimpo; east edge shifts west.
SEOUL_RIVERS=('한강','중랑천','탄천','안양천','홍제천','불광천','청계천','양재천','도림천','한강','Han River','Hangang','Jungnang','Tancheon','Anyangcheon','Hongje','Bulgwang','Cheonggye','Yangjae','Dorim')

def name_of(t):
    return ' '.join(str(t.get(k,'')) for k in ('name','name:ko','name:en','official_name','ref'))

def named_key(t):
    return str(t.get('name:ko') or t.get('name') or t.get('name:en') or t.get('ref') or '').strip()

def width_m(t):
    v=str(t.get('width','')).strip().lower()
    m=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(m|metres?|meters?|ft|feet)?',v)
    if not m: return None
    w=float(m.group(1))*(.3048 if m.group(2) in ('ft','feet') else 1)
    return w if w>0 else None

def line_width(t,m):
    # Unknown width is a narrow one-cell fallback, not 9 or 4 tiles.
    return max(1,min(35,(width_m(t) or m)/m))

def underground(t):
    return (t.get('tunnel') not in (None,'no','false','0') or t.get('location')=='underground' or t.get('covered')=='yes' or t.get('intermittent')=='yes' or t.get('seasonal')=='yes')

def line_mask(g,width=1):
    im=Image.new('1',(N,N)); d=ImageDraw.Draw(im)
    def draw(a):
        if a.is_empty: return
        if a.geom_type in ('LineString','LinearRing'):
            d.line([(math.floor(x),math.floor(y)) for x,y in a.coords],fill=1,width=int(width))
        elif hasattr(a,'geoms'):
            for b in a.geoms: draw(b)
    draw(g.intersection(box(-3,-3,N+3,N+3)))
    return np.asarray(im,dtype=bool)

def major_seoul(t):
    n=name_of(t).lower(); return any(k.lower() in n for k in SEOUL_RIVERS)

def water_masks(objects,frame,sea):
    # Use measured surface polygons first; centre lines fill only missing river geometry.
    lengths=Counter(); lines=[]; areas=[]
    for t,g,_ in objects:
        if underground(t): continue
        k=t.get('waterway'); typ=t.get('water'); n=named_key(t)
        if k in ('river','stream','canal') and g.geom_type in ('LineString','MultiLineString'):
            lines.append((t,g));
            if n: lengths[n]+=g.intersection(frame.domain).length*frame.m
        if typ in ('river','canal') or k=='riverbank':
            a=asarea(g)
            if a.geom_type in ('Polygon','MultiPolygon'): areas.append((t,a))
    selected=[]
    for t,g in lines:
        n=named_key(t); length=lengths[n] if n else g.intersection(frame.domain).length*frame.m
        w=width_m(t) or 0; k=t.get('waterway'); known=getattr(frame,'city_code','')=='SEO' and major_seoul(t)
        if k=='stream': keep=known or (bool(n) and length>=3000 and w>=20)
        elif k=='river': keep=known or (bool(n) and length>=800) or length>=2000 or w>=20
        else: keep=known or (bool(n) and length>=2000 and w>=15) or w>=40
        if t.get('_ai_width') is not None: keep=True
        if keep: selected.append((t,g))
    center=np.zeros((N,N),bool)
    for t,g in selected: center|=line_mask(g)
    surface=np.zeros_like(center); water=np.zeros_like(center); center_near=nd.binary_dilation(center,iterations=2)
    for t,a in areas:
        am=mask(a); cells=int(am.sum()); known=getattr(frame,'city_code','')=='SEO' and major_seoul(t)
        # Ignore tiny detached water patches; explicit large river surfaces are authoritative.
        if t.get('_ai_surface') or cells>=100 or known or (cells>=4 and (am & center_near).any()): surface|=am
    water|=surface
    near=nd.binary_dilation(surface,iterations=1)
    for t,g in selected:
        w=t.get('_ai_width',line_width(t,frame.m))
        if w<=1: lm=line_mask(g)
        else: lm=mask(g.buffer(w/2,cap_style=2))
        # Do not inflate a river where actual bank geometry is available.
        water|=(lm & ~near) | line_mask(g)
    water&=~sea
    labels,n=nd.label(water,structure=np.ones((3,3),bool)); sizes=np.bincount(labels.ravel()); keep=sizes>=8; keep[0]=False
    water=keep[labels]
    return water

def airport(grid,center,horizontal,max_shift=4):
    w,h=(40,16) if horizontal else (16,40); x,y=center; best=None
    for dy in range(-max_shift,max_shift+1):
        for dx in range(-max_shift,max_shift+1):
            if dx*dx+dy*dy>max_shift*max_shift: continue
            l=int(round(x-w/2))+dx; t=int(round(y-h/2))+dy
            if l<0 or t<0 or l+w>N or t+h>N: continue
            v=grid[t:t+h,l:l+w]
            if np.any(v!=LAND): continue
            score=dx*dx+dy*dy
            if best is None or score<best[0]: best=(score,l,t)
    if best is None: return None
    _,l,t=best; grid[t:t+h,l:l+w]=AIRPORT; return [l,t,w,h]

def allowed_facility_move(frame,source,target,limit=7000):
    from pyproj import Geod
    a=frame.back.transform((source.x-250)*frame.m,(250-source.y)*frame.m)
    z=frame.back.transform((target.x-250)*frame.m,(250-target.y)*frame.m)
    distance=abs(Geod(ellps='WGS84').inv(*a,*z)[2])
    guard=getattr(frame,'facility_guard',None)
    return distance<=limit and (guard is None or guard.is_empty or not guard.intersects(LineString([source,target])))

def relocate_airport(grid,frame,source,horizontal):
    w,h=(40,16) if horizontal else (16,40)
    # Integral image: exact rectangle land coverage for every possible anchor.
    integral=np.pad((grid!=LAND).astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
    sums=integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
    y,x=np.where(sums==0); d=(x+w/2-source.x)**2+(y+h/2-source.y)**2
    for i in np.argsort(d):
        if d[i]*frame.m**2>7000**2: break
        l,t=int(x[i]),int(y[i]); q=Point(l+w/2,t+h/2)
        if allowed_facility_move(frame,source,q):
            grid[t:t+h,l:l+w]=AIRPORT; return [l,t,w,h]
    return None

def relocation_record(frame,name,kind,source,target):
    from pyproj import Geod
    def coords(p):
        lon,lat=frame.back.transform((p.x-250)*frame.m,(250-p.y)*frame.m); return [lat,lon]
    a,z=coords(source),coords(target)
    return {'name':name,'kind':kind,'actual_lat_lon':a,'placed_lat_lon':z,'actual_tile':[source.x,source.y],'placed_tile':[target.x,target.y],'distance_m':abs(Geod(ellps='WGS84').inv(a[1],a[0],z[1],z[0])[2])}

def relocate_port(grid,frame,source,notes):
    sea=grid==SEA; coast=(grid==LAND)&nd.binary_dilation(sea,structure=CROSS)
    y,x=np.where(coast); d=(x-source.x)**2+(y-source.y)**2
    for i in np.argsort(d):
        if d[i]*frame.m**2>7000**2: break
        q=Point(int(x[i]),int(y[i]))
        if not allowed_facility_move(frame,source,q): continue
        trial=grid.copy(); trial_notes=[]
        if not port(trial,(q.x,q.y),None,trial_notes): continue
        yy,xx=np.where((trial==PORT)&(grid!=PORT))
        if not len(xx): continue
        actual=Point(float(xx.mean()+.5),float(yy.mean()+.5))
        if not allowed_facility_move(frame,source,actual): continue
        grid[:]=trial; notes.extend(trial_notes); return actual
    return None

def airport_anchors(frame,objects,notes):
    if hasattr(frame,'ai_airports'): return frame.ai_airports
    runways=[g for t,g,_ in objects if t.get('aeroway')=='runway' and g.geom_type=='LineString' and g.length*frame.m>=1000]
    if getattr(frame,'city_code','')=='SEO':
        c=Point(frame.xy(GIMPO[1],GIMPO[0])); nearby=[r for r in runways if r.distance(c)*frame.m<=1800]
        # Official runway alignment NW-SE; if geometry is missing, keep the nearest cardinal (vertical).
        return [(c,max(nearby,key=lambda r:r.length) if nearby else None,False,'Gimpo (GMP/RKSS)')]
    anchors=[]
    for t,g,_ in sorted((o for o in objects if o[0].get('aeroway')=='aerodrome'),key=lambda item:asarea(item[1]).area,reverse=True):
        if t.get('aerodrome') in ('military','private','airstrip') or t.get('military') not in (None,'no') or t.get('access')=='private': continue
        a=asarea(g); c=a.centroid
        nearby=[r for r in runways if a.distance(r)*frame.m<=800]
        if not (t.get('iata') or t.get('aerodrome') in ('international','public','commercial') or (t.get('icao') and nearby)): continue
        if any(c.distance(p)*frame.m<2000 for p,_,_,_ in anchors): continue
        r=max(nearby,key=lambda r:r.length) if nearby else None
        # Runway midpoint is a trustworthy anchor when inside the mapped airport.
        if r is not None: c=r.interpolate(.5,normalized=True)
        anchors.append((c,r,True,name_of(t).strip() or t.get('iata') or t.get('icao') or 'airport'))
    return anchors

def road_mask(objects,frame):
    widths={'motorway':2,'trunk':2,'primary':1,'secondary':1}; totals=Counter(); roads=[]
    for t,g,_ in objects:
        k=t.get('highway')
        if k not in widths or g.geom_type not in ('LineString','MultiLineString') or t.get('tunnel') not in (None,'no') : continue
        name=named_key(t); roads.append((t,g,name));
        if name: totals[(k,name)]+=g.intersection(frame.domain).length*frame.m
    result=np.zeros((N,N),bool)
    for t,g,name in roads:
        k=t['highway']; length=totals[(k,name)] if name else g.intersection(frame.domain).length*frame.m
        if not t.get('_ai_width') and k=='secondary' and (not name or length<4000): continue
        result|=line_mask(g.simplify(.65,preserve_topology=True),t.get('_ai_width',widths[k]))
    return result

ROAD_CONNECT_CLASSES={'motorway','trunk','primary','secondary','motorway_link','trunk_link','primary_link','secondary_link'}
def source_corridors(objects):
    road=np.zeros((N,N),bool); water=np.zeros_like(road)
    for t,g,_ in objects:
        if underground(t): continue
        if g.geom_type in ('LineString','MultiLineString'):
            if t.get('highway') in ROAD_CONNECT_CLASSES: road|=line_mask(g)
            if t.get('waterway') in ('river','stream','canal'): water|=line_mask(g)
        # A real bank polygon is not itself a new visible river. It is a valid connection corridor.
        if t.get('water') in ('river','canal') or t.get('waterway')=='riverbank':
            a=asarea(g)
            if a.geom_type in ('Polygon','MultiPolygon'): water|=mask(a)
    return road,water

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
    local_port=np.zeros_like(sea); local_port[t:t+h,l:l+w]=True
    adjacent=sea & nd.binary_dilation(local_port,structure=CROSS)
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
    if not pm.any() or not (nd.binary_dilation(pm,structure=CROSS)&sea&~pm).any() or not (nd.binary_dilation(local_port,structure=CROSS)&(grid==LAND)).any(): grid[:]=old; return False
    grid[pm]=PIER; return True

def validate(grid,airports):
    errors=[]; allowed={LAND,SEA,RIVER,MOUNTAIN,PORT,AIRPORT,ROAD,PIER,ROAD_RIVER}
    if grid.shape!=(N,N) or not set(np.unique(grid)).issubset(allowed): errors.append('크기 또는 타일 코드 오류')
    for tile,other in ((PORT,PIER),(PIER,PORT)):
        lab,n=nd.label(grid==tile,structure=CROSS)
        for i in range(1,n+1):
            m=lab==i; adj=nd.binary_dilation(m,structure=CROSS)
            if not (adj & (grid==other)).any(): errors.append(f'{tile} 연결 오류')
            if tile==PIER and not (adj & (grid==SEA)).any(): errors.append('부두 해상 공간 없음')
            if tile==PORT:
                if not (adj & np.isin(grid,[LAND,ROAD,ROAD_RIVER])).any(): errors.append('항구 육지 연결 오류')
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
        if t.get('water')=='sea':
            a=asarea(g)
            if a.geom_type in ('Polygon','MultiPolygon'): grid[mask(a)]=SEA
    source_objects=objects
    if hasattr(frame,'network_source_elements'):
        source_objects=[]
        for e in frame.network_source_elements:
            try:
                g=geom(e,frame)
                if g is not None and not g.is_empty and g.intersects(frame.domain): source_objects.append((e.get('tags',{}),g,e.get('id')))
            except Exception as ex: notes.append(f'연결 원자료 객체 {e.get("id")} 처리 실패: {ex}')
    road_corridor,river_corridor=source_corridors(source_objects)
    water=water_masks(objects,frame,grid==SEA)
    selected_water=water.copy()
    water,river_stats=connected_network(water,river_corridor,grid!=SEA)
    frame.network_removed={'river':selected_water & ~water}
    grid[water]=RIVER
    # Adjustable mountain proxy: both relative relief and slope; preserve actual lowland.
    if elev is None: notes.append('고도 자료 없음: 산 검증 불가')
    else:
        low=nd.minimum_filter(elev,size=61); relief=elev-low; gy,gx=np.gradient(elev,frame.m); slope=np.hypot(gx,gy)
        mountain=((relief>100)&(slope>.12))|((relief>180)&(elev>150)); mountain=nd.binary_opening(nd.binary_closing(mountain,iterations=2),iterations=1)
        grid[mountain & (grid==LAND)]=MOUNTAIN
    original_terrain=grid.copy()
    airports=[]; frame.airport_details=[]; frame.facility_relocations=[]; frame.facility_unresolved=[]
    for c,r,horizontal,name in airport_anchors(frame,objects,notes):
        if r is not None:
            aa,zz=r.coords[0],r.coords[-1]; horizontal=abs(zz[0]-aa[0])>=abs(zz[1]-aa[1])
        else: notes.append(f'{name}: 활주로 방향 원자료 없음, 기본 방향 적용')
        w,h=(40,16) if horizontal else (16,40)
        fits=box(w/2,h/2,N-w/2,N-h/2).covers(c)
        a=airport(grid,(c.x,c.y),horizontal,max_shift=max(1,min(4,int(200/frame.m)))) if fits else None
        relocated=False
        if not a and not fits and name in getattr(frame,'airport_relocation_names',set()):
            a=relocate_airport(grid,frame,c,horizontal); relocated=bool(a)
        if a:
            airports.append(a); q=Point(a[0]+a[2]/2,a[1]+a[3]/2)
            detail=relocation_record(frame,name,'airport',c,q); detail['rectangle']=a; detail['relocated']=relocated
            frame.airport_details.append(detail)
            if relocated: frame.facility_relocations.append(detail)
        else:
            notes.append(f'공항 {name} 허용 위치 배치 실패'); frame.facility_unresolved.append(name)
    candidates=[]
    port_anchors=[asarea(g).centroid for t,g,_ in objects if t.get('harbour') not in (None,'no') or t.get('landuse')=='port' or t.get('industrial')=='port' or t.get('amenity')=='ferry_terminal']
    for t,g,_ in objects:
        pier=t.get('man_made') in ('pier','quay'); other=t.get('harbour') not in (None,'no') or t.get('landuse')=='port' or t.get('industrial')=='port' or t.get('amenity')=='ferry_terminal'
        if pier or other:
            c=asarea(g).centroid
            if pier and not other and not any(c.distance(a)<30 for a in port_anchors): continue
            if frame.domain.covers(c): candidates.append((not pier,c,g if pier else None))
    ports=[]; unresolved=[]
    # Inland ferry stops and quays are not ocean ports under current game rules.
    sea_dist=nd.distance_transform_edt(grid!=SEA) if np.any(grid==SEA) else None
    for _,c,g in sorted(candidates,key=lambda a:a[0]):
        x,y=min(N-1,max(0,int(c.x))),min(N-1,max(0,int(c.y)))
        if sea_dist is None or sea_dist[y,x]>25: continue
        if any(c.distance(p)<12 for p in ports): continue
        if port(grid,(c.x,c.y),g,notes): ports.append(c)
        else: unresolved.append(c)
    for c in unresolved:
        if not any(c.distance(p)<25 for p in ports): notes.append(f'항구/부두 배치 실패: {c.x:.0f},{c.y:.0f}')
    # Only externally verified, evidence-approved off-map ports may be abstracted onto nearby coast.
    for ident,(source,kind,name) in getattr(frame,'relocatable_facilities',{}).items():
        if kind!='port' or frame.domain.covers(source): continue
        if not np.any(grid==SEA): frame.facility_unresolved.append(name); continue
        if any(source.distance(p)*frame.m<500 for p in ports): continue
        q=relocate_port(grid,frame,source,notes)
        if q is None: frame.facility_unresolved.append(name); notes.append('항구/부두 배치 실패: '+name)
        else:
            ports.append(q); frame.facility_relocations.append(relocation_record(frame,name,'port',source,q))
    # Road wins on a real river crossing, but not on sea, mountains or facility tiles.
    road_allowed=np.isin(grid,[LAND,RIVER])
    roads=road_mask(objects,frame)&road_allowed
    selected_roads=roads.copy()
    roads,road_stats=connected_network(roads,road_corridor,road_allowed)
    frame.network_removed['road']=selected_roads & ~roads
    crossings=roads & water
    grid[roads]=ROAD
    grid[crossings]=ROAD_RIVER
    frame.network_stats={'roads':road_stats,'rivers':river_stats,'road_over_river_tiles':int(crossings.sum()),'logical_river_components':int(components(np.isin(grid,[RIVER,ROAD_RIVER]))[1]),'visible_road_components':int(components(np.isin(grid,[ROAD,ROAD_RIVER]))[1])}
    frame.river_crossings=crossings
    errors=validate(grid,airports)
    if frame.network_stats['visible_road_components']>1: errors.append('도로망 연결 오류')
    if frame.network_stats['logical_river_components']>1: errors.append('강 연결 오류')
    for name,stats in [('도로',road_stats),('강',river_stats)]:
        if stats['removed_tiles']: notes.append(f'{name}: 실제 연결 경로가 없는 {stats["removed_tiles"]}칸 제외; AI 재검토 대상')
        if stats['added_tiles']: notes.append(f'{name}: 실제 지도 경로 및 대각선 보정으로 {stats["added_tiles"]}칸 연결')
    # Compare against terrain derived directly from the source before gameplay simplification.
    # A pier is the only permitted change to sea; rivers and mountains must remain intact.
    invalid_water=((original_terrain==SEA)&~np.isin(grid,[SEA,PIER]))|((original_terrain==RIVER)&~np.isin(grid,[RIVER,ROAD_RIVER]))
    if invalid_water.any(): errors.append('원자료 수역과 불일치')
    if ((original_terrain==MOUNTAIN)&(grid!=MOUNTAIN)).any(): errors.append('원자료 산지와 불일치')
    land=grid==LAND; labels,n=nd.label(land,structure=CROSS); sizes=np.bincount(labels.ravel()); largest=int(sizes[1:].max()) if n else 0
    if land.sum()<N*N*.12 or largest<2500: notes.append('건축용 연결 평지 부족 가능성: 맵 범위/산 기준 검토')
    notes.append('강과 도로의 교차 지점은 어두운 노란색 도로 81로 표시하며 도로와 강 양쪽 연결로 인정; 바다·산·시설은 보존')
    return grid,airports,len(ports),notes,errors,original_terrain

def save(out,city,grid,report):
    report=dict(report); report['river_under_road']=np.flatnonzero(grid.ravel()==ROAD_RIVER).tolist()
    out.mkdir(parents=True,exist_ok=True); name=str(city['code']); colors=json.dumps(COLORS); data=','.join(map(str,grid.ravel())); safe=json.dumps(city['name'],ensure_ascii=False).replace('<','\\u003c')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>500×500 city map</title><style>body{{margin:0;background:#111;color:#ddd;text-align:center;font:14px sans-serif}}canvas{{width:min(94vw,85vh);height:auto;image-rendering:pixelated}}</style><h2 id="name"></h2><canvas id="mapCanvas" width="500" height="500"></canvas><p>© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · <a href="https://registry.opendata.aws/terrain-tiles/">AWS Terrain Tiles / elevation sources</a></p><script>const gridData = new Uint8Array([{data}]);const colors={colors};const tileCodes={{51:'5-1',8:'8',81:'81'}};const riverUnderRoad={json.dumps(report.get('river_under_road',[]),separators=(',',':'))};document.getElementById('name').textContent={safe};const ctx=document.getElementById('mapCanvas').getContext('2d');for(let i=0;i<gridData.length;i++){{ctx.fillStyle=colors[gridData[i]];ctx.fillRect(i%500,Math.floor(i/500),1,1);}}</script></html>'''
    (out/(name+'.html')).write_text(html,encoding='utf-8'); (out/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    palette=np.zeros((256,3),np.uint8)
    for k,v in COLORS.items(): palette[k]=[int(v[i:i+2],16) for i in (1,3,5)]
    Image.fromarray(palette[grid]).save(out/(name+'.png'))

CITY_LIST_URL = 'https://raw.githubusercontent.com/leo1109k-7063/mapaditer/main/ctlist.html'

def parse_cities(s, is_json=False):
    if is_json:
        cities=json5.loads(s)
    else:
        m=re.search(r'\b(?:const|let|var)\s+citiesData\s*=\s*(\[.*?\])\s*;',s,re.S)
        if not m: raise ValueError('citiesData 배열 없음 (Excel 프레임 HTML일 수 있음)')
        cities=json5.loads(m.group(1))
    if not isinstance(cities,list) or not cities or any(not isinstance(c,dict) or not c.get('code') or not c.get('name') for c in cities):
        raise ValueError('도시 목록 형식 오류 또는 빈 목록')
    return cities

def load_cities(path):
    path=Path(path)
    try:
        cities=parse_cities(path.read_text(encoding='utf-8-sig'),path.suffix.lower()=='.json')
        print(f'CITY LIST: {path} ({len(cities)} cities)',flush=True)
        return cities
    except (OSError,ValueError) as ex:
        if path.resolve()!=(ROOT/'ctlist.html').resolve(): raise
        print(f'LOCAL CITY LIST INVALID: {ex}; fetching original repository',flush=True)
    r=SESSION.get(CITY_LIST_URL,timeout=30); r.raise_for_status()
    cities=parse_cities(r.content.decode('utf-8-sig'))
    print(f'CITY LIST: {CITY_LIST_URL} ({len(cities)} cities)',flush=True)
    return cities

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cities',type=Path,default=ROOT/'ctlist.html'); p.add_argument('--codes',nargs='*'); p.add_argument('--limit',type=int,default=3); p.add_argument('--km',type=float,default=25); p.add_argument('--out',type=Path,default=ROOT/'generated_maps'); p.add_argument('--retry',action='store_true'); p.add_argument('--allow-no-dem',action='store_true'); p.add_argument('--list-cities',action='store_true'); a=p.parse_args()
    try:
        cities=load_cities(a.cities)
        if a.codes:
            unknown=set(a.codes)-{str(c['code']) for c in cities}
            if unknown: raise ValueError('도시 목록에 없는 코드: '+', '.join(sorted(unknown)))
        if a.list_cities:
            for c in cities: print(c['code'],c['name'],c.get('country',''))
            return 0
        if a.limit<1 or a.km<=0: raise ValueError('limit 및 km는 양수여야 함')
    except Exception as ex:
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'city-list.error.json').write_text(json.dumps({'stage':'city_list','error':str(ex),'source':str(a.cities),'fallback':CITY_LIST_URL},ensure_ascii=False,indent=2),encoding='utf-8')
        print('CITY LIST FAILED:',ex,flush=True)
        return 1
    failures=0; count=0
    for city in cities:
        code=str(city['code']);
        if not re.fullmatch(r'[\w-]+',code): print('unsafe code skipped:',code); continue
        if a.codes and code not in a.codes: continue
        if (a.out/(code+'.html')).exists() and not a.retry: continue
        if count>=a.limit: break
        count+=1; print('BUILD',code,city['name'],flush=True)
        try:
            km=float(city.get('km',a.km))
            if 'map_lat' in city and 'map_lon' in city: lat,lon=float(city['map_lat']),float(city['map_lon'])
            elif code=='SEO' and km<34: lat,lon=SEOUL_CENTER
            else: lat,lon=geocode(city)
            frame=Frame(lat,lon,km); frame.city_code=code
            key=f'{code}-{lat:.5f}-{lon:.5f}-{frame.km}'; elements=osm(frame,key)
            if not elements: raise RuntimeError('OSM 데이터가 비어 있음')
            try: elev=dem(frame)
            except Exception:
                if not a.allow_no_dem: raise
                elev=None
            grid,airports,ports,notes,errors,reference=build(frame,elements,elev)
            report={'city':city,'center':[lat,lon],'km':frame.km,'tile_m':frame.m,'tile_px':32,'orientation':'north-up','counts':{str(k):int(v) for k,v in Counter(grid.ravel()).items()},'airports':airports,'airport_details':frame.airport_details,'connectivity':frame.network_stats,'river_under_road':np.flatnonzero(frame.river_crossings.ravel()).tolist(),'generator_version':'2.0','ports':ports,'status':'REVIEW_REQUIRED' if notes or errors else 'STRUCTURE_OK','structural_errors':errors,'notes':notes,'sources':['OpenStreetMap contributors (ODbL)','AWS Terrain Tiles (see source attribution)'],'geographic_validation':'Source terrain checked for sea/river/mountain changes; independent map or human review still needed','comparison_image':'left: source-derived terrain; right: game map'}
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
