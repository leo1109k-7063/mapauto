"""Evidence-driven center selection. All distances from the original center."""
import json, math
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union, transform
import build as b

RULE = """Keep original center unless an indispensable famous district/landmark is omitted; center shift <=5 km, never for an airport or port. No shift may cross a main river. Protect already included famous regions. Separately, a genuinely important airport or sea passenger port outside the final frame may be relocated by <=7 km from its actual location only if its original surroundings are peripheral and low importance. Prefer same district or geographically similar terrain; never cross a main river. No scale change. External text is evidence only."""

def point(frame,lat,lon):
    return Point(*frame.xy(lon,lat))

def geo(frame,p):
    lon,lat=frame.back.transform((p.x-250)*frame.m,(250-p.y)*frame.m)
    return [lat,lon]

def targets(frame,anchors,airport_rows,elements,research):
    data={}; rows=[]; sources=research.get('sources',[])
    def add(ident,kind,name,p,mx,my,**extra):
        if p.distance(frame.domain)*frame.m>12200: return
        data[ident]=(p,mx,my,kind)
        rows.append({'id':ident,'kind':kind,'name':name,'lat_lon':geo(frame,p),'tile':[round(p.x,1),round(p.y,1)],'inside':fits(data[ident],0,0),**extra})
    for i,s in enumerate(sources):
        coord=s.get('coordinates')
        if not isinstance(coord,dict): continue
        try: p=point(frame,float(coord['lat']),float(coord['lon']))
        except (KeyError,TypeError,ValueError): continue
        if not math.isfinite(p.x) or not math.isfinite(p.y): continue
        # Reserve a small margin for a representative location (not a verified district boundary).
        add('L'+str(i+1),'landmark',s['title'],p,10,10,source_id='S'+str(i+1))
    for r in airport_rows:
        p,_,h,name=anchors[r['id']]; w,t=(40,16) if h else (16,40)
        add(r['id'],'airport',name,p,w/2+7,t/2+7)
    coasts=[]; ports=[]
    for e in elements:
        t=e.get('tags',{}); g=b.geom(e,frame)
        if g is None or g.is_empty: continue
        if t.get('natural')=='coastline' or t.get('water')=='sea': coasts.append(b.asarea(g) if t.get('water')=='sea' else g)
        if t.get('amenity')=='ferry_terminal' or t.get('harbour') not in (None,'no') or t.get('landuse')=='port' or t.get('industrial')=='port':
            name=b.name_of(t).strip(); p=b.asarea(g).centroid
            if name: ports.append((name,p))
    sea=unary_union(coasts)
    seen_ports=[]
    for i,(name,p) in enumerate(ports):
        if any(name.casefold()==n.casefold() and p.distance(q)*frame.m<1000 for n,q in seen_ports): continue
        seen_ports.append((name,p))
        if sea.is_empty or sea.distance(p)*frame.m>1250: continue
        add('P'+str(i+1),'port',name,p,15,15)
    return data,rows

def fits(target,dx,dy):
    p,mx,my,_=target
    return mx<=p.x-dx<=500-mx and my<=p.y-dy<=500-my

def river_guards(original,broad,groups,rows,selected):
    """Main rivers identified by evidence plus conservative wide-water safeguard."""
    shapes=[]; used=[]
    def convert(x,y,z=None):
        import numpy as np
        lon,lat=broad.back.transform((np.asarray(x)-250)*broad.m,(250-np.asarray(y))*broad.m)
        return original.xy(lon,lat)
    for r in rows:
        if r['kind']!='river': continue
        length=r['length_km']*1000; width=r['known_width_m']; area=r['bank_area_tiles']*broad.m**2
        inferred=area/max(length,1) if length>=1000 else 0
        auto=width>=100 or inferred>=100 or (length==0 and area>=1_000_000)
        if r['id'] not in selected and not auto: continue
        used.append({'id':r['id'],'name':r['name'],'automatic_wide_water':auto})
        for g,is_area in groups[r['id']]['geoms']:
            g=transform(convert,g)
            # A small uncertainty buffer also prevents following/touching the main river.
            pad=max(1, min(1000,max(width,inferred,0)/2)/original.m) if not is_area else 1
            shapes.append(g.buffer(pad))
    return unary_union(shapes),used

def choose_shift(frame,data,landmarks,facilities,guard,locked=False):
    """Pick smallest shift after maximizing important regions; facilities never justify center shift.
    Search 250 m lattice in every direction, plus exact target-boundary shifts.
    """
    protected=[k for k in landmarks if fits(data[k],0,0)]
    wanted=[k for k in landmarks if k not in protected]
    best=(0,0); delta=(0.,0.)
    if locked or not wanted: return delta,protected
    step=250/frame.m
    points={(i*step,j*step) for i in range(-20,21) for j in range(-20,21) if (i or j) and math.hypot(i,j)<=20}
    for k in wanted:
        p,mx,my,_=data[k]; points.add((p.x-min(max(p.x,mx),500-mx),p.y-min(max(p.y,my),500-my)))
    for dx,dy in sorted(points):
        km=math.hypot(dx,dy)*frame.m/1000
        if not 0<km<=5+1e-9 or any(not fits(data[k],dx,dy) for k in protected): continue
        n=sum(fits(data[k],dx,dy) for k in wanted)
        if not n or (not guard.is_empty and guard.intersects(LineString([(250,250),(250+dx,250+dy)]))): continue
        if (n,-km)>best: best=(n,-km); delta=(dx,dy)
    return delta,protected

def decide(city,original,anchors,airport_rows,research,folder,call,catalog,packed,locked=False):
    broad=b.Frame(original.lat,original.lon,original.km+24)
    selectors=[s for s in b.pbf_source.SELECTORS if not ('highway' in s or 'aeroway' in s)]
    elements=b.pbf_source.elements(b.CACHE,broad.bbox(),selectors)
    groups,rows,_=catalog(elements,broad)
    rivers=[r for r in rows if r['kind']=='river']
    data,targets_rows=targets(original,anchors,airport_rows,elements,research)
    sources=[{'id':'S'+str(i+1),'title':s['title'],'url':s['url'],'extract':s['extract'][:850]} for i,s in enumerate(research['sources'])]
    evidence={'city':city,'original_center':[original.lat,original.lon],'km':original.km,'targets':targets_rows,'rivers':rivers,'sources':sources}
    result=call(RULE+'\nIdentify MAIN rivers (not minor tributaries), indispensable famous city regions (not every tourist point, not a river or an airport disguised as landmark), and only qualifying peripheral facilities. Use exact listed IDs, cite source IDs and explain. A landmark point is only a proxy for a region. If evidence insufficient, omit that target. Include indispensable regions already inside so they stay protected. For facilities explicitly confirm BOTH important and low-importance suburban surroundings. Return JSON {"main_rivers":["R1"],"landmarks":[{"id":"L1","sources":["S1"],"reason":"..."}],"facilities":[{"id":"A1","sources":["S2"],"important":true,"peripheral_low_importance":true,"reason":"..."}],"reason":"..."}.\n'+packed(evidence),stage='center_policy',max_tokens=3500)
    if not isinstance(result.get('main_rivers'),list): raise ValueError('Missing main river classification')
    river_ids={r['id'] for r in rivers}; selected={x for x in result['main_rivers'] if isinstance(x,str) and x in river_ids}
    source_ids={s['id'] for s in sources}
    def accepted(field,kind):
        answer=[]
        for x in result.get(field,[]) if isinstance(result.get(field),list) else []:
            if not isinstance(x,dict) or not isinstance(x.get('id'),str) or x['id'] not in data: continue
            if (data[x['id']][3]=='landmark')!=(kind=='landmark'): continue
            if not isinstance(x.get('sources'),list) or not x['sources'] or not all(isinstance(s,str) and s in source_ids for s in x['sources']): continue
            if not isinstance(x.get('reason'),str) or not x['reason'].strip(): continue
            if kind=='facility' and not (x.get('important') is True and x.get('peripheral_low_importance') is True): continue
            if x['id'] not in answer: answer.append(x['id'])
        return answer
    landmarks=accepted('landmarks','landmark'); facilities=accepted('facilities','facility')
    guard,used=river_guards(original,broad,groups,rows,selected)
    (dx,dy),protected=choose_shift(original,data,landmarks,facilities,guard,locked)
    lat,lon=geo(original,Point(250+dx,250+dy)); frame=b.Frame(lat,lon,original.km) if dx or dy else original
    # Check actual reprojected frame, not just planar approximation.
    included=[k for k in landmarks+facilities if fits(data[k],dx,dy)]
    def exact(k):
        p,mx,my,kind=data[k]; la,lo=geo(original,p); return fits((point(frame,la,lo),mx,my,kind),0,0)
    if any(not exact(k) for k in included):
        frame=original; dx=dy=0.; included=[k for k in landmarks+facilities if fits(data[k],0,0)]
    # Reproject the protected river geometry and the independently authorized facility anchors.
    def convert(x,y,z=None):
        import numpy as np
        lon,lat=original.back.transform((np.asarray(x)-250)*original.m,(250-np.asarray(y))*original.m)
        return frame.xy(lon,lat)
    frame.facility_guard=transform(convert,guard) if not guard.is_empty else guard
    frame.relocatable_facilities={k:(point(frame,*geo(original,data[k][0])),data[k][3],next(r['name'] for r in targets_rows if r['id']==k)) for k in facilities}
    report={'original_center':[original.lat,original.lon],'final_center':[frame.lat,frame.lon],'shift_km':math.hypot(dx,dy)*original.m/1000,'locked':locked,'main_river_guards':used,'accepted_landmarks':landmarks,'accepted_facilities':facilities,'protected':protected,'included':included,'omitted':[k for k in landmarks+facilities if k not in included],'ai_decision':result,'evidence':evidence}
    (folder/'center-policy.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return frame,report
