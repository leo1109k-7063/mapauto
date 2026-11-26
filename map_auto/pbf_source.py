"""Local OSM extracts via osmium-tool; no Overpass calls."""
import hashlib, json, os, re, shutil, subprocess, time, tempfile
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import requests
from shapely.geometry import shape, box, Point

INDEX='https://download.geofabrik.de/index-v1.json'
SELECTORS=[
 'way[natural=coastline]','nwr[water=sea]',
 'nwr[natural=water][water~"^(river|canal|sea)$"]','nwr[waterway=riverbank]',
 'way[waterway~"^(river|stream|canal)$"]',
 'way[highway~"^(motorway|trunk|primary)$"]','way[highway=secondary]',
 'nwr[aeroway=aerodrome]','way[aeroway=runway]',
 'nwr[harbour][harbour!=no]','nwr[landuse=port]','nwr[industrial=port]',
 'nwr[amenity=ferry_terminal]','nwr[man_made=pier]','nwr[man_made=quay]']
SESSION=requests.Session()
SESSION.headers['User-Agent']='mapauto-pbf/1.0 (https://github.com/leo1109k-7063/mapauto)'
MAX_BYTES=2*1024**3


def select_region(index,bbox):
    s,w,n,e=bbox; target=box(w,s,e,n); center=target.centroid
    choices=[]
    for f in index['features']:
        p=f.get('properties',{}); url=p.get('urls',{}).get('pbf')
        if not url or not f.get('geometry'): continue
        g=shape(f['geometry'])
        if not g.is_valid: g=g.buffer(0)
        if not g.covers(center): continue
        coverage=g.intersection(target).area/target.area
        choices.append((g.area,coverage,url,p.get('id','?')))
    if not choices: raise RuntimeError('No Geofabrik region covers map center; specify PBF_URL or PBF_PATH')
    # Prefer local region covering the request; avoid continent-sized downloads if a
    # nearby national extract covers the center but not all water/border margin.
    local=[x for x in choices if x[0]<500]
    pool=local or choices
    full=[x for x in pool if x[1]>=.99]
    chosen=min(full or pool,key=lambda x:x[0]) if full else min(pool,key=lambda x:(-x[1],x[0]))
    return chosen[2],chosen[3],chosen[1]


def source(cache,bbox):
    override=os.getenv('PBF_PATH')
    if override:
        p=Path(override).resolve()
        if not p.is_file(): raise RuntimeError('PBF_PATH does not exist: '+str(p))
        return p
    url=os.getenv('PBF_URL'); region='manual'
    if not url:
        indexpath=cache/'geofabrik-index.json'
        if not indexpath.exists() or time.time()-indexpath.stat().st_mtime>86400*7:
            r=SESSION.get(INDEX,timeout=(15,60)); r.raise_for_status(); index=r.json()
            if not isinstance(index.get('features'),list): raise RuntimeError('Invalid Geofabrik index')
            tmp=indexpath.with_suffix('.tmp'); tmp.write_text(json.dumps(index)); tmp.replace(indexpath)
        index=json.loads(indexpath.read_text())
        url,region,coverage=select_region(index,bbox)
        print(f'[PBF] region {region}, extract polygon covers {coverage:.1%} of requested bbox',flush=True)
        if coverage<.99: print('[PBF] WARNING: cross-border/sea margin may not be covered; verify completeness or set PBF_URL/PBF_PATH',flush=True)
    if urlparse(url).scheme!='https': raise RuntimeError('PBF_URL must use HTTPS')
    d=cache/'pbf'; d.mkdir(exist_ok=True)
    p=d/(hashlib.sha256(url.encode()).hexdigest()[:16]+'.osm.pbf')
    if p.exists() and p.stat().st_size>0:
        print('[PBF] reuse',p.name,flush=True); return p
    tmp=p.with_suffix('.part')
    for attempt in range(2):
        try:
            print('[PBF] download',url,flush=True)
            with SESSION.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status(); total=int(r.headers.get('Content-Length') or 0)
                if total>MAX_BYTES: raise RuntimeError('PBF exceeds 2 GiB; select smaller regional extract')
                count=0; start=time.monotonic(); last=0
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        if not chunk: continue
                        count+=len(chunk)
                        if count>MAX_BYTES or time.monotonic()-start>1200: raise RuntimeError('PBF download size/time budget exceeded')
                        f.write(chunk)
                        if count-last>=50*1024**2: print(f'[PBF] downloaded {count//1024**2} MiB',flush=True); last=count
                if not count or (total and count!=total): raise RuntimeError('Incomplete PBF download')
            # Verify readable header before caching.
            run(['osmium','fileinfo',str(tmp),'-F','pbf'],timeout=60)
            tmp.replace(p); return p
        except (requests.RequestException,OSError,RuntimeError) as ex:
            tmp.unlink(missing_ok=True)
            if attempt: raise RuntimeError(f'PBF download failed for {region}: {ex}') from ex
            time.sleep(5)


def run(args,timeout=900):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode: raise RuntimeError(f'{args[0]} {args[1]} failed: {p.stderr[-1500:]}')


def parse_selector(s):
    m=re.fullmatch(r'(nwr|way|node|rel)((?:\[[^\]]+\])+)',s)
    if not m: raise ValueError('Unsupported selector '+s)
    types={'nwr':('node','way','relation'),'way':('way',),'node':('node',),'rel':('relation',)}[m[1]]
    tests=[]
    for test in re.findall(r'\[([^\]]+)\]',m[2]):
        x=re.fullmatch(r'([^!=~]+)(?:(!=|=|~)(.*))?',test)
        if not x: raise ValueError('Unsupported tag test '+test)
        tests.append((x[1],x[2],x[3].strip('"') if x[3] is not None else None))
    return types,tests


def matches(typ,t,parsed):
    for types,tests in parsed:
        if typ not in types: continue
        def ok(k,op,v):
            if op is None: return k in t
            if op=='=': return t.get(k)==v
            if op=='!=': return t.get(k)!=v
            if op=='~': return bool(re.search(v,t.get(k,'')))
            return False
        if all(ok(*test) for test in tests): return True
    return False


def filter_expressions(parsed):
    # First test is a necessary condition. Final matching enforces every test.
    result=set()
    for types,tests in parsed:
        k,op,v=tests[0]; prefix=''.join({'node':'n','way':'w','relation':'r'}[t] for t in types)
        if op=='~' and re.fullmatch(r'\^\([a-z_|]+\)\$',v):
            result.add(f'{prefix}/{k}='+','.join(v[2:-2].split('|')))
        else: result.add(f'{prefix}/{k}'+('='+v if op=='=' else ''))
    return sorted(result)


def read_xml(path,parsed):
    nodes={}; ways={}; relations=[]; result=[]; missing=[]
    def tags(e): return {t.attrib['k']:t.attrib['v'] for t in e.findall('tag')}
    def geom(refs):
        if any(r not in nodes for r in refs): return None
        return [nodes[r] for r in refs]
    events=ET.iterparse(path,events=('start','end')); _,root=next(events)
    for event,e in events:
        if event!='end' or e.tag not in ('node','way','relation'): continue
        ident=int(e.attrib['id']); t=tags(e); selected=matches(e.tag,t,parsed)
        if e.tag=='node':
            coord={'lat':float(e.attrib['lat']),'lon':float(e.attrib['lon'])}; nodes[ident]=coord
            if selected: result.append({'type':'node','id':ident,'tags':t,**coord})
        elif e.tag=='way':
            g=geom([int(n.attrib['ref']) for n in e.findall('nd')]); ways[ident]=g
            if selected:
                if g is None: missing.append(f'way/{ident}')
                elif len(g)>=2: result.append({'type':'way','id':ident,'tags':t,'geometry':g})
        elif selected:
            relations.append((ident,t,[dict(m.attrib) for m in e.findall('member')]))
        e.clear(); root.clear()
    for ident,t,members in relations:
        ms=[]; incomplete=False
        for m in members:
            if m['type']!='way' or m.get('role','') not in ('outer','inner',''): continue
            ref=int(m['ref']); g=ways.get(ref)
            if g is None: incomplete=True; continue
            ms.append({'type':'way','ref':ref,'role':m.get('role',''),'geometry':g})
        if incomplete: missing.append(f'relation/{ident}')
        elif ms: result.append({'type':'relation','id':ident,'tags':t,'members':ms})
    if missing: raise RuntimeError(f'Incomplete OSM references ({len(missing)}): {missing[:8]}; select a more complete regional PBF')
    return result


def elements(cache,bbox,selectors):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    if not shutil.which('osmium'): raise RuntimeError('osmium-tool missing; install using apt-get install osmium-tool')
    s,w,n,e=bbox; parsed=[parse_selector(x) for x in selectors]; p=source(cache,bbox)
    key=hashlib.sha256(json.dumps([str(p),p.stat().st_size,p.stat().st_mtime_ns,list(bbox),selectors]).encode()).hexdigest()[:24]
    out=cache/('pbf-elements-v1-'+key+'.json')
    if out.exists(): return json.loads(out.read_text())['elements']
    with tempfile.TemporaryDirectory(prefix='mapauto-pbf-') as d:
        local=Path(d)/'local.osm.pbf'; filtered=Path(d)/'filtered.osm.pbf'; xml=Path(d)/'filtered.osm'
        print('[PBF] extracting bbox',bbox,flush=True)
        run(['osmium','extract','--strategy','smart','--bbox',f'{w},{s},{e},{n}','-o',str(local),str(p)])
        run(['osmium','tags-filter',str(local),*filter_expressions(parsed),'-o',str(filtered)])
        run(['osmium','cat',str(filtered),'-o',str(xml)])
        data=read_xml(xml,parsed)
    tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps({'elements':data},ensure_ascii=False)); tmp.replace(out)
    print(f'[PBF] {len(data)} selected objects',flush=True); return data
