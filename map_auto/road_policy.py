"""Preserve whole named corridors; distinguish genuine crossings from bank overlaps."""
import re, unicodedata
from collections import defaultdict
import numpy as np
from scipy import ndimage as nd
from connectivity import components, orthogonalize, connected_network, CROSS
from shapely.geometry import LineString
from skimage.morphology import skeletonize

BASE={'motorway','trunk','primary','secondary'}
LINKS={x+'_link' for x in BASE}
EXTENSIONS={'tertiary','unclassified','residential'}

def name_key(t):
    s=t.get('_route_name') or t.get('name:ko') or t.get('name') or t.get('name:en') or t.get('ref') or ''
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC',str(s)).casefold())

def is_covered(t):
    return (str(t.get('tunnel','no')).lower() not in ('no','false','0','') or t.get('location')=='underground' or t.get('covered') in ('yes','true','1') or t.get('waterway')=='culvert')

def is_road_tunnel(t):
    return bool(t.get("highway")) and (str(t.get("tunnel", "no")).lower() not in ("no","false","0","","culvert","building_passage") or t.get("location")=="underground")

def is_bridge(t):
    return str(t.get('bridge','no')).lower() not in ('no','false','0','') and not is_covered(t)

def lines(g):
    if g.geom_type in ('LineString','LinearRing'): yield g
    elif hasattr(g,'geoms'):
        for p in g.geoms: yield from lines(p)

def associate_tunnels(elements,geometry):
    """Join differently named tunnel segments to the same surface corridor only
    when exact shared geometry endpoints at both mouths identify it unambiguously.
    Do not guess from proximity or a shared highway number alone.
    """
    surface=defaultdict(set); names={}; tunnels=[]; endpoint_index=defaultdict(list)
    def ends(g):
        return {tuple(round(float(v),4) for v in p[:2]) for l in lines(g) for p in (l.coords[0],l.coords[-1])}
    for e in elements:
        t=e.get('tags',{}); k=t.get('highway'); name=name_key(t)
        if k not in BASE|EXTENSIONS|LINKS or k in LINKS: continue
        g=geometry(e)
        if g is None or g.is_empty: continue
        pts=ends(g)
        if is_road_tunnel(t):
            i=len(tunnels); tunnels.append((e,pts))
            for pt in pts: endpoint_index[pt].append(i)
        elif name:
            names[name]=t.get('name:ko') or t.get('name') or t.get('name:en') or t.get('ref')
            for pt in pts: surface[pt].add(name)
    seen=set(); aliases={}
    for i in range(len(tunnels)):
        if i in seen: continue
        stack=[i]; seen.add(i); group=[]; contacts={}
        while stack:
            j=stack.pop(); group.append(j)
            for pt in tunnels[j][1]:
                if pt in surface: contacts[pt]=surface[pt]
                for q in endpoint_index[pt]:
                    if q not in seen: seen.add(q); stack.append(q)
        if len(contacts)<2: continue
        candidates=set.intersection(*contacts.values())
        if len(candidates)!=1: continue
        key=next(iter(candidates))
        for j in group:
            e,_=tunnels[j]; aliases[(e['type'],e['id'])]=names[key]
    return aliases


def bridge_cells(g,t,water,raster,wet_geom=None):
    """An explicit bridge must span water transversely, not run along it.
    Estimate local river direction from the wet pixels near each intersection.
    A tagged bridge that doesn't meet the transverse test is a bank/elevated road.
    """
    out=np.zeros_like(water)
    if not is_bridge(t): return out
    for line in lines(g):
        hit=raster(line)&water
        labels,n,sizes=components(orthogonalize(hit,water))
        for i in range(1,n+1):
            part=labels==i; yy,xx=np.where(part)
            if not len(xx): continue
            # Local water shape, estimated around the bridge midpoint; enough context for wide rivers.
            cx,cy=float(xx.mean()),float(yy.mean()); radius=max(7,min(45,int(max(np.ptp(xx),np.ptp(yy)))+6))
            y0=max(0,int(cy)-radius); y1=min(water.shape[0],int(cy)+radius+1); x0=max(0,int(cx)-radius); x1=min(water.shape[1],int(cx)+radius+1)
            wy,wx=np.where(water[y0:y1,x0:x1]); wx=wx+x0; wy=wy+y0
            coords=np.asarray(line.coords); seg=np.diff(coords,axis=0); mid=(coords[1:]+coords[:-1])/2
            if not len(seg) or len(wx)<3: continue
            # Nearest local road segment rather than direction of the entire named road.
            from shapely.geometry import Point
            p=Point(cx+.5,cy+.5); pos=line.project(p); a=line.interpolate(max(0,pos-2)); z=line.interpolate(min(line.length,pos+2)); v=np.array([z.x-a.x,z.y-a.y]); norm=np.linalg.norm(v)
            if norm==0: continue
            v/=norm; cov=np.cov(np.stack([wx,wy])); vals,vecs=np.linalg.eigh(cov)
            # Parallel overlap is not a bridge. Ambiguous broad/isotropic water needs a span test.
            if vals[-1]>1.35*max(vals[0],.01) and abs(v@vecs[:,-1])>.75: continue
            # Require opposite-side dry land near both ends of the wet crossing, not a dead-end over water.
            span=np.array([xx,yy]).T@v; lo,hi=float(span.min()),float(span.max())
            base=np.array([cx,cy]); baseproj=base@v
            def dry(sign,edge):
                for d in range(1,6):
                    q=base+v*(edge-baseproj+sign*d); x,y=np.rint(q).astype(int)
                    if 0<=x<water.shape[1] and 0<=y<water.shape[0] and not water[y,x]: return True
                return False
            if dry(-1,lo) and dry(1,hi): out|=part
    return out

def shift_to_land(raw,land,bridge,max_shift=8,nearest=None):
    """Snap quantized bank overlap to nearby dry land; keep explicit crossing cells."""
    result=raw&(land|bridge); displaced=raw&~(land|bridge)
    if not displaced.any() or not land.any(): return result
    # Choose the bank best supported by the same source feature, not an arbitrary opposite bank.
    # For a segment with dry anchors, use that bank's local connected corridor as preference.
    dist,idx=nearest if nearest is not None else nd.distance_transform_edt(~land,return_indices=True)
    y,x=np.where(displaced & (dist<=max_shift))
    # Stay near the source segment's dry anchors where possible (important on narrow rivers).
    if len(x) and (raw&land).any():
        yy,xx=np.where(raw); y0=max(0,int(yy.min())-max_shift); y1=min(raw.shape[0],int(yy.max())+max_shift+1); x0=max(0,int(xx.min())-max_shift); x1=min(raw.shape[1],int(xx.max())+max_shift+1)
        anchor=(raw&land)[y0:y1,x0:x1]
        if anchor.any():
            ad,ai=nd.distance_transform_edt(~anchor,return_indices=True)
            use=ad[y-y0,x-x0]<=2
            result[ai[0,y[use]-y0,x[use]-x0]+y0,ai[1,y[use]-y0,x[use]-x0]+x0]=True
            y,x=y[~use],x[~use]
    result[idx[0,y,x],idx[1,y,x]]=True
    return result

def local_join(selected,allowed,max_gap=3):
    """Small local joins only, never across water or blocked terrain."""
    if not selected.any(): return selected
    close=nd.binary_dilation(selected,structure=CROSS,iterations=max_gap)&allowed
    # Connected network may use these only when joining a real selected segment.
    return close

def merge_parallel(route,land,bridge):
    # Fill only narrow gaps bracketed by the same road. Thin back to one centerline.
    dry=route&land; filled=dry.copy()
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        a=np.roll(dry,(dy,dx),(0,1)); z=np.roll(dry,(-dy,-dx),(0,1)); gap=a&z&land
        gap[[0,-1],:]=False; gap[:,[0,-1]]=False
        filled|=gap
    if np.array_equal(filled,dry): return route
    thin=skeletonize(filled)
    # Do not alter actual bridge approaches; never discard isolated original pieces.
    near=nd.binary_dilation(bridge,iterations=2)
    thin=(thin&~near)|(dry&near)|bridge
    uncovered=dry&~nd.binary_dilation(thin,structure=np.ones((3,3),bool))
    return thin|uncovered

def prune_spurs(route, protected, max_length=8):
    """Remove only short terminal branches ending at a junction of the same corridor.
    Keep genuine isolated/linear routes, boundary exits and bridges/tunnels.
    """
    result=route.copy(); protected=protected.copy()
    protected[[0,-1],:]=True; protected[:,[0,-1]]=True
    h,w=route.shape
    def neighbors(y,x):
        return [(a,b) for a,b in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=a<h and 0<=b<w and result[a,b]]
    for _ in range(3):
        degree=nd.convolve(result.astype(np.uint8),CROSS.astype(np.uint8),mode='constant')-result
        removed=0
        for y,x in zip(*np.where(result & (degree==1) & ~protected)):
            if not result[y,x]: continue
            chain=[(y,x)]; prev=None; cur=(y,x)
            while len(chain)<=max_length:
                ns=[q for q in neighbors(*cur) if q!=prev]
                if len(ns)!=1: break
                nxt=ns[0]; deg=len(neighbors(*nxt))
                if deg>=3:
                    if not any(protected[q] for q in chain):
                        for q in chain: result[q]=False
                        removed+=len(chain)
                    break
                if deg<=1 or protected[nxt]: break
                chain.append(nxt); prev,cur=cur,nxt
        if not removed: break
    return result


def simplify_route(route,land,protected):
    """Collapse nearby carriageways and small interchange loops, not long distinct sections."""
    # Closing bridges small raster gaps; restrict to actual dry land and keep tunnel/bridge cells.
    # Fill small enclosed gaps between carriageways (including irregular 1–3 cell gaps).
    dry=route&land&~protected
    filled=dry.copy()
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        for distance in (2,3,4):
            a=np.roll(dry,(dy,dx),(0,1)); z=np.roll(dry,(-(distance-1)*dy,-(distance-1)*dx),(0,1)); gap=a&z&land
            gap[:4,:]=False; gap[-4:,:]=False; gap[:,:4]=False; gap[:,-4:]=False
            for k in range(distance-1):
                filled|=np.roll(gap,(k*dy,k*dx),(0,1))&land
    # Only fill tiny enclosed holes; large loops remain real geography.
    holes=nd.binary_fill_holes(filled)&~filled&land
    labels,n,sizes=components(holes)
    filled|=holes & (sizes[labels]<=20)
    if np.array_equal(filled,dry): return route
    thin=skeletonize(filled)
    near=nd.binary_dilation(protected,structure=CROSS,iterations=2)
    thin=(thin&~near)|(dry&near)|protected
    # Preserve frame exits and any genuine portions not represented by the centerline.
    edges=np.zeros_like(dry); edges[[0,-1],:]=True; edges[:,[0,-1]]=True
    uncovered=dry & ~nd.binary_dilation(thin,structure=np.ones((3,3),bool),iterations=2)
    return thin|uncovered|(dry&edges)


def layout(objects,source,water,land,raster,frame,tunnel_allowed=None):
    """Full named corridors, simplified centerlines and only necessary ramp connectors."""
    groups=defaultdict(list)
    if tunnel_allowed is None: tunnel_allowed=land|water
    for t,g,i in objects:
        k=t.get('highway'); name=name_key(t)
        if k not in BASE|LINKS|EXTENSIONS or (k in LINKS|EXTENSIONS and not name): continue
        if k in EXTENSIONS and t.get('_ai_width') is None: continue
        if g.geom_type not in ('LineString','MultiLineString'): continue
        groups[name or f'way/{i}'].append((t,g,i))
    selected=np.zeros_like(water); bridge=np.zeros_like(water); tunnel=np.zeros_like(water); corridor=np.zeros_like(water)
    details=[]; missing=np.zeros_like(water); nearest=nd.distance_transform_edt(~land,return_indices=True); land_distance=nearest[0]
    # Build independent route masks first so true junctions with other selected roads are protected.
    records=[]; overlap=np.zeros_like(water,dtype=np.uint16)
    for key,parts in groups.items():
        route=np.zeros_like(water); raw_all=np.zeros_like(water); route_bridge=np.zeros_like(water); route_tunnel=np.zeros_like(water); ramp=np.zeros_like(water); ramp_bridge=np.zeros_like(water); ramp_tunnel=np.zeros_like(water); route_width=1
        has_main=any(t.get('highway') not in LINKS for t,g,i in parts)
        for t,g,i in parts:
            width=int(t.get('_ai_width',1)); route_width=max(route_width,width)
            raw=raster(g.simplify(.6,preserve_topology=True)); is_ramp=has_main and t.get('highway') in LINKS
            if not is_ramp: raw_all|=raw
            if is_road_tunnel(t):
                line=orthogonalize(raw,tunnel_allowed); true=np.zeros_like(water); tun=line
            else:
                true=bridge_cells(g,t,water,raster,None); line=shift_to_land(raw&(land|water),land,true,nearest=nearest)|true; tun=np.zeros_like(water)
            if is_ramp: ramp|=line; ramp_bridge|=true; ramp_tunnel|=tun
            else: route|=line; route_bridge|=true; route_tunnel|=tun
        overlap+=route
        records.append((key,parts,route,raw_all,route_bridge,route_tunnel,ramp,ramp_bridge,ramp_tunnel,route_width))
    for key,parts,route,raw_all,route_bridge,route_tunnel,ramp,ramp_bridge,ramp_tunnel,route_width in records:
        original=route.copy(); allowed=land|route_bridge|route_tunnel|ramp_bridge|ramp_tunnel
        protected=route_bridge|route_tunnel
        # Prune original short branches before closing can distort their junction.
        junction=nd.binary_dilation((overlap>0)&~original,structure=CROSS)| (overlap>1)
        route=orthogonalize(route,allowed)
        before=int(route.sum()); route=prune_spurs(route,protected|junction,max_length=8)
        pruned=before-int(route.sum())
        if len(parts)>1: route=simplify_route(route,land,protected)
        route=orthogonalize(route,allowed)
        before=int(route.sum()); route=prune_spurs(route,protected|junction,max_length=8)
        pruned+=before-int(route.sum())
        local=local_join(route,land)|route_bridge|route_tunnel|ramp
        merged,stat=connected_network(route,local,allowed,preserve=True)
        if route_width>1:
            # A two-cell strip, rather than the former three-cell cross dilation.
            merged|=nd.binary_dilation(merged,structure=np.ones((2,2),bool))&land&nd.binary_dilation(route,structure=np.ones((2,2),bool))&~route_tunnel
        selected|=merged; bridge|=route_bridge|ramp_bridge; tunnel|=route_tunnel|ramp_tunnel; corridor|=local
        lost=raw_all&~(land|water|route_tunnel); missing|=lost
        details.append({'name':str(parts[0][0].get('name') or parts[0][0].get('name:ko') or key),'source_segments':len(parts),'route_components':int(components(merged)[1]),'unplaced_nonwater_tiles':int(lost.sum()),'unplaced_water_tiles':int((raw_all&water&~(route_bridge|route_tunnel)&(land_distance>8)).sum()),'bridge_tiles':int((merged&route_bridge).sum()),'tunnel_tiles':int((merged&route_tunnel).sum()),'pruned_spur_tiles':pruned,'optional_ramp_tiles':int(ramp.sum())})
    for t,g,i in source:
        k=t.get('highway'); name=name_key(t)
        if k not in LINKS or (name and name not in groups): continue
        raw=raster(g.simplify(.6,preserve_topology=True))
        if is_road_tunnel(t):
            tun=orthogonalize(raw,tunnel_allowed); tunnel|=tun; corridor|=tun
        else:
            true=bridge_cells(g,t,water,raster,None); bridge|=true; corridor|=orthogonalize(shift_to_land(raw&(land|water),land,true,nearest=nearest)|true,land|true)
    allowed=land|bridge|tunnel
    corridor|=local_join(selected,land,max_gap=1)
    roads,stats=connected_network(selected,corridor,allowed,preserve=True)
    stats['review_required']=bool(stats['review_required'] or any(d['unplaced_nonwater_tiles']>=3 or d['unplaced_water_tiles']>=3 for d in details))
    stats['named_routes']=details; stats['route_count']=len(details); stats['unplaced_source_tiles']=int(missing.sum()); stats['pruned_spur_tiles']=sum(d['pruned_spur_tiles'] for d in details)
    frame.road_tunnels=roads&tunnel
    return roads,roads&water&bridge&~frame.road_tunnels,stats,missing
