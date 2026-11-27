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
    s=t.get('name:ko') or t.get('name') or t.get('name:en') or t.get('ref') or ''
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC',str(s)).casefold())

def is_covered(t):
    return (str(t.get('tunnel','no')).lower() not in ('no','false','0','') or t.get('location')=='underground' or t.get('covered') in ('yes','true','1') or t.get('waterway')=='culvert')

def is_bridge(t):
    return str(t.get('bridge','no')).lower() not in ('no','false','0','') and not is_covered(t)

def lines(g):
    if g.geom_type in ('LineString','LinearRing'): yield g
    elif hasattr(g,'geoms'):
        for p in g.geoms: yield from lines(p)

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

def layout(objects,source,water,land,raster,frame):
    """Road IDs remain whole; only unnamed/selected-name ramps supply short connectors.
    Any remaining disconnection is retained and explicitly reported rather than erased.
    """
    groups=defaultdict(list)
    for t,g,i in objects:
        k=t.get('highway'); name=name_key(t)
        if k not in BASE|LINKS|EXTENSIONS or (k in LINKS|EXTENSIONS and not name): continue
        if k in EXTENSIONS and t.get('_ai_width') is None: continue
        if g.geom_type not in ('LineString','MultiLineString'): continue
        groups[name or f'way/{i}'].append((t,g,i))
    selected=np.zeros_like(water); bridge=np.zeros_like(water); corridor=np.zeros_like(water)
    details=[]; missing=np.zeros_like(water); nearest=nd.distance_transform_edt(~land,return_indices=True); land_distance=nearest[0]
    for key,parts in groups.items():
        route=np.zeros_like(water); raw_all=np.zeros_like(water); route_bridge=np.zeros_like(water); route_width=1
        for t,g,i in parts:
            width=int(t.get('_ai_width',1)); route_width=max(route_width,width)
            raw=raster(g); raw_all|=raw
            true=bridge_cells(g,t,water,raster,None); route_bridge|=true
            route|=shift_to_land(raw&(land|water),land,true,nearest=nearest)
        # A small join consolidates separated carriageways/ramps within a route.
        allowed=land|route_bridge
        if len(parts)>1: route=merge_parallel(route,land,route_bridge)
        route=orthogonalize(route,allowed)
        local=local_join(route,land)|route_bridge
        # Connect selected pieces but do not discard ones that cannot safely join.
        merged,stat=connected_network(route,local,allowed,preserve=True)
        if route_width>1:
            # Width must not manufacture extra bridge cells or eat into riverbanks.
            merged|=nd.binary_dilation(merged,structure=CROSS)&land&nd.binary_dilation(route,structure=CROSS)
        selected|=merged; bridge|=route_bridge; corridor|=local
        lost=raw_all&~(land|water); missing|=lost
        details.append({'name':str(parts[0][0].get('name') or parts[0][0].get('name:ko') or key),'source_segments':len(parts),'route_components':int(components(merged)[1]),'unplaced_nonwater_tiles':int(lost.sum()),'unplaced_water_tiles':int((raw_all&water&~route_bridge&(land_distance>8)).sum()),'bridge_tiles':int(route_bridge.sum())})
    # Use genuine ramps only, rather than re-adding arbitrary fragments of deleted named roads.
    for t,g,i in source:
        k=t.get('highway'); name=name_key(t)
        if k not in LINKS or (name and name not in groups): continue
        raw=raster(g); true=bridge_cells(g,t,water,raster,None)
        bridge|=true; corridor|=orthogonalize(shift_to_land(raw&(land|water),land,true,nearest=nearest),land|true)
    allowed=land|bridge
    # Small inter-route gap correction only where selected routes meet/approach on land.
    corridor|=local_join(selected,land,max_gap=1)
    roads,stats=connected_network(selected,corridor,allowed,preserve=True)
    # Preserve selected full routes. Unresolved fragments must never earn approval.
    stats['review_required']=bool(stats['review_required'] or any(d['unplaced_nonwater_tiles']>=3 or d['unplaced_water_tiles']>=3 for d in details))
    stats['named_routes']=details
    stats['route_count']=len(details)
    stats['unplaced_source_tiles']=int(missing.sum())
    return roads,roads&water&bridge,stats,missing
