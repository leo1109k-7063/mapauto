"""4-neighbor topology repair on real rasterized feature corridors only."""
from collections import deque
import heapq
import numpy as np
from scipy import ndimage as nd
CROSS=nd.generate_binary_structure(2,1)

def components(mask):
    lab,n=nd.label(mask,structure=CROSS)
    sizes=np.bincount(lab.ravel(),minlength=n+1); sizes[0]=0
    return lab,n,sizes

def orthogonalize(mask,allowed):
    """Fill one orthogonal pixel for diagonal-only adjacent pixels (grid quantization)."""
    m=mask.copy() & allowed
    for flip in (False,True):
        v=m[:,::-1] if flip else m
        a=allowed[:,::-1] if flip else allowed
        gap=v[:-1,:-1]&v[1:,1:]&~v[:-1,1:]&~v[1:,:-1]
        y,x=np.where(gap)
        top=a[y,x+1]; bottom=a[y+1,x]&~top
        v[y[top],x[top]+1]=True; v[y[bottom]+1,x[bottom]]=True
    return m

def connected_network(selected,corridor,allowed):
    """Keep selected features connected to largest selected component, using real corridor cells.
    Unreachable components are explicitly reported; no straight invented roads/rivers.
    Added paths are one tile wide and prefer existing selected tiles.
    """
    selected=np.asarray(selected,bool)&allowed
    original=selected.copy()
    selected=orthogonalize(selected,allowed)
    corridor=orthogonalize((corridor|selected)&allowed,allowed)|selected
    labels,n,sizes=components(selected)
    if not n:
        return selected,{'before_components':0,'after_components':0,'added_tiles':0,'removed_tiles':0,'unreachable_components':0,'largest_removed_component':0,'selected_tiles':0,'review_required':False}
    root=labels==int(np.argmax(sizes)); h,w=selected.shape
    # One weighted multi-source search. A path along existing selected lines is cheaper.
    active,nc=nd.label(corridor,structure=CROSS)
    reachable=active==int(active[root][0]); keep=selected&reachable
    missing=selected&~reachable
    result=keep.copy()
    if np.any(keep&~root):
        flat=reachable.ravel(); chosen=selected.ravel(); dist=np.full(h*w,2**30,np.int32); parent=np.full(h*w,-1,np.int32)
        starts=np.flatnonzero(root.ravel()); dist[starts]=0
        # Start only at frontier of largest network, to avoid unnecessary heap entries.
        frontier=root & nd.binary_dilation(reachable&~root,structure=CROSS)
        heap=[(0,int(i)) for i in np.flatnonzero(frontier.ravel())]; heapq.heapify(heap)
        targets=set(int(x) for x in np.unique(labels[keep]) if x and x!=int(labels[root][0]))
        labflat=labels.ravel(); out=result.ravel()
        while heap and targets:
            d,i=heapq.heappop(heap)
            if d!=dist[i]: continue
            label=int(labflat[i])
            if label in targets:
                targets.remove(label); j=i
                while j>=0:
                    out[j]=True; j=int(parent[j])
            y,x=divmod(i,w)
            for j in (i-w if y else -1,i+w if y+1<h else -1,i-1 if x else -1,i+1 if x+1<w else -1):
                if j<0 or not flat[j]: continue
                ndist=d+(1 if chosen[j] else 5)
                if ndist<dist[j]:
                    dist[j]=ndist; parent[j]=i; heapq.heappush(heap,(ndist,j))
    _,after,_=components(result); _,lost_n,lost_sizes=components(missing)
    removed=int((original&~result).sum()); added=int((result&~original).sum()); total=int(original.sum())
    biggest=int(lost_sizes.max()) if lost_n else 0
    return result,{'before_components':int(components(original)[1]),'after_components':int(after),'added_tiles':added,'removed_tiles':removed,'unreachable_components':int(lost_n),'largest_removed_component':biggest,'selected_tiles':total,'review_required':bool(removed and (biggest>=40 or removed>=max(10,total*.05)))}
