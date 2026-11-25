import tempfile,re
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from build import *

def test():
    notes=[]; sea=coast_mask([LineString([(250,-20),(250,520)])],notes)
    assert sea[:,10].all() and not sea[:,490].any(),notes
    g=np.zeros((500,500),np.uint8); g[:,:100]=SEA; g[200:207,160:500]=RIVER
    a=airport(g,(300,100),True); assert a and a[2:]==[40,16]
    assert port(g,(100,150),LineString([(100,150),(86,150)]),notes)
    g[300:303,110:490]=ROAD
    assert not validate(g,[a]),validate(g,[a])
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); save(p,{'code':'test','name':'Synthetic test'},g,{'test':True})
        m=re.search(r'const\s+gridData\s*=\s*new\s+Uint8Array\s*\(\s*\[([\d,\s]+)\]\s*\)',(p/'test.html').read_text()); assert m and len(m.group(1).split(','))==250000
    broken=g.copy(); broken[broken==PIER]=SEA; assert validate(broken,[a])
    broken=g.copy(); broken[a[1],a[0]]=LAND; assert validate(broken,[a])
    print('PASS: coast direction, airport rectangle, port/pier adjacency, terminal area, export parser, deliberate error detection')
    return g
if __name__=='__main__':
    g=test(); save(Path(__file__).resolve().parents[1]/'test_example',{'code':'synthetic','name':'가상 지형 테스트 (실제 도시 아님)'},g,{'test':True,'status':'synthetic only'})
