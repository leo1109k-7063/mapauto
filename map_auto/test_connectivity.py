import unittest, tempfile, json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from unittest.mock import patch
import build as b
from connectivity import connected_network,components

class NetworkTests(unittest.TestCase):
    def test_diagonal(self):
        x=np.eye(8,dtype=bool); y,s=connected_network(x,x,np.ones_like(x)); self.assertEqual(components(y)[1],1); self.assertEqual(s['removed_tiles'],0)
    def test_real_connector(self):
        x=np.zeros((12,12),bool); x[5,1:4]=True; x[5,8:11]=True
        c=x.copy(); c[5,3:9]=True
        y,s=connected_network(x,c,np.ones_like(x)); self.assertTrue(y[5,1:11].all()); self.assertEqual(s['added_tiles'],4)
    def test_no_fake_connector(self):
        x=np.zeros((30,30),bool); x[:5,:5]=True; x[20:,20:]=True
        y,s=connected_network(x,x,np.ones_like(x)); self.assertEqual(s['removed_tiles'],25); self.assertTrue(s['review_required']); self.assertEqual(components(y)[1],1)
    def test_obstacle(self):
        x=np.zeros((10,10),bool); x[3,1:3]=True; x[3,7:9]=True; c=x.copy(); c[3,:]=True; a=np.ones_like(x); a[:,5]=False
        y,s=connected_network(x,c,a); self.assertFalse(y[:,5].any()); self.assertEqual(s['removed_tiles'],2)
    def test_road_priority_and_hidden_river(self):
        f=b.Frame(37.56,126.98,25); f.ai_airports=[]
        def geom(e,frame): return e['g']
        r={'type':'way','id':1,'tags':{'waterway':'river','name':'Main','_ai_width':3},'g':LineString([(0,250),(499,250)])}
        d={'type':'way','id':2,'tags':{'highway':'primary','bridge':'yes','_ai_width':1},'g':LineString([(250,0),(250,499)])}
        with patch.object(b,'geom',side_effect=geom):
            grid,airports,ports,notes,errors,ref=b.build(f,[r,d],np.zeros((500,500)))
        self.assertFalse(errors,errors); self.assertEqual(grid[250,250],b.ROAD_RIVER); self.assertGreater(f.network_stats['road_over_river_tiles'],0); self.assertEqual(f.network_stats['logical_river_components'],1); self.assertEqual(components(np.isin(grid,[b.ROAD,b.ROAD_RIVER]))[1],1); self.assertEqual(components(grid==b.RIVER)[1],2); self.assertEqual(components(np.isin(grid,[b.RIVER,b.ROAD_RIVER]))[1],1); self.assertEqual(int((grid==b.ROAD_RIVER).sum()),int(f.river_crossings.sum())); self.assertEqual(grid[100,250],b.ROAD)
        with tempfile.TemporaryDirectory() as td:
            b.save(Path(td),{'code':'T','name':'Test'},grid,{'river_under_road':np.flatnonzero(f.river_crossings.ravel()).tolist()}); self.assertIn('const riverUnderRoad=[',Path(td,'T.html').read_text()); self.assertIn("81:'81'",Path(td,'T.html').read_text()); self.assertEqual(json.loads(Path(td,'T.json').read_text())['river_under_road'],np.flatnonzero(grid.ravel()==81).tolist()); from PIL import Image; self.assertEqual(Image.open(Path(td,'T.png')).getpixel((250,250)),(184,134,11))
    def test_full_source_supplies_unselected_road_link(self):
        f=b.Frame(37.56,126.98,25); f.ai_airports=[]
        def d(i,a,z,kind='primary'): return {'type':'way','id':i,'tags':{'highway':kind,'_ai_width':1},'g':LineString([a,z])}
        a=d(1,(20,40),(60,40)); c=d(2,(100,40),(150,40)); link=d(3,(60,40),(100,40),'primary_link'); f.network_source_elements=[a,c,link]
        with patch.object(b,'geom',side_effect=lambda e,frame:e['g']):
            g,_,_,_,e,_=b.build(f,[a,c],np.zeros((500,500)))
        self.assertFalse(e,e); self.assertTrue((g[40,20:151]==b.ROAD).all()); self.assertEqual(f.network_stats['roads']['after_components'],1)
if __name__=='__main__': unittest.main()
