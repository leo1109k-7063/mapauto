import unittest
from unittest.mock import patch
import numpy as np
from shapely.geometry import LineString
import build as b, ai_agent as ai, road_policy as rp
from connectivity import components

def e(i,kind,name,points,**tags):
    t={'name':name,**tags}
    t['waterway' if kind=='river' else 'highway']='river' if kind=='river' else kind
    return {'id':i,'type':'way','tags':t,'g':LineString(points)}

class RoadPolicyTests(unittest.TestCase):
    def setUp(self):
        self.f=b.Frame(37.56,126.98,25); self.f.ai_airports=[]
    def build(self,selected,source=None):
        self.f.network_source_elements=source or selected
        with patch.object(b,'geom',side_effect=lambda e,f:e['g']):
            return b.build(self.f,selected,np.zeros((500,500)))
    def test_real_crossing_and_parallel_elevated(self):
        river=e(1,'river','Main',[(0,250),(499,250)],_ai_width=8)
        cross=e(2,'primary','Cross',[(200,230),(200,270)],bridge='yes',_ai_width=1)
        parallel=e(3,'trunk','Bank',[(20,249),(190,249)],bridge='yes',_ai_width=1)
        grid,*_=self.build([river,cross,parallel]); self.assertEqual(grid[250,200],81); self.assertFalse((grid[240:260,20:185]==81).any()); self.assertTrue((grid[240:260,20:185]==8).any())
    def test_bank_overlap_not_bridge(self):
        river=e(1,'river','Main',[(0,250),(499,250)],_ai_width=6)
        road=e(2,'primary','Bank',[(20,247),(400,247)],_ai_width=1)
        grid,*_=self.build([river,road]); self.assertFalse((grid==81).any()); self.assertGreater((grid==8).sum(),300)
    def test_covered_reach_logical_only(self):
        left=e(1,'river','Main',[(20,250),(100,250)],_ai_width=1)
        hidden=e(2,'river','Main',[(100,250),(200,250)],tunnel='culvert')
        right=e(3,'river','Main',[(200,250),(300,250)],_ai_width=1)
        road=e(4,'primary','Surface',[(100,250),(200,250)],_ai_width=1)
        grid,*_=self.build([left,right,road],[left,hidden,right,road]); self.assertEqual(grid[250,150],8); self.assertTrue(self.f.hidden_rivers[250,150]); self.assertEqual(self.f.network_stats['logical_river_components'],1); self.assertFalse((grid[250,110:190]==81).any())
    def test_name_group_includes_link_tunnel_and_extension(self):
        a=e(1,'primary','Same Road',[(10,20),(30,20)])
        z=e(2,'trunk','SameRoad',[(30,20),(50,20)],tunnel='yes')
        c=e(3,'tertiary','Same Road',[(50,20),(80,20)])
        d=e(4,'primary_link','Same Road',[(80,20),(90,20)])
        other=e(5,'tertiary','Different',[(90,20),(100,20)])
        with patch.object(b,'geom',side_effect=lambda e,f:e['g']): groups,rows,rest=ai.catalog([a,z,c,d,other],self.f)
        self.assertEqual(len(groups),1); ident=next(iter(groups)); chosen=ai.choose(groups,rest,{'roads':[{'id':ident,'width':1}],'rivers':[]}); self.assertEqual({x['id'] for x in chosen},{1,2,3,4})
        grid,*_=self.build(chosen); self.assertTrue(np.isin(grid[20,10:81],[8,82]).all()); self.assertEqual(grid[20,40],82); self.assertEqual(grid[20,89],0)
    def test_small_interchange_gap_preserve_both_ends(self):
        a=e(1,'primary','Same',[(10,20),(30,20)],_ai_width=1); c=e(2,'primary','Same',[(34,20),(80,20)],_ai_width=1)
        grid,*_=self.build([a,c]); self.assertTrue((grid[20,10:81]==8).all()); self.assertEqual(self.f.network_stats['visible_road_components'],1)
    def test_no_deletion_of_unreachable_named_segment(self):
        a=e(1,'primary','Same',[(10,20),(30,20)],_ai_width=1); c=e(2,'primary','Same',[(300,20),(350,20)],_ai_width=1)
        grid,*_=self.build([a,c]); self.assertEqual(grid[20,10],8); self.assertEqual(grid[20,350],8); self.assertTrue(self.f.network_stats['roads']['review_required'])
    def test_parallel_carriageways_merge(self):
        a=e(1,'trunk','Same',[(10,20),(100,20)],oneway='yes',_ai_width=1); c=e(2,'trunk','Same',[(10,22),(100,22)],oneway='yes',_ai_width=1)
        grid,*_=self.build([a,c]); self.assertEqual(self.f.network_stats['visible_road_components'],1); self.assertLessEqual((grid[:,25:85]==8).sum(),65)
    def test_evidence_override_requires_real_quote(self):
        import tempfile
        from pathlib import Path
        a=e(1,'river','Main',[(10,20),(30,20)])
        a['geometry']=[{'lat':37.5,'lon':126.9},{'lat':37.51,'lon':126.9}]
        source={'title':'Main','url':'https://example.test/source','extract':'Main river is covered between Alpha and Beta.'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(b,'geom',side_effect=lambda e,f:e['g']), patch.object(ai,'call',return_value={'covered':[{'id':'1','source_id':0,'quote':'Invented covered quote','reason':'Match exact endpoints' }]}):
            result=ai.surface_evidence([a],self.f,{'sources':[source]},Path(tmp)); self.assertFalse(result['applied']); self.assertFalse(rp.is_covered(a['tags']))
        with tempfile.TemporaryDirectory() as tmp, patch.object(b,'geom',side_effect=lambda e,f:e['g']), patch.object(ai,'call',return_value={'covered':[{'id':'1','source_id':0,'quote':source['extract'],'reason':'Match exact endpoints'}]}):
            result=ai.surface_evidence([a],self.f,{'sources':[source]},Path(tmp)); self.assertEqual(len(result['applied']),1); self.assertTrue(rp.is_covered(a['tags']))
    def test_deleted_named_road_not_connector(self):
        a=e(1,'primary','A',[(10,20),(30,20)],_ai_width=1); c=e(2,'primary','B',[(100,20),(120,20)],_ai_width=1); deleted=e(3,'primary','Deleted',[(30,20),(100,20)])
        grid,*_=self.build([a,c],[a,c,deleted]); self.assertEqual(grid[20,60],0); self.assertTrue(self.f.network_stats['roads']['review_required'])

if __name__=='__main__': unittest.main()
