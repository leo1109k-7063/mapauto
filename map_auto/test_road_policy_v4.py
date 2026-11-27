import unittest, tempfile, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from test_road_policy import e
import build as b, ai_agent as ai, road_policy as rp
from connectivity import components

class V4(unittest.TestCase):
    def setUp(self): self.f=b.Frame(37.56,126.98,25); self.f.ai_airports=[]
    def build(self,items,elev=None):
        with patch.object(b,'geom',side_effect=lambda e,f:e['g']):
            return b.build(self.f,items,np.zeros((500,500)) if elev is None else elev)
    def test_tunnel_color_and_river_connectivity(self):
        river=e(1,'river','Main',[(0,250),(499,250)],_ai_width=7)
        a=e(2,'primary','A',[(200,200),(200,240)],_ai_width=1)
        t=e(3,'primary','A',[(200,240),(200,260)],tunnel='yes',_ai_width=1)
        z=e(4,'primary','A',[(200,260),(200,300)],_ai_width=1)
        grid,_,_,_,errors,_=self.build([river,a,t,z]); self.assertFalse(errors,errors)
        self.assertEqual(grid[250,200],81); self.assertEqual(grid[220,200],8); self.assertFalse((grid==82).any())
        self.assertEqual(self.f.network_stats['visible_road_components'],1); self.assertEqual(self.f.network_stats['logical_river_components'],1)
        self.assertEqual(self.f.tunnel_underlying,{})
        with tempfile.TemporaryDirectory() as d:
            b.save(Path(d),{'code':'TEST','name':'Test'},grid,{'tunnel_underlying':self.f.tunnel_underlying})
            self.assertEqual(Image.open(Path(d)/'TEST.png').getpixel((200,250)),(184,134,11))
            self.assertIn("82:'82'",(Path(d)/'TEST.html').read_text())
            self.assertNotIn(250*500+200,json.loads((Path(d)/'TEST.json').read_text())['road_tunnel_tiles'])
    def test_actual_tunnel_through_mountain(self):
        elev=np.zeros((500,500)); elev[200:280,200:280]=400
        t=e(1,'trunk','Tunnel',[(190,240),(290,240)],tunnel='yes',_ai_width=1)
        grid,_,_,_,errors,terrain=self.build([t],elev)
        self.assertFalse(errors,errors); self.assertTrue(((terrain==3)&(grid==82)).any()); self.assertEqual(self.f.network_stats['visible_road_components'],1)
    def test_mountain_overlap_without_tunnel_tag(self):
        elev=np.zeros((500,500)); elev[200:280,200:280]=400
        t=e(1,'trunk','Road',[(190,240),(290,240)],_ai_width=1)
        grid,_,_,_,errors,terrain=self.build([t],elev)
        self.assertFalse(errors,errors)
        self.assertTrue(((terrain==3)&(grid==82)).any())
        self.assertFalse(((terrain!=3)&(grid==82)).any())
        self.assertEqual(grid[240,190],8); self.assertEqual(grid[240,290],8)
        self.assertEqual(self.f.network_stats['visible_road_components'],1)
    def test_spurs_removed_main_preserved(self):
        a=e(1,'primary','A',[(0,100),(499,100)],_ai_width=1)
        z=e(2,'primary','A',[(120,100),(120,106)],_ai_width=1)
        k=e(3,'primary','A',[(200,100),(200,104)],_ai_width=1)
        grid,*_=self.build([a,z,k]); self.assertTrue((grid[100,:]==8).all()); self.assertEqual(grid[106,120],0); self.assertEqual(grid[104,200],0)
        self.assertGreater(self.f.network_stats['roads']['pruned_spur_tiles'],0)
    def test_real_intersection_and_short_independent_road_kept(self):
        a=e(1,'primary','A',[(0,100),(499,100)],_ai_width=1)
        z=e(2,'primary','A',[(120,100),(120,106)],_ai_width=1)
        c=e(3,'primary','B',[(120,106),(126,106)],_ai_width=1)
        grid,*_=self.build([a,z,c]); self.assertEqual(grid[106,126],8); self.assertEqual(self.f.network_stats['visible_road_components'],1)
    def test_ramp_only_if_necessary(self):
        a=e(1,'primary','A',[(10,100),(80,100)],_ai_width=1)
        z=e(2,'primary','A',[(110,100),(200,100)],_ai_width=1)
        bridge=e(3,'primary_link','A',[(80,100),(110,100)],_ai_width=1)
        spur=e(4,'primary_link','A',[(40,100),(40,120)],_ai_width=1)
        grid,*_=self.build([a,z,bridge,spur]); self.assertTrue((grid[100,10:201]==8).all()); self.assertEqual(grid[120,40],0)
    def test_tunnel_alias_by_both_ends(self):
        a=e(1,'primary','Main',[(10,20),(30,20)])
        t=e(2,'trunk','Named Tunnel',[(30,20),(50,20)],tunnel='yes')
        z=e(3,'primary','Main',[(50,20),(80,20)])
        with patch.object(b,'geom',side_effect=lambda e,f:e['g']): groups,rows,rest=ai.catalog([a,t,z],self.f)
        self.assertEqual(len(groups),1); self.assertEqual(rows[0]['name'],'Main'); self.assertEqual(rows[0]['tunnel_segments'],1)
        selected=ai.choose(groups,rest,{'rivers':[],'roads':[{'id':rows[0]['id'],'width':1}]})
        grid,*_=self.build(selected); self.assertEqual(grid[20,40],8); self.assertEqual(self.f.network_stats['visible_road_components'],1)
    def test_ambiguous_tunnel_not_moved_to_wrong_name(self):
        a=e(1,'primary','Main',[(10,20),(30,20)])
        t=e(2,'trunk','Named Tunnel',[(30,20),(50,20)],tunnel='yes')
        z=e(3,'primary','Different',[(50,20),(80,20)])
        with patch.object(b,'geom',side_effect=lambda e,f:e['g']): groups,rows,rest=ai.catalog([a,t,z],self.f)
        self.assertEqual(len(groups),3)
    def test_parallel_3_tile_gap(self):
        a=e(1,'trunk','A',[(10,20),(100,20)],_ai_width=1)
        z=e(2,'trunk','A',[(10,23),(100,23)],_ai_width=1)
        grid,*_=self.build([a,z]); self.assertLessEqual((grid[:,25:85]==8).sum(),65); self.assertEqual(self.f.network_stats['visible_road_components'],1)
