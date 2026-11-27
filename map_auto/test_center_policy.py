import unittest, tempfile
from pathlib import Path
from unittest.mock import patch
from shapely.geometry import Point, LineString, GeometryCollection
import center_policy as c
import build as b

class CenterPolicyTests(unittest.TestCase):
    def setUp(self):
        self.f=b.Frame(37.56,126.98,25)
        self.empty=GeometryCollection()
    def data(self,x,y,kind='landmark',margin=10):
        return (Point(x,y),margin,margin,kind)
    def shift(self,data,landmarks=(),facilities=(),guard=None,locked=False):
        return c.choose_shift(self.f,data,landmarks,facilities,self.empty if guard is None else guard,locked)[0]
    def test_no_justification_no_move(self):
        self.assertEqual(self.shift({'A1':self.data(-50,250,'airport')}),(0,0))
    def test_famous_within_5(self):
        x,y=self.shift({'L1':self.data(-70,250)},['L1']); self.assertAlmostEqual(x,-80); self.assertEqual(y,0)
    def test_famous_beyond_5(self):
        self.assertEqual(self.shift({'L1':self.data(-110,250)},['L1']),(0,0))
    def test_qualifying_facility_7(self):
        self.assertEqual(self.shift({'A1':self.data(-115,250,'airport')},facilities=['A1']),(0,0))
        self.assertEqual(self.shift({'A1':self.data(-135,250,'airport')},facilities=['A1']),(0,0))
    def test_diagonal_euclidean(self):
        self.assertEqual(self.shift({'L1':self.data(-70,-70)},['L1']),(0,0))
    def test_river_blocks_both(self):
        guard=LineString([(200,-100),(200,600)]).buffer(3)
        self.assertEqual(self.shift({'L1':self.data(-70,250)},['L1'],guard=guard),(0,0))
        self.assertEqual(self.shift({'A1':self.data(-115,250,'airport')},facilities=['A1'],guard=guard),(0,0))
    def test_protect_existing_landmark(self):
        self.assertEqual(self.shift({'L1':self.data(-70,250),'L2':self.data(480,250)},['L1','L2']),(0,0))
    def test_locked(self):
        self.assertEqual(self.shift({'L1':self.data(-70,250)},['L1'],locked=True),(0,0))
    def test_wide_river_detected_even_if_ai_omits(self):
        g={'R1':{'geoms':[(LineString([(200,0),(200,500)]),False)]}}
        rows=[{'id':'R1','kind':'river','name':'test','length_km':25,'bank_area_tiles':0,'known_width_m':200}]
        guard,used=c.river_guards(self.f,self.f,g,rows,set()); self.assertTrue(used[0]['automatic_wide_water']); self.assertFalse(guard.is_empty)
    def test_evidence_required_and_facility_flag(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b.pbf_source,'elements',return_value=[]):
            rows=[{'id':'A1'}]; anchors={'A1':(Point(-50,250),None,True,'Airport')}; research={'sources':[{'title':'Airport','url':'https://example.org','extract':'Airport'}]}
            fake=lambda *args,**kwargs:{'main_rivers':[],'landmarks':[],'facilities':[{'id':'A1','sources':['S1'],'important':True,'peripheral_low_importance':False,'reason':'airport'}]}
            frame,report=c.decide({},self.f,anchors,rows,research,Path(d),fake,lambda *args:({},[],[]),str)
            self.assertIs(frame,self.f);self.assertEqual(report['accepted_facilities'],[])
if __name__=='__main__': unittest.main()
