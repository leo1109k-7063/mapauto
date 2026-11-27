import unittest
import numpy as np
from shapely.geometry import Point, LineString
import build as b
import center_policy as c

class FacilityTests(unittest.TestCase):
    def setUp(self):
        self.f=b.Frame(37.56,126.98,25)
        self.g=np.zeros((500,500),np.uint8)
    def test_airport_relocation_no_center_shift(self):
        p=Point(-78,266)
        a=b.relocate_airport(self.g,self.f,p,False)
        self.assertIsNotNone(a); self.assertEqual(a[2:],[16,40]); self.assertEqual(self.f.lon,126.98)
        q=Point(a[0]+8,a[1]+20)
        self.assertLessEqual(b.relocation_record(self.f,'Gimpo','airport',p,q)['distance_m'],7000)
    def test_airport_too_far(self):
        self.assertIsNone(b.relocate_airport(self.g,self.f,Point(-150,250),True))
    def test_river_blocks_airport(self):
        self.f.facility_guard=LineString([(-20,-500),(-20,1000)]).buffer(1)
        self.assertIsNone(b.relocate_airport(self.g,self.f,Point(-50,250),True))
    def test_port_relocation(self):
        self.g[:,:60]=b.SEA; p=Point(60,-40)
        q=b.relocate_port(self.g,self.f,p,[])
        self.assertIsNotNone(q); self.assertFalse(b.validate(self.g,[]))
        self.assertLessEqual(b.relocation_record(self.f,'Port','port',p,q)['distance_m'],7000)
    def test_port_inland(self):
        self.assertIsNone(b.relocate_port(self.g,self.f,Point(-20,250),[]))
    def test_port_river_guard(self):
        self.g[:,:60]=b.SEA; self.f.facility_guard=LineString([(-500,-20),(1000,-20)]).buffer(1)
        self.assertIsNone(b.relocate_port(self.g,self.f,Point(60,-40),[]))
    def test_no_unapproved_airport(self):
        self.f.ai_airports=[(Point(-50,250),None,True,'Airport')]
        g,a,_,_,_,_=b.build(self.f,[],np.zeros((500,500)))
        self.assertEqual(a,[]); self.assertEqual(self.f.facility_unresolved,['Airport'])
    def test_approved_airport(self):
        self.f.ai_airports=[(Point(-50,250),None,True,'Airport')]; self.f.airport_relocation_names={'Airport'}
        g,a,_,_,e,_=b.build(self.f,[],np.zeros((500,500)))
        self.assertEqual(len(a),1); self.assertFalse(e); self.assertEqual(len(self.f.facility_relocations),1)
if __name__=='__main__': unittest.main()
