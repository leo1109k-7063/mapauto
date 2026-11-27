import unittest
from unittest.mock import patch
import ai_agent as ai

class RoadRevision(unittest.TestCase):
    def setUp(self):
        self.ctx={'city':{'code':'TEST'},'fields':['id','name','classes','length_km'], 'catalog':{'road':[[f'D{i}',f'Road {i}', ['trunk' if i<4 else 'secondary'],12-i] for i in range(1,11)]}}
        self.plan={'roads':[{'id':f'D{i}','width':2} for i in range(1,11)],'rivers':[{'id':'R1','width':8,'banks':'actual'}],'airports':[{'id':'A1'}]}
    def test_ai_subset(self):
        with patch.object(ai,'call',return_value={'keep_road_ids':['D1','D4','D7']}):
            nxt=ai.next_revision({'next_plan':self.plan},self.plan,self.ctx,{'road_fraction':.09},[], 'A1','revise')
        self.assertEqual(nxt['roads'],[{'id':x,'width':1} for x in ['D1','D4','D7']])
        self.assertEqual(nxt['rivers'],self.plan['rivers']); self.assertEqual(nxt['airports'],self.plan['airports'])
    def test_repeated_selection_fallback(self):
        with patch.object(ai,'call',return_value={'keep_road_ids':[x['id'] for x in self.plan['roads']]}):
            nxt=ai.next_revision({'next_plan':self.plan},self.plan,self.ctx,{'road_fraction':.09},[], 'A1','revise')
        self.assertEqual(len(nxt['roads']),6); self.assertIn('fallback',nxt['reason']); self.assertLess(ai.road_load(nxt,self.ctx),ai.road_load(self.plan,self.ctx))
    def test_normal_revision(self):
        nxt={**self.plan,'roads':self.plan['roads'][:3]}
        with patch.object(ai,'call') as call:
            self.assertEqual(ai.next_revision({'next_plan':nxt},self.plan,self.ctx,{'road_fraction':.03},[], 'A1','revise'),nxt); call.assert_not_called()
    def test_invalid_ids(self):
        nxt=ai.thin_roads(self.plan,self.ctx,['D1','INVALID']); self.assertEqual(len(nxt['roads']),6); self.assertIn('fallback',nxt['reason'])
if __name__=='__main__': unittest.main()
