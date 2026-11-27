import json, unittest, os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from shapely.geometry import LineString
import ai_agent as ai

class ReviewV2(unittest.TestCase):
    def setUp(self):
        self.ctx={'city':{'code':'TEST'},'fields':[], 'catalog':{'road':[[f'D{i}',f'Road {i}',['trunk'],10] for i in range(1,11)],'river':[['R1','Main',['river'],10,1000],['R2','Minor',['stream'],2,10]]}}
        self.plan={'roads':[{'id':f'D{i}','width':1} for i in range(1,11)],'rivers':[{'id':'R1','width':8,'banks':'actual'},{'id':'R2','width':2,'banks':'actual'}],'airports':[{'id':'A1'}]}
    def test_below_threshold_road_complaint(self):
        with patch.object(ai,'call',return_value={'keep_road_ids':['D1','D2','D3']}) as call:
            nxt=ai.next_revision({'issues':['Road density is excessive'],'next_plan':self.plan},self.plan,self.ctx,{'road_fraction':.049},[],'A1','revise')
        self.assertEqual(len(nxt['roads']),3); self.assertEqual(nxt['rivers'],self.plan['rivers']); self.assertEqual(call.call_count,1)
    def test_both_repairs_and_main_protection(self):
        with patch.object(ai,'call',side_effect=[{'keep_road_ids':['D1','D2']},{'rivers':[{'id':'R2','width':1,'banks':'narrow'}]}]):
            nxt=ai.next_revision({'issues':['Road density excessive','Minor tributaries need revision'],'next_plan':self.plan},self.plan,self.ctx,{'road_fraction':.049,'connectivity':{'rivers':{'review_required':True}}},[],'A1','revise')
        self.assertEqual(len(nxt['roads']),2); self.assertIn(self.plan['rivers'][0],nxt['rivers']); self.assertIn({'id':'R2','width':1,'banks':'narrow'},nxt['rivers']);self.assertEqual(nxt['airports'],self.plan['airports'])
    def test_invalid_revision_does_not_invent(self):
        with patch.object(ai,'call',side_effect=[{'rivers':[{'id':'FAKE','width':1}]},{'next_plan':None}]):
            nxt=ai.next_revision({'issues':['River connectivity'],'next_plan':self.plan},self.plan,self.ctx,{'road_fraction':.02},[],'A1','revise')
        self.assertIsNone(nxt)
    def test_feature_loss(self):
        class F: pass
        f=F(); g=LineString([(10,10),(20,10)]); m=ai.b.line_mask(g,1); f.network_removed={'road':m,'river':np.zeros_like(m)}
        got=ai.feature_loss({'D1':{'geoms':[(g,False)],'name':'Example'}},self.plan,f)
        self.assertGreater(got['roads'][0]['overlap_removed_tiles'],0);self.assertEqual(got['roads'][0]['id'],'D1')
    def test_real_report_replay(self):
        path=os.getenv('REVIEW_FIXTURE')
        if not path: self.skipTest('Optional external report fixture')
        p=Path(path);ctx=json.loads((p/'catalog.json').read_text());h=json.loads((p/'history.json').read_text())[-1];plan=h['plan']; metrics=h['metrics'];rivers=plan['rivers'][:-1]
        with patch.object(ai,'call',side_effect=[{'keep_road_ids':[x['id'] for x in plan['roads'][:30]]},{'rivers':rivers}]):
            nxt=ai.next_revision(h['review'],plan,ctx,metrics,[],'A2','revise')
        self.assertEqual(len(nxt['roads']),30);self.assertEqual(len(nxt['rivers']),13); self.assertIn(plan['rivers'][0],nxt['rivers'])
if __name__=='__main__': unittest.main()
