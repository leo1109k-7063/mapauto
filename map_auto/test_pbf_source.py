import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import pbf_source as p

XML='''<osm version="0.6"><node id="1" lat="37.5" lon="126.8"/><node id="2" lat="37.6" lon="126.9"/><node id="3" lat="37.5" lon="127"/><node id="4" lat="37.5" lon="126.8"><tag k="aeroway" v="aerodrome"/></node><way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/></way><way id="11"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way><way id="12"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/></way><relation id="20"><member type="way" ref="12" role="outer"/><tag k="type" v="multipolygon"/><tag k="natural" v="water"/><tag k="water" v="river"/></relation></osm>'''

def feature(id,x0,y0,x1,y1):
    return {'properties':{'id':id,'urls':{'pbf':'https://example.test/'+id+'.osm.pbf'}},'geometry':{'type':'Polygon','coordinates':[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}}

class TestPBF(unittest.TestCase):
    def test_region(self):
        index={'features':[feature('wide',0,0,180,90),feature('country',125,34,130,39),feature('sub',126,37,128,38)]}
        self.assertEqual(p.select_region(index,(37.4,126.7,37.7,127.2))[1],'sub')
        self.assertEqual(p.select_region(index,(36.8,126.7,37.7,127.2))[1],'country')
    def test_xml(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.osm'; path.write_text(XML)
            data=p.read_xml(path,[p.parse_selector(s) for s in p.SELECTORS]); ids={(e['type'],e['id']) for e in data}
            self.assertEqual(ids,{('node',4),('way',10),('relation',20)})
            self.assertEqual(len(data[-1]['members'][0]['geometry']),4)
            path.write_text(XML.replace('ref="3"','ref="99"'))
            with self.assertRaisesRegex(RuntimeError,'Incomplete OSM references'): p.read_xml(path,[p.parse_selector(s) for s in p.SELECTORS])
    def test_filters(self):
        parsed=[p.parse_selector(s) for s in p.SELECTORS]
        self.assertIn('w/highway=motorway,trunk,primary',p.filter_expressions(parsed))
        self.assertFalse(p.matches('node',{'harbour':'no'},parsed))
        self.assertTrue(p.matches('way',{'waterway':'stream'},parsed))
    def test_cached_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d); src=cache/'data.pbf'; src.write_bytes(b'mock')
            calls=[]
            def fake_run(args,timeout=900):
                calls.append(args); out=Path(args[args.index('-o')+1]); out.write_text(XML if args[1]=='cat' else 'mock')
            with patch.object(p,'source',return_value=src),patch.object(p.shutil,'which',return_value='/mock/osmium'),patch.object(p,'run',side_effect=fake_run):
                a=p.elements(cache,(37,126,38,128),p.SELECTORS); b=p.elements(cache,(37,126,38,128),p.SELECTORS)
                self.assertEqual(a,b); self.assertEqual(len(calls),3)
    def test_legacy_airport_query(self):
        import build as b
        with tempfile.TemporaryDirectory() as d, patch.object(b,'CACHE',Path(d)), patch.object(b.pbf_source,'elements',return_value=[{'type':'node','id':1}]) as local, patch.object(b.SESSION,'post',side_effect=AssertionError('Overpass must not be called')):
            out=b.cached_json('airport','https://overpass-api.de/api/interpreter',post={'data':'[out:json];(nwr[aeroway=aerodrome](37.0,126.0,38.0,128.0);way[aeroway=runway](37.0,126.0,38.0,128.0););out geom;'})
            self.assertEqual(out['elements'][0]['id'],1); local.assert_called_once()

if __name__=='__main__': unittest.main()
