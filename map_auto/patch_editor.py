"""Apply small changes to original edit.html; fail safely on unexpected versions."""
from pathlib import Path
import re,sys
p=Path(sys.argv[1] if len(sys.argv)>1 else 'edit.html'); s=p.read_text(encoding='utf-8'); old=s
if 'id="btnPier"' in s: print('already patched'); sys.exit(0)
def rep(a,b,n=1):
 global s
 if s.count(a)!=n: raise SystemExit(f'Editor version mismatch ({s.count(a)} != {n}): {a[:90]} — no file changed')
 s=s.replace(a,b)
rep('항구 (5*10)','항구 (자유형)'); rep('공항 (10*50)','공항 (40*16)')
anchor='<button class="tool-btn" id="btnAirport"'
rep(anchor,'''<button class="tool-btn" id="btnPier" onclick="setBrush('pier')"><span class="color-dot" style="background:#795548;"></span> 부두 (5-1)</button>
        <button class="tool-btn" id="btnRoad" onclick="setBrush('road')"><span class="color-dot" style="background:#FFFF00;"></span> 도로 (8)</button>
        '''+anchor)
rep("airport: 'btnAirport'", "airport: 'btnAirport', pier: 'btnPier', road: 'btnRoad'")
rep('<option value="airport">공항</option>','<option value="airport">공항</option><option value="pier">부두</option><option value="road">도로</option>')
rep("if (targetType < 4) return null;","if (![4,6].includes(targetType)) return null;")
rep("['sea', 'river', 'land', 'mountain'].includes(currentTool)","['sea', 'river', 'land', 'mountain', 'port', 'pier', 'road'].includes(currentTool)")
rep('let valMap = { sea: 1, river: 2, land: 0, mountain: 3 };','let valMap = { sea: 1, river: 2, land: 0, mountain: 3, port: 5, pier: 51, road: 8 };')
rep('if (gridData[py * 500 + px] < 4) {','if (![4,6].includes(gridData[py * 500 + px])) {')
# Port branch becomes unreachable after adding free painting; remove it.
rep("            } else if (currentTool === 'port') {\n                let w = isRotated ? 10 : 5;\n                let h = isRotated ? 5 : 10;\n                fillStamp(gx, gy, w, h, 5);\n",'')
rep('let w = isRotated ? 50 : 10;','let w = isRotated ? 16 : 40;'); rep('let h = isRotated ? 10 : 50;','let h = isRotated ? 40 : 16;')
rep("else if (brushBeforeFill === 'port') newValMap = 5;","else if (brushBeforeFill === 'port') newValMap = 5;\n            else if (brushBeforeFill === 'pier') newValMap = 51;\n            else if (brushBeforeFill === 'road') newValMap = 8;")
rep('port: 5, airport: 6 };','port: 5, airport: 6, pier: 51, road: 8 };')
rep("6: '#424242'","6: '#424242', 8: '#FFFF00', 51: '#795548'",2)
rep('function fillStamp(startX, startY, w, h, typeVal) {','function fillStamp(startX, startY, w, h, typeVal) {\n            if (startX < 0 || startY < 0 || startX + w > 500 || startY + h > 500) return;')
# Keep original as backup; manual editing remains permissive, auto maps get validation.
backup=p.with_suffix('.html.bak'); backup.write_text(old,encoding='utf-8'); p.write_text(s,encoding='utf-8'); print('patched:',p,'backup:',backup)
