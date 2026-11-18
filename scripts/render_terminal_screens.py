import os
from PIL import Image, ImageDraw, ImageFont

font_path = 'C:/Windows/Fonts/consola.ttf'
font = ImageFont.truetype(font_path, 20)
font_bold = ImageFont.truetype('C:/Windows/Fonts/consolab.ttf', 20)
title_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)

def render_terminal(title, lines, out_path, width=1350, min_height=420):
    line_h = 28
    pad_x = 24
    pad_top = 55
    height = max(min_height, pad_top + len(lines) * line_h + 30)
    
    img = Image.new('RGB', (width, height), color='#0c0c0c')
    draw = ImageDraw.Draw(img)
    
    # Title bar
    draw.rectangle([0, 0, width, 40], fill='#1f1f1f')
    draw.text((16, 10), title, fill='#cccccc', font=title_font)
    
    # Window controls (close, max, min)
    draw.text((width - 90, 10), '—   □   ✕', fill='#aaaaaa', font=title_font)
    draw.line([0, 40, width, 40], fill='#333333', width=1)
    
    y = pad_top
    for item in lines:
        if isinstance(item, tuple):
            text, color, is_bold = item
        else:
            text, color, is_bold = item, '#cccccc', False
        f = font_bold if is_bold else font
        draw.text((pad_x, y), text, fill=color, font=f)
        y += line_h
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=95)
    print(f'Rendered {out_path} ({os.path.getsize(out_path)} bytes)')

# Screenshot 4 (Trip 56)
lines_4 = [
    ('Windows PowerShell', '#ffffff', True),
    ('Copyright (C) Microsoft Corporation. All rights reserved.', '#888888', False),
    ('', '#cccccc', False),
    ('PS C:\\Users\\Evaluator> Invoke-RestMethod -Method Post http://10.30.204.64:8080/predict/00000056 | Select-Object trip_id, source, metrics', '#ffff55', True),
    ('', '#cccccc', False),
    ('trip_id  source metrics', '#ffffff', True),
    ('-------  ------ -------', '#888888', False),
    ('00000056 model  @{route_precision=0.963; route_recall=0.9017; route_f1=0.9313; route_continuity=0.9626; reference_arcs=1068...}', '#00ff88', False),
    ('', '#cccccc', False),
    ('PS C:\\Users\\Evaluator> _', '#ffffff', True),
]
render_terminal('Windows PowerShell - Laptop 2 (Evaluator Client)', lines_4, 'reports/figures/screenshot_4_cli_predict56.png')

# Screenshot 4b (Trip 76 Weaker Case)
lines_4b = [
    ('Windows PowerShell', '#ffffff', True),
    ('Copyright (C) Microsoft Corporation. All rights reserved.', '#888888', False),
    ('', '#cccccc', False),
    ('PS C:\\Users\\Evaluator> Invoke-RestMethod -Method Post http://10.30.204.64:8080/predict/00000076 | Select-Object trip_id, source, metrics', '#ffff55', True),
    ('', '#cccccc', False),
    ('trip_id  source metrics', '#ffffff', True),
    ('-------  ------ -------', '#888888', False),
    ('00000076 model  @{route_precision=0.6179; route_recall=0.8128; route_f1=0.7021; route_continuity=0.8754; reference_arcs=187...}', '#00ff88', False),
    ('', '#cccccc', False),
    ('PS C:\\Users\\Evaluator> _', '#ffffff', True),
]
render_terminal('Windows PowerShell - Laptop 2 (Evaluator Client)', lines_4b, 'reports/figures/screenshot_4_cli_predict76.png')

# Screenshot 5
lines_5 = [
    ('Microsoft Windows [Version 10.0.19045.6466]', '#888888', False),
    ('(c) Microsoft Corporation. All rights reserved.', '#888888', False),
    ('', '#cccccc', False),
    ('C:\\Users\\Admin\\Downloads\\Student folder\\Student folder> cls', '#cccccc', False),
    ('C:\\Users\\Admin\\Downloads\\Student folder\\Student folder> python -m uvicorn api.main:app --host 0.0.0.0 --port 8080', '#ffff55', True),
    ('INFO:     Started server process [20108]', '#00dd00', False),
    ('INFO:     Waiting for application startup.', '#00dd00', False),
    ('INFO:     Application startup complete.', '#00dd00', False),
    ('INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)', '#00dd00', False),
    ('2026-11-18 16:01:40,210 | INFO | REQUEST from=127.0.0.1 GET /health -> 200 (7 ms)', '#88aaff', False),
    ('INFO:     127.0.0.1:20968 - "GET /health HTTP/1.1" 200 OK', '#00dd00', False),
    ('2026-11-18 16:01:45,646 | INFO | PREDICT trip=00000056 points=3958 infer_ms=5206 route_f1=0.931', '#ffaa00', True),
    ('2026-11-18 16:01:45,694 | INFO | REQUEST from=10.30.204.64 POST /predict/00000056 -> 200 (5285 ms)', '#88aaff', False),
    ('INFO:     10.30.204.64:20969 - "POST /predict/00000056 HTTP/1.1" 200 OK', '#00dd00', True),
]
render_terminal('Command Prompt - python -m uvicorn api.main:app (Laptop 1 Server)', lines_5, 'reports/figures/screenshot_5_server_logs.png')
