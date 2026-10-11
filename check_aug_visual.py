import glob, json, os, re
import numpy as np
from PIL import Image, ImageDraw

d = 'data/aug_check_offline'
files = sorted(glob.glob(f'{d}/train/img/*aug0_*.png'))
f = files[len(files) // 2]               # change the index to look at other slices
name = os.path.basename(f)
stem = name[:-4]
orig = name.replace('aug0', '')

log = json.load(open(f'{d}/aug_log.json'))
# per-volume mode logs one entry per patient copy, per-slice mode one per slice
p = log.get(stem) or log[re.sub(r'_\d+$', '', stem)]

sp = p['spatial']
lines = [stem,
         ('spatial: angle %.1f deg, shift (%.1f, %.1f) px, scale %.3f' %
          (sp['angle_deg'], sp['tx_px'], sp['ty_px'], sp['scale'])) if sp else 'spatial: none',
         ('intensity shift: %+.1f' % p['shift']) if p['shift'] is not None else 'intensity shift: none',
         'gaussian noise: ' + ('yes' if p['noise'] else 'no')]
print('\n'.join(lines))

o = np.asarray(Image.open(f'{d}/train/img/{orig}').convert('L'))
a = np.asarray(Image.open(f'{d}/train/img/{name}').convert('L'))
g = np.asarray(Image.open(f'{d}/train/gt/{name}'))

rgb = np.stack([a, a, a], -1).astype(np.float32)
m = g > 0
rgb[m] = 0.6 * rgb[m] + 0.4 * np.array([255, 0, 0])
panel = np.concatenate([np.stack([o, o, o], -1), rgb.astype(np.uint8)], 1)

strip = Image.new('RGB', (panel.shape[1], 14 * len(lines) + 8), 'white')
dr = ImageDraw.Draw(strip)
for i, t in enumerate(lines):
    dr.text((6, 4 + 14 * i), t, fill='black')
Image.fromarray(np.concatenate([panel, np.asarray(strip)], 0)).save('aug_check.png')