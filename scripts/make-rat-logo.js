// One-off: 把 icon.jpg 转成 rat_logo.png，仅保留中心圆形区域，其余透明。
const { Jimp } = require('jimp');
const path = require('path');

const SRC = path.resolve(__dirname, '..', 'miniprogram/assets/images/icon.jpg');
const DST = path.resolve(__dirname, '..', 'miniprogram/assets/images/rat_logo.png');

async function run() {
  const img = await Jimp.read(SRC);
  // 先裁成正方形再缩放到 512×512，logo 在小程序里最多显示 ~120px
  const w0 = img.bitmap.width;
  const h0 = img.bitmap.height;
  const side = Math.min(w0, h0);
  if (w0 !== h0) {
    img.crop({ x: Math.round((w0 - side) / 2), y: Math.round((h0 - side) / 2), w: side, h: side });
  }
  img.resize({ w: 256, h: 256 });

  const w = img.bitmap.width;
  const h = img.bitmap.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) / 2;
  const r2 = r * r;

  img.scan(0, 0, w, h, (x, y, idx) => {
    const dx = x + 0.5 - cx;
    const dy = y + 0.5 - cy;
    if (dx * dx + dy * dy > r2) {
      img.bitmap.data[idx + 3] = 0; // alpha = 0
    }
  });

  img.deflateLevel = 9;
  await img.write(DST);
  console.log('written', DST, w + 'x' + h);
}

run().catch((e) => { console.error(e); process.exit(1); });
