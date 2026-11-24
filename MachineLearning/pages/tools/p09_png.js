// p09_png.js — 第 09 篇原型用的极简 PNG 输出（Node 自带 zlib）：把灰度小图网格 / RGB 缓冲写成 PNG，便于在原型阶段直接看结果。
const zlib = require('zlib'), fs = require('fs');
const CRC = (() => { const t = new Int32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c; } return t; })();
function crc32(buf) { let c = -1; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; }
function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
}
/** rgb: Uint8Array(W·H·3) */
function writeRGB(file, rgb, W, H) {
    const raw = Buffer.alloc((W * 3 + 1) * H);
    for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; for (let x = 0; x < W * 3; x++) raw[y * (W * 3 + 1) + 1 + x] = rgb[y * W * 3 + x]; }
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}
/** 灰度小图网格：imgs = [Float32Array(w·h)]（0..1），每张放大 scale 倍 */
function writeGrid(file, imgs, w, h, cols, scale = 4, gap = 2, range = [0, 1]) {
    const rows = Math.ceil(imgs.length / cols), W = cols * (w * scale + gap) + gap, H = rows * (h * scale + gap) + gap, rgb = new Uint8Array(W * H * 3).fill(40);
    imgs.forEach((im, k) => {
        const ox = gap + (k % cols) * (w * scale + gap), oy = gap + Math.floor(k / cols) * (h * scale + gap);
        for (let y = 0; y < h * scale; y++) for (let x = 0; x < w * scale; x++) {
            let v = (im[Math.floor(y / scale) * w + Math.floor(x / scale)] - range[0]) / (range[1] - range[0]); v = Math.max(0, Math.min(1, v));
            const o = ((oy + y) * W + ox + x) * 3; rgb[o] = rgb[o + 1] = rgb[o + 2] = Math.round(255 * v);
        }
    });
    writeRGB(file, rgb, W, H);
}
const PAL = [[106, 176, 224], [224, 160, 96], [90, 208, 122], [198, 120, 221], [224, 108, 117], [86, 182, 194], [232, 194, 90], [209, 154, 102], [150, 150, 150], [240, 240, 240]];
/** 散点：pts = [[x, y, cls]]，画到 S×S 画布 */
function writeScatter(file, pts, S = 400, r = 2) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    pts.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
    const rgb = new Uint8Array(S * S * 3).fill(16), sx = (S - 20) / (x1 - x0 || 1), sy = (S - 20) / (y1 - y0 || 1);
    pts.forEach(([x, y, c]) => {
        const px = Math.round(10 + (x - x0) * sx), py = Math.round(S - 10 - (y - y0) * sy), col = PAL[c % 10];
        for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const X = px + dx, Y = py + dy; if (X >= 0 && X < S && Y >= 0 && Y < S && dx * dx + dy * dy <= r * r) { const o = (Y * S + X) * 3; rgb[o] = col[0]; rgb[o + 1] = col[1]; rgb[o + 2] = col[2]; } }
    });
    writeRGB(file, rgb, S, S);
    return { x0, x1, y0, y1 };
}
module.exports = { writeRGB, writeGrid, writeScatter, PAL };
