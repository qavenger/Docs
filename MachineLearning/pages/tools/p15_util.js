// p15_util.js — 第 15 篇的 Node 工具：加载 15_nerf_gaussian_splatting.lab.js 里的 P15 命名空间（与页面同一份代码）+ PNG 输出
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../15_nerf_gaussian_splatting.lab.js'), 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const P15 = new Function('LM', head + '\nreturn P15;')(LM);

const CRC = (() => { const t = new Int32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c; } return t; })();
function crc32(buf) { let c = -1; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; }
function chunk(type, data) { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td)); return Buffer.concat([len, td, crc]); }
/** rgb: Float32Array W*H*3 in 0..1 (linear display values) → PNG file, upscaled by `scale` (nearest) */
function savePNG(file, rgb, W, H, scale = 4) {
    const w = W * scale, h = H * scale, raw = Buffer.alloc((w * 3 + 1) * h);
    for (let y = 0; y < h; y++) {
        raw[y * (w * 3 + 1)] = 0;
        for (let x = 0; x < w; x++) {
            const s = ((y / scale | 0) * W + (x / scale | 0)) * 3, o = y * (w * 3 + 1) + 1 + x * 3;
            for (let c = 0; c < 3; c++) raw[o + c] = Math.max(0, Math.min(255, Math.round(rgb[s + c] * 255)));
        }
    }
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}
/** tile several W×H rgb images into one row-major grid */
function mosaic(imgs, W, H, cols) {
    const rows = Math.ceil(imgs.length / cols), out = new Float32Array(cols * W * rows * H * 3);
    imgs.forEach((im, k) => { const ox = (k % cols) * W, oy = Math.floor(k / cols) * H; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) out[((oy + y) * cols * W + ox + x) * 3 + c] = im[(y * W + x) * 3 + c]; });
    return { rgb: out, W: cols * W, H: rows * H };
}
module.exports = { P15, LM, savePNG, mosaic, fs, path };
