// 高 · 机型与配置都在这里改，index.html 不用动
window.GSR_CONFIG = {
  tier: "高",
  device: "荣耀 Magic V3",
  hardware: "骁龙 8 Gen3 / Adreno 750 · 16 GB",
  accent: "#f0a24a",     // 标题高亮色
  mode: "wipe",             // 打开时的模式：wipe 擦除 / side 并排
  left:  "默认720p.png",        // 左右默认选中的图（写文件名）
  right: "720-1080.png",

  // 10 个固定槽位：换图时同名覆盖即可，槽位名不用改
  // 默认槽位名带各自的默认分辨率（默认576p / 默认720p …），换机型时改这一行文件名即可
  shots: [
    "默认720p.png",
    "720p.png",
    { file: "1080p.png", note: "当前状态过热降频" },
    "540-720.png",
    "540-720+GSR.png",
    "540-1080.png",
    "540-1080+GSR.png",
    "720-1080.png",
    "720-1080+GSR.png",
    "1080-720.png"
  ]
};
