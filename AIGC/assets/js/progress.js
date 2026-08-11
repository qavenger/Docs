/* ========== AIGC 手册 · 阅读进度系统 ==========
 * 数据存 localStorage（键 aigc-reader-v1）：
 * { last: "01-01", pages: { "01-01": {scroll, pct, read, last} } }
 * 章节页 <body data-page="01-01">；目录页 <body data-index data-total="13">
 */
(function () {
  var KEY = 'aigc-reader-v1';

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch (e) { return {}; }
  }
  function save(d) {
    try { localStorage.setItem(KEY, JSON.stringify(d)); } catch (e) {}
  }

  /* ---------------- 章节页 ---------------- */
  var pageId = document.body.getAttribute('data-page');
  if (pageId) {
    var bar = document.querySelector('.readbar > div');
    var toggle = document.querySelector('.read-toggle');

    function paint(cur) {
      if (bar) bar.style.width = (cur.pct || 0) + '%';
      if (toggle) {
        toggle.classList.toggle('done', !!cur.read);
        toggle.textContent = cur.read ? '✓ 已读完' : '标记为已读';
      }
    }

    var d0 = load();
    var cur0 = (d0.pages && d0.pages[pageId]) || {};
    paint(cur0);

    // 恢复上次滚动位置（超过一屏才恢复，避免打扰）
    if (cur0.scroll && cur0.scroll > 400 && !cur0.read) {
      setTimeout(function () { window.scrollTo(0, cur0.scroll); }, 60);
    }

    var timer = null;
    window.addEventListener('scroll', function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        var h = document.documentElement;
        var max = h.scrollHeight - h.clientHeight;
        var pct = max > 0 ? Math.round((h.scrollTop / max) * 100) : 100;
        var d = load(); d.pages = d.pages || {};
        var cur = d.pages[pageId] || {};
        cur.scroll = h.scrollTop;
        cur.pct = Math.max(cur.pct || 0, Math.min(100, pct));
        if (pct >= 92) cur.read = true;
        cur.last = Date.now();
        d.pages[pageId] = cur;
        d.last = pageId;
        save(d);
        paint(cur);
      }, 180);
    }, { passive: true });

    if (toggle) {
      toggle.addEventListener('click', function () {
        var d = load(); d.pages = d.pages || {};
        var cur = d.pages[pageId] || {};
        cur.read = !cur.read;
        if (cur.read) cur.pct = 100;
        cur.last = Date.now();
        d.pages[pageId] = cur; d.last = pageId;
        save(d); paint(cur);
      });
    }
    return;
  }

  /* ---------------- 目录页 ---------------- */
  if (document.body.hasAttribute('data-index')) {
    var d = load();
    var pages = d.pages || {};
    var cards = document.querySelectorAll('.chapter-card[data-chapter]');
    var total = cards.length, readCount = 0, pctSum = 0;

    cards.forEach(function (card) {
      var id = card.getAttribute('data-chapter');
      var cur = pages[id] || {};
      var status = card.querySelector('.status');
      var pctEl = card.querySelector('.c-pct');
      if (cur.read) {
        card.classList.add('read');
        if (status) status.textContent = '✓';
        if (pctEl) pctEl.textContent = '已读完';
        readCount++; pctSum += 100;
      } else if (cur.pct > 3) {
        card.classList.add('reading');
        if (status) status.textContent = '…';
        if (pctEl) pctEl.textContent = '读到 ' + (cur.pct || 0) + '%';
        pctSum += cur.pct || 0;
      } else {
        if (pctEl) pctEl.textContent = '未读';
      }
    });

    var overall = total ? Math.round(pctSum / total) : 0;
    var barEl = document.querySelector('.overall .bar > div');
    var metaEl = document.querySelector('.overall .meta span');
    if (barEl) barEl.style.width = overall + '%';
    if (metaEl) metaEl.textContent = '已读完 ' + readCount + ' / ' + total + ' 章 · 总进度 ' + overall + '%';

    // “继续阅读”按钮：优先回到最后读的页，否则第一个未读完的章节
    var btn = document.querySelector('.continue-btn');
    if (btn) {
      var target = null;
      if (d.last && pages[d.last] && !pages[d.last].read) {
        target = document.querySelector('.chapter-card[data-chapter="' + d.last + '"]');
      }
      if (!target) {
        for (var i = 0; i < cards.length; i++) {
          var cid = cards[i].getAttribute('data-chapter');
          if (!(pages[cid] && pages[cid].read)) { target = cards[i]; break; }
        }
      }
      if (target) {
        btn.href = target.getAttribute('href');
        btn.textContent = (readCount === 0 ? '开始阅读' : '继续阅读') + ' →';
      } else {
        btn.textContent = '全部读完，重温一遍 →';
        if (cards.length) btn.href = cards[0].getAttribute('href');
      }
    }

    var reset = document.querySelector('.reset-link');
    if (reset) {
      reset.addEventListener('click', function () {
        if (confirm('确定要清空全部阅读进度吗？')) {
          localStorage.removeItem(KEY);
          location.reload();
        }
      });
    }
  }
})();
