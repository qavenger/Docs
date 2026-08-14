/* ========== 提示词美学定义 · 阅读进度系统 ==========
 * 数据存 localStorage（键 aigc-prompt-aesthetics-v1）：
 * { last: "01-01", pages: { "01-01": {scroll, pct, read, last} } }
 * 章节页 <body data-page="01-01">；目录页 <body data-index data-total="39">
 */
(function () {
  var KEY = 'aigc-prompt-aesthetics-v1';

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch (e) { return {}; }
  }
  function save(d) {
    try { localStorage.setItem(KEY, JSON.stringify(d)); } catch (e) {}
  }

  /* ---------------- 图片灯箱 ---------------- */
  function initImageViewer() {
    var images = Array.prototype.slice.call(document.querySelectorAll('.wrap img'));
    if (!images.length) return;

    var lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', '图片查看器');
    lightbox.setAttribute('aria-hidden', 'true');
    lightbox.innerHTML =
      '<div class="lightbox-toolbar">' +
        '<span class="lightbox-count" aria-live="polite"></span>' +
        '<span class="lightbox-hint">点击图片查看原始尺寸 · ← → 切换 · Esc 关闭</span>' +
        '<button class="lightbox-btn lightbox-close" type="button" aria-label="关闭图片">×</button>' +
      '</div>' +
      '<div class="lightbox-stage">' +
        '<button class="lightbox-btn lightbox-prev" type="button" aria-label="上一张图片">‹</button>' +
        '<img class="lightbox-image" alt="">' +
        '<button class="lightbox-btn lightbox-next" type="button" aria-label="下一张图片">›</button>' +
      '</div>' +
      '<div class="lightbox-caption"></div>';
    document.body.appendChild(lightbox);

    var viewerImage = lightbox.querySelector('.lightbox-image');
    var stage = lightbox.querySelector('.lightbox-stage');
    var caption = lightbox.querySelector('.lightbox-caption');
    var count = lightbox.querySelector('.lightbox-count');
    var closeButton = lightbox.querySelector('.lightbox-close');
    var prevButton = lightbox.querySelector('.lightbox-prev');
    var nextButton = lightbox.querySelector('.lightbox-next');
    var current = 0;
    var lastFocus = null;

    function imageCaption(source) {
      var figure = source.closest ? source.closest('figure') : null;
      var figcaption = figure ? figure.querySelector('figcaption') : null;
      return figcaption ? figcaption.textContent.trim() : (source.alt || '');
    }

    function show(index) {
      current = (index + images.length) % images.length;
      var source = images[current];
      var sourceUrl = source.currentSrc || source.src;
      viewerImage.classList.remove('zoomed');
      stage.classList.remove('zoomed');
      viewerImage.style.width = '';
      viewerImage.setAttribute('data-vector', /\.svg(?:$|[?#])/i.test(sourceUrl) ? 'true' : 'false');
      viewerImage.src = sourceUrl;
      viewerImage.alt = source.alt || '';
      caption.textContent = imageCaption(source);
      count.textContent = '图片 ' + (current + 1) + ' / ' + images.length;
      prevButton.hidden = images.length < 2;
      nextButton.hidden = images.length < 2;
      stage.scrollTop = 0;
      stage.scrollLeft = 0;
    }

    function open(index) {
      lastFocus = document.activeElement;
      show(index);
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.classList.add('lightbox-open');
      closeButton.focus();
    }

    function close() {
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('lightbox-open');
      viewerImage.classList.remove('zoomed');
      stage.classList.remove('zoomed');
      viewerImage.removeAttribute('src');
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    images.forEach(function (image, index) {
      image.tabIndex = 0;
      image.setAttribute('role', 'button');
      image.setAttribute('aria-label', (image.alt || '图片') + '，点击放大');
      image.addEventListener('click', function () { open(index); });
      image.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          open(index);
        }
      });
    });

    closeButton.addEventListener('click', close);
    prevButton.addEventListener('click', function () { show(current - 1); });
    nextButton.addEventListener('click', function () { show(current + 1); });
    viewerImage.addEventListener('click', function () {
      var zooming = !viewerImage.classList.contains('zoomed');
      viewerImage.classList.toggle('zoomed', zooming);
      stage.classList.toggle('zoomed', zooming);
      viewerImage.style.width = zooming && viewerImage.getAttribute('data-vector') === 'true' ? '1600px' : '';
      viewerImage.setAttribute('aria-label', zooming ? '点击适应屏幕' : '点击查看原始尺寸');
    });
    stage.addEventListener('click', function (event) {
      if (event.target === stage) close();
    });
    lightbox.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') close();
      if (event.key === 'ArrowLeft') show(current - 1);
      if (event.key === 'ArrowRight') show(current + 1);
    });
  }

  initImageViewer();

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
