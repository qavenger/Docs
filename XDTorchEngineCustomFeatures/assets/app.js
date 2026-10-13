/* XD 引擎特性文档 - 交互脚本 */
(function(){
  // 主题
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('xd-doc-theme'); } catch(e){}
  if (saved) root.setAttribute('data-theme', saved);
  else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) root.setAttribute('data-theme','dark');
  var btn = document.getElementById('themeBtn');
  if (btn) btn.addEventListener('click', function(){
    var cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', cur);
    try { localStorage.setItem('xd-doc-theme', cur); } catch(e){}
  });

  // 列表筛选
  Array.prototype.forEach.call(document.querySelectorAll('input.filter'), function(inp){
    var scope = document.querySelector(inp.getAttribute('data-scope') || 'body');
    inp.addEventListener('input', function(){
      var q = inp.value.trim().toLowerCase();
      Array.prototype.forEach.call(scope.querySelectorAll('.macro, tr, .idxline'), function(el){
        var t = (el.textContent || '').toLowerCase();
        el.classList.toggle('hide', q !== '' && t.indexOf(q) < 0);
      });
      Array.prototype.forEach.call(scope.querySelectorAll('h2.author'), function(h){
        var sib = h.nextElementSibling, any = false;
        while (sib && !(sib.classList && sib.classList.contains('author'))){
          if (!sib.classList.contains('hide')) { any = true; break; }
          sib = sib.nextElementSibling;
        }
        h.style.display = (!q || any) ? '' : 'none';
      });
    });
  });

  // 锚点高亮
  if (location.hash) {
    var el = document.querySelector(location.hash.replace(/[^A-Za-z0-9_#\-\.]/g,''));
    if (el) { el.style.transition='box-shadow .4s'; el.style.boxShadow='0 0 0 3px rgba(224,93,43,.45)';
      setTimeout(function(){ el.style.boxShadow=''; }, 2200); }
  }

  // 全局搜索页
  var gs = document.getElementById('globalSearch');
  if (gs && window.XD_DATA) {
    var render = function(q){
      q = q.trim().toLowerCase();
      var box = document.getElementById('searchResults');
      if (!q) { box.innerHTML = '<p class="lead">输入关键字后开始搜索（宏名 / 中文说明 / 作者 / 文件 / 主题）。</p>'; return; }
      var keys = Object.keys(window.XD_DATA), hits = [];
      keys.forEach(function(k){
        var d = window.XD_DATA[k];
        var hay = (k + ' ' + (d.zh||'') + ' ' + (d.n||'') + ' ' + (d.mn||'') + ' ' + (d.a||'') + ' ' + (d.f||'') + ' ' + (d.t||'')).toLowerCase();
        if (hay.indexOf(q) >= 0) hits.push(k);
      });
      hits.sort(function(a,b){ return (window.XD_DATA[b].u||0) - (window.XD_DATA[a].u||0); });
      var html = '<p class="lead">共命中 ' + hits.length + ' 个宏（按源码引用数排序，最多显示 300 条）</p>';
      var esc = function(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;'); };
      var MZ = {covered:'UE5 已覆盖', partial:'部分覆盖', none:'未覆盖', unknown:'待确认'};
      var DZ = {drop:'删除', patch:'常开补丁', config:'配置化', plugin:'插件化', hook:'接入点', port:'移植', keep:'保留引擎'};
      hits.slice(0, 300).forEach(function(k){
        var d = window.XD_DATA[k];
        var tp = (window.XD_TOPICS && window.XD_TOPICS[d.t]) ? window.XD_TOPICS[d.t].title : d.t;
        var href = window.XD_REFMAP ? ((window.XD_BASE || '') + (window.XD_REFMAP[k] || '#')) : '#';
        var zh = esc(d.zh);
        var mig = '';
        if (d.mu) {
          mig = '<span class="badge topic">' + (MZ[d.mu]||d.mu) + '</span><span class="badge">' + (DZ[d.md]||d.md) + '</span><span class="badge">侵入 L' + (d.mh||0) + '</span>';
        }
        var re = new RegExp('(' + q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&') + ')', 'ig');
        html += '<article class="macro"><h3 class="mname"><a href="' + href + '#' + k + '">' + k +
                '</a> <span class="v ' + (d.v==='1'?'on':(d.v==='0'?'off':'num')) + '">= ' + d.v + '</span></h3>' +
                '<div class="badges"><span class="badge topic">' + esc(tp) + '</span>' + mig + '<span class="badge">' + esc(d.a||'') +
                '</span><span class="badge use">引用 ' + (d.u||0) + '</span></div>' +
                '<p class="zh">' + zh.replace(re,'<mark>$1</mark>') + '</p></article>';
      });
      box.innerHTML = html;
    };
    gs.addEventListener('input', function(){ render(gs.value); });
    render('');
  }
})();

/* UE5 迁移矩阵筛选 */
(function(){
  var sels = document.querySelectorAll('select.migfilter');
  if (!sels.length) return;
  var apply = function(){
    var rows = document.querySelectorAll('#migmatrix tbody tr');
    var active = {};
    Array.prototype.forEach.call(sels, function(s){ if (s.value) active[s.getAttribute('data-attr')] = s.value; });
    var shown = 0;
    Array.prototype.forEach.call(rows, function(r){
      var ok = true;
      for (var k in active) { if ((r.getAttribute('data-' + k) || '') !== active[k]) { ok = false; break; } }
      r.classList.toggle('hide', !ok);
      if (ok) shown++;
    });
    var el = document.getElementById('migcount'); if (el) el.textContent = shown;
  };
  Array.prototype.forEach.call(sels, function(s){ s.addEventListener('change', apply); });
  apply();
})();

/* ---- 标签说明面板：全局常驻 ---- */
(function(){
  var el = document.getElementById('legend');
  if (!el) return;
  var btn = document.getElementById('legendToggle');
  var mq = window.matchMedia ? window.matchMedia('(max-width:1360px)') : {matches:false};
  var KEY = 'xd-doc-legend';
  var pref = null;
  try { pref = localStorage.getItem(KEY); } catch(e){}
  var open = pref === null ? true : pref === '1';
  var narrowOpen = false;
  var fab = document.createElement('button');
  fab.id = 'legendFab'; fab.type = 'button'; fab.textContent = '标签说明';
  fab.title = '查看页面标签的含义';
  document.body.appendChild(fab);
  var apply = function(){
    var isNarrow = !!mq.matches;
    el.classList.toggle('collapsed', !open && !isNarrow);
    el.classList.toggle('show', isNarrow && narrowOpen);
    if (btn) btn.textContent = open ? '－' : '＋';
    fab.textContent = (isNarrow && narrowOpen) ? '× 关闭' : '标签说明';
  };
  var toggle = function(){
    if (mq.matches) { narrowOpen = !narrowOpen; }
    else { open = !open; try { localStorage.setItem(KEY, open ? '1' : '0'); } catch(e){} }
    apply();
  };
  if (btn) btn.addEventListener('click', toggle);
  fab.addEventListener('click', toggle);
  if (mq.addEventListener) mq.addEventListener('change', apply);
  else if (mq.addListener) mq.addListener(apply);
  apply();
})();

/* ---- 页签（tabs）+ 矩阵交叉筛选 ---- */
(function(){
  // tabs
  Array.prototype.forEach.call(document.querySelectorAll('[data-tabs]'), function(root){
    var btns = root.querySelectorAll('.tabbtn[data-tab]');
    var panes = root.querySelectorAll('.tabpane[data-pane]');
    var activate = function(key, scroll){
      Array.prototype.forEach.call(btns, function(b){ b.classList.toggle('active', b.getAttribute('data-tab') === key); });
      Array.prototype.forEach.call(panes, function(p){ p.classList.toggle('active', p.getAttribute('data-pane') === key); });
      if (scroll && root.scrollIntoView) root.scrollIntoView({block:'start'});
    };
    Array.prototype.forEach.call(btns, function(b){
      b.addEventListener('click', function(){
        var k = b.getAttribute('data-tab');
        activate(k, false);
        if (history.replaceState) history.replaceState(null, '', '#' + k);
      });
    });
    Array.prototype.forEach.call(document.querySelectorAll('.tocchips a[data-tab]'), function(a){
      a.addEventListener('click', function(ev){
        ev.preventDefault();
        activate(a.getAttribute('data-tab'), true);
        if (history.replaceState) history.replaceState(null, '', '#' + a.getAttribute('data-tab'));
      });
    });
    var h = (location.hash || '').replace('#', '');
    if (h) {
      var hit = null;
      Array.prototype.forEach.call(btns, function(b){ if (b.getAttribute('data-tab') === h) hit = h; });
      if (hit) activate(hit, false);
    }
  });
  // 矩阵交叉筛选
  var mx = document.getElementById('mxmatrix');
  if (mx) {
    var detail = document.getElementById('mxdetail');
    var info = document.getElementById('mxinfo');
    var clearBtn = document.getElementById('mxclear');
    var apply = function(owner, mod, isOther){
      var rows = detail.querySelectorAll('tbody tr');
      var shown = 0;
      Array.prototype.forEach.call(rows, function(r){
        var ok = (!owner || r.getAttribute('data-owner') === owner) &&
                 (isOther ? r.getAttribute('data-other') === '1'
                          : (!mod || r.getAttribute('data-module') === mod));
        r.classList.toggle('hide', !ok);
        if (ok) shown++;
      });
      detail.style.display = '';
      if (info) info.textContent = '当前筛选：' + (owner || '全部开发者') + ' × ' + (isOther ? '其他模块' : (mod || '全部模块')) + '，共 ' + shown + ' 个宏';
      Array.prototype.forEach.call(mx.querySelectorAll('td.mx'), function(td){ td.classList.remove('active'); });
    };
    Array.prototype.forEach.call(mx.querySelectorAll('td.mx'), function(td){
      td.addEventListener('click', function(){
        td.classList.add('active');
        apply(td.getAttribute('data-owner'), td.getAttribute('data-module'), td.getAttribute('data-other') === '1');
      });
    });
    if (clearBtn) clearBtn.addEventListener('click', function(){ apply('', '', false); });
  }
})();

/* ---- 表格横向溢出时才启用内部滚动（否则保持表头 sticky） ---- */
(function(){
  var wraps = document.querySelectorAll('.tablewrap');
  if (!wraps.length) return;
  var syncWrap = function(){
    Array.prototype.forEach.call(wraps, function(w){
      var t = w.querySelector('table');
      if (!t) return;
      var need = t.scrollWidth > w.clientWidth + 2;
      w.classList.toggle('scrolling', need);
    });
  };
  syncWrap();
  window.addEventListener('resize', syncWrap);
  setTimeout(syncWrap, 300);
})();

/* ---- 底部统计条折叠 ---- */
(function(){
  var sb = document.getElementById('statbar');
  if (!sb) return;
  var btn = document.getElementById('sbToggle');
  var KEY = 'xd-doc-statbar';
  var open = true;
  try { var v = localStorage.getItem(KEY); if (v === '0') open = false; } catch(e){}
  var sync = function(){
    sb.classList.toggle('collapsed', !open);
    if (btn) btn.textContent = open ? '▾' : '▴';
  };
  if (btn) btn.addEventListener('click', function(){
    open = !open; sync();
    try { localStorage.setItem(KEY, open ? '1' : '0'); } catch(e){}
  });
  sync();
})();

/* ---- 手机竖屏：左侧导航抽屉 ---- */
(function(){
  var btn = document.getElementById('navToggle');
  var nav = document.querySelector('.sidenav');
  if (!btn || !nav) return;
  var sync = function(open){ nav.classList.toggle('open', !!open); };
  btn.addEventListener('click', function(e){
    e.stopPropagation();
    sync(!nav.classList.contains('open'));
  });
  nav.addEventListener('click', function(e){
    var t = e.target;
    if (t && t.tagName === 'A') sync(false);
  });
  document.addEventListener('click', function(e){
    if (nav.classList.contains('open') && !nav.contains(e.target) && !btn.contains(e.target)) sync(false);
  });
})();
