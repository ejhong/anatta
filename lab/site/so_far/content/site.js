/* The page's behaviour. Rooms: show the room named in the hash (else #aim) with a short entrance, and keep the
   old single-page anchors working. Glossary terms, chart marks and the field's marks open a small popover. The
   Self room's chariot comes apart and goes back together, and choosing a part opens its card; the k-NN toy
   recomputes as model B is turned and blurred. The Record's field regroups in ledger order. The ideas log
   filters, and the charts redraw when the theme changes. No storage, no network. */
(function () {
  'use strict';
  var ROOMS = ['aim', 'self', 'ideas', 'possibilities', 'experiments', 'instruments', 'record', 'next'];
  // Anchors from the single-page version and the room they now live in. Most still exist as ids
  // and are found directly; this map is the fallback if one is ever renamed.
  var MOVED = {
    standing: 'aim', frame: 'aim', calibration: 'instruments', ideas: 'ideas', experiments: 'experiments',
    specimens: 'record', learned: 'experiments', scorecard: 'record', caveats: 'record', plan: 'next',
    durability: 'experiments'
  };
  var doc = document, root = doc.documentElement;
  root.classList.add('js');
  function media(q) { try { return window.matchMedia(q); } catch (e) { return null; } }
  function onMedia(m, fn) {
    if (!m) return;
    if (m.addEventListener) m.addEventListener('change', fn); else if (m.addListener) m.addListener(fn);
  }
  var motionQ = media('(prefers-reduced-motion: reduce)');
  function still() { return !!(motionQ && motionQ.matches); }
  function one(sel, ctx) { return (ctx || doc).querySelector(sel); }
  function all(sel, ctx) { return Array.prototype.slice.call((ctx || doc).querySelectorAll(sel)); }

  // Custom properties written in style attributes (the chariot's offsets, the net's glints) are set again
  // through the CSSOM, so the drawing still works if a policy ever drops style attributes.
  all('[style*="--"]').forEach(function (el) {
    (el.getAttribute('style') || '').split(';').forEach(function (d) {
      var k = d.indexOf(':'), name = k > 0 ? d.slice(0, k).trim() : '';
      if (name.indexOf('--') === 0) el.style.setProperty(name, d.slice(k + 1).trim());
    });
  });

  // ------------------------------------------------------------------------------------------ the popover
  var pop = doc.getElementById('pop'), popFor = null, popTimer = 0, popQueued = false;
  function hidePop() {
    clearTimeout(popTimer);
    if (!pop || pop.hidden) { popFor = null; return; }
    pop.hidden = true;
    if (popFor && popFor.removeAttribute) popFor.removeAttribute('aria-describedby');
    popFor = null;
  }
  function placePop() {
    popQueued = false;
    if (!popFor || pop.hidden) return;
    var r = popFor.getBoundingClientRect(), vw = root.clientWidth, vh = window.innerHeight;
    if (r.bottom < 0 || r.top > vh) { hidePop(); return; }
    var pw = pop.offsetWidth, ph = pop.offsetHeight;
    var x = Math.max(16, Math.min(r.left + r.width / 2 - pw / 2, vw - pw - 16));
    var y = r.bottom + 10;
    if (y + ph > vh - 12 && r.top - ph - 10 > 64) y = r.top - ph - 10;
    pop.style.transform = 'translate(' + Math.round(x) + 'px, ' + Math.round(y) + 'px)';
  }
  function openPop(anchor, bits) {
    if (!pop) return;
    clearTimeout(popTimer);
    pop.textContent = '';
    bits.forEach(function (b) {
      var s = doc.createElement('span');
      s.className = b[0];
      s.textContent = b[1];
      pop.appendChild(s);
    });
    if (popFor && popFor !== anchor && popFor.removeAttribute) popFor.removeAttribute('aria-describedby');
    popFor = anchor;
    pop.hidden = false;
    if (anchor.setAttribute && anchor.tagName === 'A') anchor.setAttribute('aria-describedby', 'pop');
    placePop();
  }
  function later(fn, ms) { clearTimeout(popTimer); popTimer = setTimeout(fn, ms); }
  window.addEventListener('scroll', function () {
    if (popFor && !popQueued) { popQueued = true; requestAnimationFrame(placePop); }
  }, { passive: true });
  window.addEventListener('resize', hidePop);
  doc.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') hidePop(); });
  doc.addEventListener('pointerdown', function (ev) {
    if (popFor && !(ev.target.closest && ev.target.closest('a.term, .field .mk, [data-tip], .ch-pc, .ch-binder'))) hidePop();
  });

  // Glossary terms: hover or focus shows the definition; a click or tap shows it too, and a second tap on a
  // touch screen follows the link down to the glossary.
  var gloss = {};
  all('.glossary .g').forEach(function (g) {
    var dt = one('dt', g), dd = one('dd', g);
    if (dt && dd) gloss[g.id] = [dt.textContent, dd.textContent];
  });
  var fineQ = media('(hover: hover) and (pointer: fine)');
  function termId(a) { var id = (a.getAttribute('href') || '').slice(1); return gloss[id] ? id : null; }
  function showTerm(a) {
    var id = termId(a);
    if (id) openPop(a, [['pop-k', 'Term'], ['pop-x', gloss[id][0]], ['pop-d', gloss[id][1]]]);
  }
  all('a.term').forEach(function (a) {
    var wasOpen = false;
    a.addEventListener('mouseenter', function () { later(function () { showTerm(a); }, 110); });
    a.addEventListener('mouseleave', function () { later(hidePop, 180); });
    a.addEventListener('focus', function () { showTerm(a); });
    a.addEventListener('blur', function () { if (popFor === a) hidePop(); });
    a.addEventListener('pointerdown', function () { wasOpen = popFor === a && pop && !pop.hidden; });
    a.addEventListener('click', function (ev) {
      var second = wasOpen && !(fineQ && fineQ.matches);
      wasOpen = false;
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey || !termId(a) || second) return;
      ev.preventDefault();
      showTerm(a);
    });
  });

  // Chart marks drawn by figures.js carry their value in data-tip.
  doc.addEventListener('mouseover', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-tip]');
    if (t) openPop(t, [['pop-d', t.getAttribute('data-tip')]]);
  });
  doc.addEventListener('mouseout', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-tip]');
    if (t && popFor === t) later(hidePop, 120);
  });

  // ------------------------------------------------------------------------------------------ the ideas log filter
  var chips = all('.chip[data-group]'), logCards = all('.log-card'), shown = doc.getElementById('log-shown');
  function filter(group) {
    var n = 0;
    chips.forEach(function (c) { c.setAttribute('aria-pressed', String(c.getAttribute('data-group') === group)); });
    logCards.forEach(function (c) {
      var on = group === 'all' || c.getAttribute('data-group') === group;
      c.hidden = !on;
      if (on) n++;
    });
    if (shown) {
      shown.textContent = group === 'all' ? 'Showing all ' + logCards.length + ' entries.'
        : 'Showing ' + n + ' of ' + logCards.length + ' entries.';
    }
  }
  chips.forEach(function (c) { c.addEventListener('click', function () { filter(c.getAttribute('data-group')); }); });

  // ------------------------------------------------------------------------------------------ the chariot
  var chariot = (function () {
    var fig = doc.getElementById('chariot-fig');
    if (!fig) return null;
    var svg = one('.ch-svg', fig), btn = one('.ch-toggle', fig);
    var shapes = all('[data-part]', fig), items = all('.ch-item'), cards = all('.part-card');
    var ids = cards.map(function (c) { return c.id; });
    var info = {};
    cards.forEach(function (c) {
      var st = one('header .st', c);
      info[c.id] = { n: c.getAttribute('data-n') === '11' ? '?' : c.getAttribute('data-n'),
                     name: c.getAttribute('data-short'), st: st ? st.textContent : '' };
    });

    function setState(apart, quiet) {
      if (quiet) fig.classList.add('still');
      fig.setAttribute('data-state', apart ? 'apart' : 'whole');
      btn.textContent = apart ? 'Put it together' : 'Take it apart';
      if (quiet) requestAnimationFrame(function () { requestAnimationFrame(function () { fig.classList.remove('still'); }); });
    }
    function select(id, how) {
      if (ids.indexOf(id) < 0) return;
      cards.forEach(function (c) { c.hidden = c.id !== id; });
      items.forEach(function (a) {
        if (a.getAttribute('data-part') === id) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
      });
      shapes.forEach(function (s) { s.classList.toggle('on', s.getAttribute('data-part') === id); });
      if (how !== 'quiet') fig.classList.add('lit');
      if (how === 'show') {
        var card = doc.getElementById(id), r = card.getBoundingClientRect();
        if (r.top < 70 || r.top > window.innerHeight - 140) card.scrollIntoView({ block: 'start', behavior: still() ? 'auto' : 'smooth' });
      }
    }

    // One card at a time beside the list, with steps to the neighbouring parts.
    cards.forEach(function (c, i) {
      var bar = doc.createElement('div');
      bar.className = 'pc-nav';
      [[cards[i - 1], 'prev'], [cards[i + 1], 'next']].forEach(function (p) {
        if (!p[0]) { bar.appendChild(doc.createElement('span')); return; }
        var b = doc.createElement('button'), o = info[p[0].id];
        b.type = 'button';
        b.textContent = p[1] === 'prev' ? '← ' + o.n + ' ' + o.name : o.n + ' ' + o.name + ' →';
        b.setAttribute('data-dir', p[1]);
        b.addEventListener('click', function () {
          select(p[0].id, 'show');
          var same = one('.pc-nav button[data-dir="' + p[1] + '"]', p[0]) || one('.pc-nav button', p[0]);
          if (same) same.focus({ preventScroll: true });
        });
        bar.appendChild(b);
      });
      c.appendChild(bar);
    });

    items.forEach(function (a, i) {
      a.addEventListener('click', function (ev) {
        if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        ev.preventDefault();
        select(a.getAttribute('data-part'), 'show');
      });
      a.addEventListener('keydown', function (ev) {
        var j = ev.key === 'ArrowDown' ? i + 1 : ev.key === 'ArrowUp' ? i - 1 : -1;
        if (j < 0 || j >= items.length) return;
        ev.preventDefault();
        items[j].focus();
        select(items[j].getAttribute('data-part'), 'pick');
      });
    });
    shapes.forEach(function (s) {
      s.addEventListener('click', function () { select(s.getAttribute('data-part'), 'show'); });
      s.addEventListener('mouseenter', function () {
        var o = info[s.getAttribute('data-part')];
        if (o) openPop(s, [['pop-k', o.n === '?' ? 'The binder' : 'Part ' + o.n], ['pop-x', o.name], ['pop-s', o.st]]);
      });
      s.addEventListener('mouseleave', function () { later(hidePop, 100); });
    });
    btn.hidden = false;
    btn.addEventListener('click', function () { hidePop(); setState(fig.getAttribute('data-state') !== 'apart'); });

    // On a narrow screen the names would be too small to read: crop to the balloons and let the list name them.
    var narrowQ = media('(max-width: 640px)');
    function fit() {
      var n = !!(narrowQ && narrowQ.matches);
      fig.classList.toggle('narrow', n);
      svg.setAttribute('viewBox', n ? '150 20 800 620' : '0 0 1100 640');
    }
    onMedia(narrowQ, fit);
    fit();
    setState(still(), true);
    select(ids[0], 'quiet');
    return { select: select };
  })();

  // ------------------------------------------------------------------------------------------ the k-NN toy
  (function () {
    var A = doc.getElementById('knn-a'), B = doc.getElementById('knn-b');
    if (!A || !B) return;
    var rot = doc.getElementById('knn-rot'), noise = doc.getElementById('knn-noise');
    var out = doc.getElementById('knn-score'), chance = doc.getElementById('knn-chance');
    var pick = doc.getElementById('knn-pick');
    var N = 24, K = 3, C = 150, NS = 'http://www.w3.org/2000/svg', seed = 11;
    function rnd() {                                   // mulberry32, so every visit draws the same points
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function gauss() { return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }
    var centres = [[-60, -50], [62, -56], [-46, 60], [58, 54]], P = [], E = [];
    for (var i = 0; i < N; i++) {
      var c = centres[i % 4], x = c[0] + gauss() * 21, y = c[1] + gauss() * 21, d = Math.sqrt(x * x + y * y);
      if (d > 110) { x *= 110 / d; y *= 110 / d; }     // inside a circle, so turning keeps every point in view
      P.push([x, y]);
      E.push([gauss(), gauss()]);
    }
    function neighbours(Q) {
      return Q.map(function (p, i) {
        return Q.map(function (q, j) {
          return [j === i ? Infinity : (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]), j];
        }).sort(function (a, b) { return a[0] - b[0]; }).slice(0, K).map(function (a) { return a[1]; });
      });
    }
    function make(tag, attrs) {
      var e = doc.createElementNS(NS, tag);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }
    var nnA = neighbours(P), sel = 0, Q = P, nnB = nnA;
    function paint(svg, pts, mine, other) {
      svg.textContent = '';
      var p = pts[sel];
      mine.forEach(function (j) {
        svg.appendChild(make('line', { 'class': other.indexOf(j) >= 0 ? 'kl' : 'kl one',
          x1: (C + p[0]).toFixed(1), y1: (C + p[1]).toFixed(1), x2: (C + pts[j][0]).toFixed(1), y2: (C + pts[j][1]).toFixed(1) }));
      });
      pts.forEach(function (q, j) {
        var cls = 'kp';
        if (j === sel) cls += ' sel';
        else if (mine.indexOf(j) >= 0) cls += other.indexOf(j) >= 0 ? ' both' : ' one';
        svg.appendChild(make('circle', { 'class': cls, cx: (C + q[0]).toFixed(1), cy: (C + q[1]).toFixed(1), r: j === sel ? 6.5 : 5 }));
      });
      svg.appendChild(make('circle', { 'class': 'ksel', cx: (C + p[0]).toFixed(1), cy: (C + p[1]).toFixed(1), r: 12 }));
    }
    function draw() {
      var th = (+rot.value) * Math.PI / 180, s = (+noise.value) / 100 * 46, cs = Math.cos(th), sn = Math.sin(th);
      Q = P.map(function (p, j) {
        var x = p[0] * cs - p[1] * sn + E[j][0] * s, y = p[0] * sn + p[1] * cs + E[j][1] * s;
        return [Math.max(-138, Math.min(138, x)), Math.max(-138, Math.min(138, y))];
      });
      nnB = neighbours(Q);
      var tot = 0;
      for (var j = 0; j < N; j++) tot += nnA[j].filter(function (k) { return nnB[j].indexOf(k) >= 0; }).length / K;
      out.textContent = (tot / N).toFixed(2);
      paint(A, P, nnA[sel], nnB[sel]);
      paint(B, Q, nnB[sel], nnA[sel]);
      var both = nnA[sel].filter(function (k) { return nnB[sel].indexOf(k) >= 0; }).length;
      pick.textContent = 'Point ' + (sel + 1) + ' of ' + N + ': ' + both + ' of its ' + K +
        ' nearest neighbours are the same in both models.';
    }
    function choose(svg, ev) {
      var r = svg.getBoundingClientRect(), pts = svg === A ? P : Q;
      var x = (ev.clientX - r.left) / r.width * 300 - C, y = (ev.clientY - r.top) / r.height * 300 - C;
      var best = 0, bd = Infinity;
      pts.forEach(function (p, j) { var dd = (p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y); if (dd < bd) { bd = dd; best = j; } });
      sel = best;
      draw();
    }
    [A, B].forEach(function (svg, k) {
      svg.setAttribute('tabindex', '0');
      svg.setAttribute('aria-label', (k ? "Model B's points" : "Model A's points") + '. The arrow keys pick another point.');
      svg.addEventListener('click', function (ev) { choose(svg, ev); });
      svg.addEventListener('keydown', function (ev) {
        var step = ev.key === 'ArrowRight' || ev.key === 'ArrowDown' ? 1 : ev.key === 'ArrowLeft' || ev.key === 'ArrowUp' ? -1 : 0;
        if (!step) return;
        ev.preventDefault();
        sel = (sel + step + N) % N;
        draw();
      });
    });
    rot.addEventListener('input', draw);
    noise.addEventListener('input', draw);
    chance.textContent = 'Chance is about ' + (K / (N - 1)).toFixed(2) + '; a perfect match is 1.';
    draw();
  })();

  // ------------------------------------------------------------------------------------------ the field of marks
  (function () {
    var field = doc.getElementById('field'), order = one('.field-order');
    if (!field || !order) return;
    var byResult = one('.fd-by-result', field), byLedger = one('.fd-by-ledger', field);
    var table = doc.getElementById('score-table'), rows = table ? all('tbody tr', table) : [];
    function clip(s, n) {
      s = s.replace(/\s+/g, ' ').trim();
      return s.length > n ? s.slice(0, n - 1).replace(/\s+\S*$/, '') + '…' : s;
    }
    function light(r, on) { all('.mk[data-r="' + r + '"]', field).forEach(function (m) { m.classList.toggle('lit', on); }); }
    function show(m) {
      var r = rows[+m.getAttribute('data-r')];
      if (!r || r.cells.length < 4) return;
      openPop(m, [['pop-k', 'Prediction ' + clip(r.cells[0].textContent, 12)], ['pop-d', clip(r.cells[1].textContent, 230)],
        ['pop-s', 'Odds ' + clip(r.cells[2].textContent, 40) + ' · ' + clip(r.cells[3].textContent, 150)]]);
    }
    field.addEventListener('mouseover', function (ev) {
      var m = ev.target.closest('.mk');
      if (!m) return;
      light(m.getAttribute('data-r'), true);
      show(m);
    });
    field.addEventListener('mouseout', function (ev) {
      var m = ev.target.closest('.mk');
      if (!m) return;
      light(m.getAttribute('data-r'), false);
      later(hidePop, 120);
    });
    field.addEventListener('click', function (ev) { var m = ev.target.closest('.mk'); if (m) show(m); });
    order.hidden = false;
    order.addEventListener('click', function () {
      var toLedger = order.getAttribute('aria-pressed') !== 'true';
      var from = toLedger ? byResult : byLedger, to = toLedger ? byLedger : byResult, first = {};
      hidePop();
      if (!still()) all('.mk', from).forEach(function (m) { first[m.getAttribute('data-k')] = m.getBoundingClientRect(); });
      from.hidden = true;
      to.hidden = false;
      order.setAttribute('aria-pressed', String(toLedger));
      if (still()) return;
      var moving = all('.mk', to);
      moving.forEach(function (m) {                    // FLIP: start each mark where it was, then let it travel
        var a = first[m.getAttribute('data-k')], b = m.getBoundingClientRect();
        if (!a) return;
        m.style.transition = 'none';
        m.style.transform = 'translate(' + (a.left - b.left).toFixed(1) + 'px, ' + (a.top - b.top).toFixed(1) + 'px)';
      });
      void field.offsetWidth;
      moving.forEach(function (m, j) {
        m.style.transition = 'transform 760ms cubic-bezier(.2, .7, .2, 1) ' + Math.min(j * 3, 330) + 'ms';
        m.style.transform = '';
      });
      setTimeout(function () { moving.forEach(function (m) { m.style.transition = ''; }); }, 1200);
    });
  })();

  // ------------------------------------------------------------------------------------------ rooms
  var views = {};
  ROOMS.forEach(function (r) { views[r] = one('.view[data-view="' + r + '"]'); });
  var links = all('[data-room]'), nav = one('.rooms'), current = null;

  function target(name) {
    if (ROOMS.indexOf(name) >= 0) return { room: name, el: null };
    var el = name ? doc.getElementById(name) : null;
    if (el) {
      var view = el.closest('.view');            // a target outside the rooms (the glossary) keeps the room
      return { room: view ? view.getAttribute('data-view') : (current || 'aim'), el: el };
    }
    if (MOVED.hasOwnProperty(name)) return { room: MOVED[name], el: null };
    return { room: current || 'aim', el: null };
  }

  function show(initial) {
    var name = location.hash.slice(1);
    try { name = decodeURIComponent(name); } catch (e) { /* keep it raw */ }
    var t = target(name), moved = t.room !== current;
    hidePop();
    if (moved) {
      ROOMS.forEach(function (r) { if (views[r]) views[r].hidden = r !== t.room; });
      var v = views[t.room];
      if (v && current !== null && !still()) {
        v.classList.remove('entering');
        void v.offsetWidth;
        v.classList.add('entering');
      }
      current = t.room;
      var cur = null;
      links.forEach(function (a) {
        if (a.getAttribute('data-room') === t.room) { a.setAttribute('aria-current', 'page'); cur = a; }
        else a.removeAttribute('aria-current');
      });
      // On a phone the rooms scroll sideways; keep the current one in sight.
      if (nav && cur && cur.parentNode === nav) {
        var l = cur.offsetLeft - nav.offsetLeft;
        if (l < nav.scrollLeft || l + cur.offsetWidth > nav.scrollLeft + nav.clientWidth * 0.85) nav.scrollLeft = Math.max(0, l - 24);
      }
    }
    if (t.el) {
      var el = t.el, card = el.closest('.log-card'), part = el.closest('.part-card');
      if (card && card.hidden) filter('all');
      if (el.tagName === 'DETAILS') el.open = true;
      if (part && chariot) chariot.select(part.id, 'pick');
      requestAnimationFrame(function () {
        el.scrollIntoView({ block: 'start', behavior: moved || initial || still() ? 'auto' : 'smooth' });
      });
    } else if (moved && !initial) {
      window.scrollTo(0, 0);
    }
  }
  ROOMS.forEach(function (r) {
    if (views[r]) views[r].addEventListener('animationend', function () { views[r].classList.remove('entering'); });
  });

  // Arrow keys move along the rooms in the band.
  if (nav) {
    nav.addEventListener('keydown', function (ev) {
      var as = all('a', nav), i = as.indexOf(doc.activeElement), j = -1;
      if (i < 0) return;
      if (ev.key === 'ArrowRight') j = (i + 1) % as.length;
      else if (ev.key === 'ArrowLeft') j = (i - 1 + as.length) % as.length;
      else if (ev.key === 'Home') j = 0;
      else if (ev.key === 'End') j = as.length - 1;
      if (j < 0) return;
      ev.preventDefault();
      as[j].focus();
    });
  }

  window.addEventListener('hashchange', function () { show(false); });
  show(true);

  if (typeof drawFigures === 'function') {
    drawFigures();
    var redraw = function () { hidePop(); drawFigures(); };
    onMedia(media('(prefers-color-scheme: dark)'), redraw);
    try {
      new MutationObserver(redraw).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    } catch (e) { /* no observer support */ }
  }
})();
