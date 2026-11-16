(function () {
  // The Experiments room's charts, drawn from the data block. Colours come from the page's tokens, so a theme
  // change redraws them; text keeps to the ink tokens and the marks carry the colour. Each mark carries its
  // value in data-tip for the popover.
  var D = JSON.parse(document.getElementById('data').textContent);
  var css = getComputedStyle(document.documentElement);
  function tok(n) { return css.getPropertyValue(n).trim(); }
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function svg(fig, w, h) {
    var s = el('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img' });
    s.setAttribute('aria-labelledby', fig.id + '-cap');
    return s;
  }
  var MONO = 'DM Mono, ui-monospace, Menlo, monospace', SERIF = 'Spectral, Georgia, serif';
  function label(s, x, y, t, o) {
    o = o || {};
    var e = el('text', { x: x, y: y, 'font-family': o.serif ? SERIF : MONO, 'font-size': o.size || 12,
      fill: o.color || tok('--ink-2'), 'text-anchor': o.anchor || 'start' }, t);
    s.appendChild(e); return e;
  }
  function tip(e, text) { e.setAttribute('data-tip', text); return e; }
  function caption(fig, text) {
    var c = document.createElement('figcaption');
    c.id = fig.id + '-cap';
    c.textContent = text;
    fig.appendChild(c);
  }
  function grid(s, ticks, max, left, span, top, bottom, H) {
    ticks.forEach(function (t) {
      var x = left + span * t / max;
      s.appendChild(el('line', { x1: x, y1: top - 6, x2: x, y2: H - bottom, stroke: tok('--rule'), 'stroke-width': 1 }));
      label(s, x, H - 10, t + '%', { anchor: 'middle', color: tok('--ink-3'), size: 12.5 });
    });
  }

  function bars(fig, data, opts) {
    var W = 760, rowH = 46, top = 26, left = 250, right = 84, H = top + data.rows.length * rowH + 30;
    var s = svg(fig, W, H), max = opts.max, span = W - left - right;
    grid(s, opts.ticks, max, left, span, top, 26, H);
    data.rows.forEach(function (r, i) {
      var y = top + i * rowH;
      var color = r.kind === 'cut' ? tok('--gold-line') : (r.kind === 'control' ? tok('--ink-3') : tok('--ink-2'));
      label(s, left - 14, y + 17, r.name, { anchor: 'end', serif: true, size: 16, color: tok('--ink') });
      var w = span * r.lie / max;
      s.appendChild(tip(el('rect', { x: left, y: y + 4, width: w, height: 18, fill: color, rx: 2 }), r.name + ': ' + r.lie.toFixed(1) + '%'));
      label(s, left + w + 8, y + 18, r.lie.toFixed(1) + '%', { size: 13.5, color: tok('--ink') });
      if (r.sedation != null) {
        var w2 = span * r.sedation / max;
        s.appendChild(tip(el('rect', { x: left, y: y + 26, width: w2, height: 5, fill: tok('--cinnabar-line'), rx: 1 }), r.name + ': timid ' + r.sedation + '%'));
        label(s, left + w2 + 8, y + 32, 'timid ' + r.sedation + '%', { size: 12, color: tok('--ink-2') });
      }
    });
    fig.appendChild(s);
  }
  var cuts = document.getElementById('fig-cuts');
  bars(cuts, D.cuts, { max: 80, ticks: [0, 20, 40, 60, 80] });
  caption(cuts, D.cuts.label + '. Thick bar: share of reports that under-report the catch. Thin red bar: share of games fishing far below quota in the first season (timidity).');

  var fixed = document.getElementById('fig-fixed');
  bars(fixed, { rows: D.fixed.rows.map(function (r, i) { return { name: r.name, lie: r.lie, kind: i ? 'cut' : 'base' }; }) }, { max: 60, ticks: [0, 20, 40, 60] });
  caption(fixed, D.fixed.label + '. The boat always lands exactly the quota, so only the report is chosen.');

  (function () {
    var fig = document.getElementById('fig-framings'), data = D.framings;
    var W = 760, rowH = 58, top = 26, left = 250, right = 84, H = top + data.rows.length * rowH + 30, max = 50, span = W - left - right;
    var s = svg(fig, W, H);
    grid(s, [0, 10, 20, 30, 40, 50], max, left, span, top, 26, H);
    data.rows.forEach(function (r, i) {
      var y = top + i * rowH;
      label(s, left - 14, y + 20, r.name, { anchor: 'end', serif: true, size: 16, color: tok('--ink') });
      var wb = span * r.base / max, wc = span * r.cut / max;
      s.appendChild(tip(el('rect', { x: left, y: y + 4, width: wb, height: 14, fill: tok('--ink-2'), rx: 2 }), r.name + ', unchanged: ' + r.base + '%'));
      label(s, left + wb + 8, y + 15, 'unchanged ' + r.base + '%', { size: 12.5, color: tok('--ink-2') });
      s.appendChild(tip(el('rect', { x: left, y: y + 20, width: wc, height: 14, fill: tok('--gold-line'), rx: 2 }), r.name + ', cut: ' + r.cut + '%'));
      label(s, left + wc + 8, y + 31, 'cut ' + r.cut + '%', { size: 12.5, color: tok('--ink') });
    });
    fig.appendChild(s);
    caption(fig, data.label + '. The cuts take about 20 points off when the model has no role of its own, and about 14 when it plays itself.');
  })();

  (function () {
    var fig = document.getElementById('fig-audit'), data = D.audit;
    var W = 760, H = 330, left = 64, right = 150, top = 24, bottom = 50, pw = W - left - right, ph = H - top - bottom;
    var s = svg(fig, W, H), ymax = 70;
    [0, 20, 40, 60].forEach(function (t) {
      var y = top + ph - ph * t / ymax;
      s.appendChild(el('line', { x1: left, y1: y, x2: left + pw, y2: y, stroke: tok('--rule') }));
      label(s, left - 10, y + 4, t + '%', { anchor: 'end', color: tok('--ink-3'), size: 12.5 });
    });
    function lv(v) { return v === 0 ? 'no audits' : (v === 1 ? 'every report' : (v * 10) + ' in 10'); }
    data.levels.forEach(function (v) { label(s, left + pw * v, H - 24, lv(v), { anchor: 'middle', color: tok('--ink-3'), size: 12 }); });
    label(s, left + pw / 2, H - 6, 'stated chance that a report is audited', { anchor: 'middle', color: tok('--ink-3'), size: 12 });
    data.series.forEach(function (sr, k) {
      var color = k === 0 ? tok('--ink-2') : tok('--gold-line');
      var pts = sr.y.map(function (v, i) { return [left + pw * data.levels[i], top + ph - ph * v / ymax]; });
      s.appendChild(el('path', { d: 'M' + pts.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L'),
        fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      pts.forEach(function (p, i) {
        s.appendChild(tip(el('circle', { cx: p[0], cy: p[1], r: 5, fill: color, stroke: tok('--ground'), 'stroke-width': 2 }),
          sr.name + ', ' + lv(data.levels[i]) + ': ' + sr.y[i].toFixed(1) + '%'));
        if (i === 0 || i === pts.length - 1) label(s, p[0] + (i === 0 ? 10 : -10), p[1] + (k === 0 ? -9 : 17), sr.y[i].toFixed(0) + '%', { anchor: i === 0 ? 'start' : 'end', size: 13, color: tok('--ink') });
      });
      label(s, left + pw + 12, pts[pts.length - 1][1] + 4, sr.name, { size: 13, color: tok('--ink') });
    });
    fig.appendChild(s);
    caption(fig, data.label + '. Slopes: unchanged −37 points per unit of probability, the cut −32. A parallel shift of about 7 points.');
  })();

  (function () {
    // Stacked in the order feeds, deflates, points back, other, so that the red and the gold never touch.
    var fig = document.getElementById('fig-spiritual'), data = D.spiritual, order = [0, 2, 1, 3];
    var W = 760, rowH = 44, top = 40, left = 170, right = 30, H = top + data.rows.length * rowH + 20, span = W - left - right;
    var s = svg(fig, W, H);
    var fills = [tok('--cinnabar'), tok('--gold'), tok('--blue-text'), tok('--rule-2')];
    var inks = [tok('--ground'), tok('--ground'), tok('--ground'), tok('--ink-2')];
    order.forEach(function (j, i) {
      var x = left + i * 150;
      s.appendChild(el('rect', { x: x, y: 8, width: 11, height: 11, fill: fills[j], rx: 2 }));
      label(s, x + 17, 18, data.cats[j], { size: 12.5, color: tok('--ink-2') });
    });
    data.rows.forEach(function (r, i) {
      var y = top + i * rowH, x = left;
      label(s, left - 12, y + 18, r.name, { anchor: 'end', serif: true, size: 16, color: tok('--ink') });
      order.forEach(function (j) {
        var v = r.v[j], w = span * v / 100;
        if (w > 0) s.appendChild(tip(el('rect', { x: x, y: y + 4, width: Math.max(w - 2, 0), height: 20, fill: fills[j], rx: 2 }), r.name + ', ' + data.cats[j].toLowerCase() + ': ' + v + '%'));
        if (v >= 12) label(s, x + 6, y + 19, v.toFixed(0) + '%', { size: 12.5, color: inks[j] });
        x += w;
      });
    });
    fig.appendChild(s);
    caption(fig, data.label + '. The frontier model (Sonnet 4.6) also served as the judge, so its own row may be flattered.');
  })();

  (function () {
    var fig = document.getElementById('fig-durability'), data = D.durability;
    var W = 760, rowH = 62, top = 40, left = 230, right = 30, H = top + data.rows.length * rowH + 30, max = 80, span = W - left - right;
    var s = svg(fig, W, H);
    var series = [{ key: 'reached', keep: 'kr', name: 'Reached', color: tok('--gold-line') }, { key: 'installed', keep: 'ki', name: 'Installed', color: tok('--blue') }];
    series.forEach(function (sr, k) {
      var x = left + k * 130;
      s.appendChild(el('rect', { x: x, y: 8, width: 11, height: 11, fill: sr.color, rx: 2 }));
      label(s, x + 17, 18, sr.name, { size: 12.5, color: tok('--ink-2') });
    });
    grid(s, [0, 20, 40, 60, 80], max, left, span, top, 26, H);
    data.rows.forEach(function (r, i) {
      var y = top + i * rowH;
      label(s, left - 14, y + 20, r.name, { anchor: 'end', serif: true, size: 16, color: tok('--ink') });
      if (r.decides) label(s, left - 14, y + 38, 'decides', { anchor: 'end', size: 12, color: tok('--ink-3') });
      series.forEach(function (sr, k) {
        var v = r[sr.key], w = span * v / max, yy = y + 4 + k * 18;
        var keep = r[sr.keep] != null ? ', ' + r[sr.keep] + '% of its start kept' : '';
        s.appendChild(tip(el('rect', { x: left, y: yy, width: w, height: 14, fill: sr.color, rx: 2 }),
          sr.name + ', ' + r.name.toLowerCase() + ': ' + v.toFixed(1) + '% release' + keep));
        label(s, left + w + 8, yy + 11, sr.name.toLowerCase() + ' ' + v.toFixed(1) + '%' + (r[sr.keep] != null ? ' · kept ' + r[sr.keep] + '%' : ''), { size: 12.5, color: tok('--ink-2') });
      });
    });
    fig.appendChild(s);
    caption(fig, data.label + '. Reached: Stage A’s release arm, which found release through its own failures. Installed: a matched arm trained on releases sampled under a note saying the task is impossible. “Kept” is the share of the start that remains. The prediction, read after reinforcement learning, needed the reached arm to keep 20 points more of its start and to lose 10 points less.');
  })();
})();
