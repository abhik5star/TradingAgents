/* Page controller for the 3D market view.
 *
 * Reads the server-rendered boot payload, paints the flat table and the legend, then
 * polls /api/feeds and pushes new candles into the scene. Everything it shows already
 * exists in the DOM without it: with JavaScript off the page is a readable table, and
 * with no WebGL the stage shows an explanation over the chart rather than a blank box.
 */
'use strict';

(function () {
  const bootElement = document.getElementById('feeds-boot');
  if (!bootElement) return;

  let boot;
  try {
    boot = JSON.parse(bootElement.textContent || '{}');
  } catch (error) {
    boot = {};
  }

  const pollUrl = boot.poll_url || '/api/feeds';
  const pollSeconds = Math.max(1, Number(boot.poll_seconds) || 2.5);
  const canvas = document.getElementById('stage-canvas');
  const stage = document.getElementById('stage');
  const legend = document.getElementById('legend');
  const statusPill = document.getElementById('stage-status');
  const fpsBox = document.getElementById('stage-fps');
  const timeBox = document.getElementById('stage-clock');
  const fallback = document.getElementById('stage-fallback');
  const tableBody = document.getElementById('tape-body');

  const view = {
    payload: boot.initial || null,
    signature: '',
    failures: 0,
    scene: null,
    lastFrame: 0,
    frames: 0,
  };

  function text(value) {
    if (value === null || value === undefined || value === '') return '—';
    return String(value);
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character];
    });
  }

  function money(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    const number = Number(value);
    const decimals = Math.abs(number) >= 1000 ? 2 : (Math.abs(number) >= 1 ? 4 : 8);
    return number.toLocaleString(undefined, {
      minimumFractionDigits: 2, maximumFractionDigits: decimals,
    });
  }

  function signed(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    const number = Number(value);
    return (number >= 0 ? '+' : '') + number.toFixed(2) + '%';
  }

  function tone(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 'flat';
    if (number > 0.0001) return 'up';
    if (number < -0.0001) return 'down';
    return 'flat';
  }

  function setStatus(label, kind) {
    if (!statusPill) return;
    statusPill.textContent = label;
    statusPill.className = 'pill ' + (kind || 'idle');
  }

  function feedColor(feed, index) {
    if (feed.color) return feed.color;
    const palette = ['#6cc6f5', '#f2b134', '#67d18a', '#e86a92', '#b28df0', '#4fd1c5'];
    return palette[index % palette.length];
  }

  function colourFor(feeds, symbol) {
    for (let i = 0; i < feeds.length; i += 1) {
      if (feeds[i].symbol === symbol) return feedColor(feeds[i], i);
    }
    return '#6cc6f5';
  }

  function renderLegend(feeds) {
    if (!legend) return;
    legend.innerHTML = '';
    feeds.forEach(function (feed, index) {
      const series = feed.series || {};
      const change = series.change_pct;
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'legend-item ' + tone(change);
      item.title = (feed.spec && feed.spec.label) || feed.symbol;
      item.innerHTML =
        '<span class="dot" style="background:' + feedColor(feed, index) + '"></span>' +
        '<span class="sym">' + esc(text(feed.symbol)) + '</span>' +
        '<span class="px">' + money(series.price) + '</span>' +
        '<span class="chg">' + signed(change) + '</span>' +
        '<span class="tag ' + esc(text(feed.quality)) + '">' + esc(text(feed.quality)) +
        '</span>';
      legend.appendChild(item);
    });
  }

  function renderTable(feeds) {
    if (!tableBody) return;
    tableBody.innerHTML = '';
    feeds.forEach(function (feed) {
      const series = feed.series || {};
      const spec = feed.spec || {};
      const row = document.createElement('tr');
      const cells = [
        ['', esc(text(feed.symbol))],
        ['', esc(text(feed.provider))],
        ['', '<span class="tag ' + esc(text(feed.quality)) + '">' + esc(text(feed.quality)) +
              '</span>'],
        ['', esc(text(feed.status))],
        ['num', money(series.price)],
        ['num ' + tone(series.change_pct), signed(series.change_pct)],
        ['num', esc(text(series.candles))],
        ['num', esc(text(series.trades))],
        ['note', esc(text(feed.note || feed.error || spec.url))],
      ];
      cells.forEach(function (cell) {
        const td = document.createElement('td');
        if (cell[0]) td.className = cell[0];
        td.innerHTML = cell[1];
        row.appendChild(td);
      });
      tableBody.appendChild(row);
    });
  }

  function signatureOf(feeds) {
    return feeds.map(function (feed) {
      const series = feed.series || {};
      const candles = feed.candles || [];
      const last = candles.length ? candles[candles.length - 1] : null;
      return [feed.symbol, feed.quality, candles.length,
              last ? last.close : '', last ? last.high : '', last ? last.low : ''].join(':');
    }).join('|');
  }

  function labelLayer() {
    if (!stage) return null;
    let layer = document.getElementById('stage-labels');
    if (!layer) {
      layer = document.createElement('div');
      layer.className = 'stage-labels';
      layer.id = 'stage-labels';
      stage.appendChild(layer);
    }
    return layer;
  }

  function ensureLabels(feeds) {
    const layer = labelLayer();
    if (!layer) return;
    const wanted = {};
    feeds.forEach(function (feed) {
      wanted[feed.symbol] = true;
      const id = 'label-' + String(feed.symbol).replace(/[^A-Za-z0-9_-]/g, '_');
      let node = document.getElementById(id);
      if (!node) {
        node = document.createElement('span');
        node.id = id;
        node.className = 'price-tag';
        layer.appendChild(node);
      }
      const series = feed.series || {};
      node.innerHTML = '<b>' + esc(text(feed.symbol)) + '</b> ' + money(series.price);
      node.className = 'price-tag ' + tone(series.change_pct) +
        (feed.quality === 'simulated' ? ' sim' : '');
      node.dataset.symbol = feed.symbol;
    });
    Array.prototype.slice.call(layer.children).forEach(function (node) {
      if (!wanted[node.dataset.symbol]) layer.removeChild(node);
    });
  }

  /* Price tags ride the newest candle: the scene is 3D, but the number stays readable. */
  function positionLabels() {
    if (!view.scene || !view.scene.projectPoint) return;
    const layer = document.getElementById('stage-labels');
    if (!layer) return;
    const bySymbol = {};
    Array.prototype.slice.call(layer.children).forEach(function (node) {
      bySymbol[node.dataset.symbol] = node;
      node.style.display = 'none';
    });
    (view.scene.tracks || []).forEach(function (track) {
      const node = bySymbol[track.symbol];
      if (!node) return;
      if (!track.count) {
        node.style.display = 'none';
        return;
      }
      const point = view.scene.projectPoint(
        [track.lastX, (track.last - track.min) * track.scaleY, track.z]);
      if (!point || !point.visible) {
        node.style.display = 'none';
        return;
      }
      node.style.display = 'block';
      node.style.transform = 'translate(-50%, -130%) translate(' +
        point.x.toFixed(1) + 'px,' + point.y.toFixed(1) + 'px)';
    });
  }

  function paint(payload) {
    if (!payload || !payload.feeds) return;
    view.payload = payload;
    const feeds = payload.feeds.map(function (feed, index) {
      return Object.assign({}, feed, { color: feedColor(feed, index) });
    });
    renderLegend(feeds);
    renderTable(feeds);
    ensureLabels(feeds);

    const connected = feeds.filter(function (feed) { return feed.connected; }).length;
    const simulated = feeds.some(function (feed) { return feed.quality === 'simulated'; });
    if (simulated) {
      setStatus('SIMULATED · ' + connected + '/' + feeds.length + ' live', 'sim');
    } else if (connected === feeds.length) {
      setStatus('LIVE · ' + connected + '/' + feeds.length, 'live');
    } else if (connected === 0) {
      setStatus('NO FEED · ' + feeds.length + ' configured', 'down');
    } else {
      setStatus('PARTIAL · ' + connected + '/' + feeds.length, 'warn');
    }
    if (timeBox) {
      timeBox.textContent = payload.generated_at
        ? new Date(payload.generated_at).toLocaleTimeString() : '—';
    }

    const signature = signatureOf(feeds);
    if (view.scene && signature !== view.signature) {
      view.signature = signature;
      view.scene.setData(feeds);
    }
  }

  function fetchState() {
    const controller = new AbortController();
    const timer = setTimeout(function () { controller.abort(); }, 6000);
    fetch(pollUrl, { cache: 'no-store', signal: controller.signal,
                     headers: { Accept: 'application/json' } })
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (payload) {
        view.failures = 0;
        paint(payload);
      })
      .catch(function () {
        view.failures += 1;
        // The numbers on screen came from the server at render time. When polling stops
        // working, say that they are stale rather than pretending they are current.
        if (view.failures >= 3) {
          setStatus('OFFLINE · render-time data', 'down');
        } else {
          setStatus('STALE · render-time data', 'warn');
        }
      })
      .then(function () { clearTimeout(timer); });
  }

  function frame(now) {
    if (view.scene) {
      view.scene.resize();
      view.scene.render(now);
      positionLabels();
      view.frames += 1;
      if (now - view.lastFrame > 1000) {
        if (fpsBox) fpsBox.textContent = view.frames + ' fps';
        view.frames = 0;
        view.lastFrame = now;
      }
    }
    requestAnimationFrame(frame);
  }

  function toggleButton(id, handler) {
    const button = document.getElementById(id);
    if (button) button.addEventListener('click', handler);
    return button;
  }

  function initScene() {
    if (!canvas || !window.startMarketScene) {
      if (fallback) fallback.hidden = false;
      return;
    }
    try {
      view.scene = window.startMarketScene(canvas);
    } catch (error) {
      view.scene = null;
    }
    if (!view.scene) {
      if (fallback) {
        fallback.hidden = false;
        fallback.textContent = 'WebGL is not available in this browser, so the 3D stage ' +
          'cannot draw. The tape table below carries the same numbers.';
      }
      if (statusPill) statusPill.classList.add('flat-only');
      return;
    }
    view.scene.onFrame = null;
    view.scene.onToggleRotate = function (on) {
      const button = document.getElementById('btn-rotate');
      if (button) button.setAttribute('aria-pressed', on ? 'true' : 'false');
    };
    const first = view.payload && view.payload.feeds;
    if (first) {
      const feeds = first.map(function (feed, index) {
        return Object.assign({}, feed, { color: feedColor(feed, index) });
      });
      view.scene.setData(feeds);
      view.signature = signatureOf(feeds);
      ensureLabels(feeds);
    }
    requestAnimationFrame(frame);
  }

  toggleButton('btn-rotate', function () {
    if (!view.scene) return;
    view.scene.state.autoRotate = !view.scene.state.autoRotate;
    this.setAttribute('aria-pressed', view.scene.state.autoRotate ? 'true' : 'false');
  });

  toggleButton('btn-reset', function () {
    if (!view.scene) return;
    view.scene.state.azimuth = 0.55;
    view.scene.state.elevation = 0.45;
    view.scene.frameCameraToTracks();
    view.scene.state.autoRotate = true;
  });

  toggleButton('btn-flat', function () {
    if (!stage) return;
    const flat = stage.classList.toggle('flat');
    this.setAttribute('aria-pressed', flat ? 'true' : 'false');
    this.textContent = flat ? '3D view' : 'Flat table';
  });

  document.addEventListener('keydown', function (event) {
    if (event.target && /input|select|textarea/i.test(event.target.tagName)) return;
    if (event.key === 'r' && view.scene) {
      view.scene.state.autoRotate = !view.scene.state.autoRotate;
    }
    if (event.key === 'f') {
      const button = document.getElementById('btn-flat');
      if (button) button.click();
    }
  });

  if (view.payload) paint(view.payload);
  initScene();
  fetchState();
  setInterval(fetchState, pollSeconds * 1000);
  window.addEventListener('resize', function () {
    if (view.scene) view.scene.resize();
  });
})();
