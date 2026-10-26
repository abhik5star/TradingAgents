#!/usr/bin/env node
/* Render the market scene to standalone SVG, using the application's own camera.
 *
 * The design board needs a picture of the 3D view that is not an artist's impression.
 * So this script imports `tradingagents/research/assets/scene-math.js` — the same
 * matrices, the same track layout, the same palette the browser uses — builds the same
 * candle boxes, and projects them with the same view-projection. The only thing it adds
 * is a painter's-algorithm sort, because SVG has no depth buffer.
 *
 *   node docs/design/make_scene_svg.js > docs/design/scene.svg
 *
 * Bars come from the CSV fixtures when given, otherwise from a small deterministic
 * series, so the file is reproducible on a machine that has neither.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const SCENE_MATH = path.join(__dirname, '..', '..', 'tradingagents', 'research', 'assets',
                             'scene-math.js');
const M = require(SCENE_MATH);

const WIDTH = 1200;
const HEIGHT = 675;
const CANDLES = 34;
const BACKGROUND = [0.031, 0.047, 0.071];
const LIGHT = normalize([0.35, 0.9, 0.45]);

function normalize(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* A deterministic walk, used only when no fixture is supplied. */
function syntheticBars(count, seed, drift) {
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const bars = [];
  let price = 100;
  for (let index = 0; index < count; index += 1) {
    const open = price;
    const close = Math.max(1, open * (1 + (random() - 0.5) * 0.03 + drift));
    const high = Math.max(open, close) * (1 + random() * 0.012);
    const low = Math.min(open, close) * (1 - random() * 0.012);
    bars.push({ open, high, low, close });
    price = close;
  }
  return bars;
}

function readCsv(file, limit) {
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n').slice(1);
  return lines.slice(-limit).map((line) => {
    const [, open, high, low, close] = line.split(',');
    return { open: +open, high: +high, low: +low, close: +close };
  });
}

const FEEDS = [
  { symbol: 'ALPHA', color: '#6cc6f5', file: '/tmp/preview_csv/ALPHA-1d.csv', seed: 11, drift: 0.0012 },
  { symbol: 'BETA', color: '#f2b134', file: '/tmp/preview_csv/BETA-1d.csv', seed: 29, drift: -0.0008 },
  { symbol: 'GAMMA', color: '#67d18a', file: '/tmp/preview_csv/GAMMA-1d.csv', seed: 47, drift: 0.0004 },
].map((feed) => ({
  symbol: feed.symbol,
  color: feed.color,
  candles: fs.existsSync(feed.file)
    ? readCsv(feed.file, CANDLES)
    : syntheticBars(CANDLES, feed.seed, feed.drift),
}));

const OPTIONS = { maxCandles: CANDLES, spacingX: 0.11, spacingZ: 1.35, bodyWidth: 0.072,
                  depth: 0.5, height: 1.5 };
const geometry = M.buildTracks(FEEDS, OPTIONS);
const tracks = geometry.tracks;
const widest = tracks.reduce((max, track) => Math.max(max, track.width || 0), 0);
const radius = Math.max(4.2, Math.min(11.5, widest * 0.8 + 1.8));

const projection = M.perspective(42, WIDTH / HEIGHT, 0.1, 120);
const centre = [0, 0.62, 0];
const eye = M.orbitEye(centre, 0.55, 0.45, radius);
const view = M.lookAt(eye, centre, [0, 1, 0]);
const viewProjection = M.multiply(projection, view);

function toScreen(point) {
  const ndc = M.project(viewProjection, point);
  if (ndc[3] <= 0.0001) return null;
  return [(ndc[0] * 0.5 + 0.5) * WIDTH, (1 - (ndc[1] * 0.5 + 0.5)) * HEIGHT, ndc[2]];
}

function eyeDepth(point) {
  // Row 3 of the view matrix holds the eye-space Z, which is what a painter's sort needs.
  const m = view;
  return m[2] * point[0] + m[6] * point[1] + m[10] * point[2] + m[14];
}

function shade(base, normal, ndcZ) {
  const lambert = Math.max(0, dot(normal, LIGHT));
  const rim = 0.44 + 0.62 * lambert;
  const fog = Math.min(1, Math.max(0, -ndcZ / 42)) * 0.55;
  const lit = [base[0] * rim, base[1] * rim, base[2] * rim];
  const mixed = [
    lit[0] * (1 - fog) + BACKGROUND[0] * fog,
    lit[1] * (1 - fog) + BACKGROUND[1] * fog,
    lit[2] * (1 - fog) + BACKGROUND[2] * fog,
  ];
  return '#' + mixed.map((channel) => {
    const value = Math.round(Math.pow(Math.min(1, Math.max(0, channel)), 1 / 2.2) * 255);
    return value.toString(16).padStart(2, '0');
  }).join('');
}

const faces = [];

function emitQuad(corners, normal, tint) {
  const points = corners.map(toScreen);
  if (points.some((point) => point === null)) return;
  const depth = corners.reduce((sum, point) => sum + eyeDepth(point), 0) / corners.length;
  const ndcZ = points.reduce((sum, point) => sum + point[2], 0) / points.length;
  faces.push({
    depth,
    points: points.map((point) => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(' '),
    fill: shade(tint, normal, ndcZ),
  });
}

function withTone(hex, tone) {
  const rgb = M.hexToRgb(hex);
  return [Math.min(1, rgb[0] * tone), Math.min(1, rgb[1] * tone), Math.min(1, rgb[2] * tone)];
}

tracks.forEach((track, trackIndex) => {
  if (!track.count) return;
  const candles = FEEDS[trackIndex].candles;
  const half = OPTIONS.bodyWidth / 2;
  const halfDepth = OPTIONS.depth / 2;
  candles.forEach((candle, index) => {
    const x = track.xOffset + index * OPTIONS.spacingX;
    const z = track.z;
    const up = candle.close >= candle.open;
    const base = withTone(track.color, up ? 1.12 : 0.78);
    const yLow = Math.min(candle.open, candle.close) * 0 + (Math.min(candle.open, candle.close)
      - track.min) * track.scaleY;
    const yHigh = (Math.max(candle.open, candle.close) - track.min) * track.scaleY;
    const yMin = (candle.low - track.min) * track.scaleY;
    const yMax = (candle.high - track.min) * track.scaleY;
    const x0 = x - half;
    const x1 = x + half;
    const z0 = z - halfDepth;
    const z1 = z + halfDepth;

    // Body: the three faces a camera at (+x, +y, +z) can see.
    emitQuad([[x0, yLow, z1], [x1, yLow, z1], [x1, yHigh, z1], [x0, yHigh, z1]],
             [0, 0, 1], base);
    emitQuad([[x1, yLow, z0], [x1, yLow, z1], [x1, yHigh, z1], [x1, yHigh, z0]],
             [1, 0, 0], base);
    emitQuad([[x0, yHigh, z0], [x1, yHigh, z0], [x1, yHigh, z1], [x0, yHigh, z1]],
             [0, 1, 0], base);

    // Wick: a thin column through the whole range.
    const wick = OPTIONS.bodyWidth / 6;
    const dark = withTone(track.color, up ? 0.9 : 0.62);
    emitQuad([[x - wick, yMin, z1], [x + wick, yMin, z1], [x + wick, yMax, z1],
              [x - wick, yMax, z1]], [0, 0, 1], dark);
    emitQuad([[x + wick, yMin, z0], [x + wick, yMin, z1], [x + wick, yMax, z1],
              [x + wick, yMax, z0]], [1, 0, 0], dark);
  });
});

faces.sort((a, b) => b.depth - a.depth);

const grid = M.buildGrid({ lines: 16, step: 0.85 });
const gridLines = [];
for (let i = 0; i < grid.positions.length; i += 6) {
  const a = toScreen([grid.positions[i], grid.positions[i + 1], grid.positions[i + 2]]);
  const b = toScreen([grid.positions[i + 3], grid.positions[i + 4], grid.positions[i + 5]]);
  if (!a || !b) continue;
  const depth = Math.min(1, Math.max(0, -((a[2] + b[2]) / 2) / 42));
  const opacity = Math.max(0.06, 0.5 * (1 - depth * 0.85));
  gridLines.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" ` +
                 `x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" ` +
                 `stroke="#3f5f7d" stroke-opacity="${opacity.toFixed(2)}" stroke-width="1"/>`);
}

const background = M.hexToRgb('#080c12');
const svg = [
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${WIDTH} ${HEIGHT}" ` +
  `width="${WIDTH}" height="${HEIGHT}" role="img" ` +
  'aria-label="Three tracks of candles in perspective over a wireframe floor">',
  '<defs>',
  `<radialGradient id="glow" cx="20%" cy="0%" r="120%">`,
  '<stop offset="0" stop-color="#16273a"/>',
  '<stop offset="0.45" stop-color="#0a1018"/>',
  '<stop offset="1" stop-color="#05080c"/>',
  '</radialGradient>',
  '</defs>',
  `<rect width="${WIDTH}" height="${HEIGHT}" fill="url(#glow)"/>`,
  `<rect width="${WIDTH}" height="${HEIGHT}" fill="rgb(${background.map((c) => Math.round(c * 255)).join(',')})" opacity="0.35"/>`,
  '<g>', gridLines.join(''), '</g>',
  '<g>', faces.map((face) => `<polygon points="${face.points}" fill="${face.fill}"/>`).join(''),
  '</g>',
  '</svg>',
].join('\n');

process.stdout.write(svg + '\n');
process.stderr.write(`scene: ${tracks.map((t) => `${t.symbol}:${t.count}`).join(' ')} · ` +
                     `${faces.length} faces · radius ${radius.toFixed(2)}\n`);
