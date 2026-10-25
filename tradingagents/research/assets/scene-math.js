/* Scene mathematics for the 3D market view.
 *
 * Deliberately dependency-free and DOM-free: this file is loaded by the browser and
 * also run under Node by the test suite, so the matrices and the geometry are checked
 * rather than assumed. Nothing here touches WebGL, the document or the network.
 *
 * Conventions: column-major 4x4 matrices (the order WebGL's uniformMatrix4fv wants),
 * right-handed coordinates, +Y up, camera looking down -Z before the view transform.
 */
'use strict';

const DEG = Math.PI / 180;

function identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[column * 4 + k];
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

function perspective(fovDegrees, aspect, near, far) {
  const f = 1 / Math.tan((fovDegrees * DEG) / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * range, -1,
    0, 0, 2 * near * far * range, 0,
  ]);
}

function normalize3(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lookAt(eye, target, up) {
  const forward = normalize3([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const right = normalize3(cross3(up, forward));
  const trueUp = cross3(forward, right);
  return new Float32Array([
    right[0], trueUp[0], forward[0], 0,
    right[1], trueUp[1], forward[1], 0,
    right[2], trueUp[2], forward[2], 0,
    -dot3(right, eye), -dot3(trueUp, eye), -dot3(forward, eye), 1,
  ]);
}

/* Project a world point to normalised device coordinates, for hit-testing and for the
 * tests that assert the scene actually lands inside the viewport. */
function project(viewProjection, point) {
  const m = viewProjection;
  const x = m[0] * point[0] + m[4] * point[1] + m[8] * point[2] + m[12];
  const y = m[1] * point[0] + m[5] * point[1] + m[9] * point[2] + m[13];
  const z = m[2] * point[0] + m[6] * point[1] + m[10] * point[2] + m[14];
  const w = m[3] * point[0] + m[7] * point[1] + m[11] * point[2] + m[15];
  const safe = Math.abs(w) < 1e-9 ? 1e-9 : w;
  return [x / safe, y / safe, z / safe, w];
}

/* Camera orbiting a target: azimuth/elevation in radians, radius in world units. */
function orbitEye(target, azimuth, elevation, radius) {
  const cosEl = Math.cos(elevation);
  return [
    target[0] + radius * cosEl * Math.sin(azimuth),
    target[1] + radius * Math.sin(elevation),
    target[2] + radius * cosEl * Math.cos(azimuth),
  ];
}

function hexToRgb(hex) {
  const text = String(hex || '#5aa9e6').replace('#', '');
  const value = parseInt(text.length === 3 ? text.split('').map((c) => c + c).join('') : text, 16);
  if (Number.isNaN(value)) return [0.35, 0.66, 0.9];
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

/* ------------------------------------------------------------------------------------
 * Scene geometry
 *
 * One "track" per symbol is laid along Z, candles along X, price along Y. Each candle is
 * a box (12 triangles) with face normals, so a directional light gives it depth; a wick
 * column runs through the body so the high/low range is visible in three dimensions.
 * ---------------------------------------------------------------------------------- */

// body + wick + floor tick, twelve triangles each
const TRIANGLES_PER_CANDLE = 36;

function buildTracks(feeds, options) {
  const opts = options || {};
  const maxCandles = opts.maxCandles || 90;
  const spacingX = opts.spacingX || 0.11;
  const spacingZ = opts.spacingZ || 1.35;
  const bodyWidth = opts.bodyWidth || 0.072;
  const depth = opts.depth === undefined ? 0.5 : opts.depth;
  const height = opts.height === undefined ? 1.5 : opts.height;

  const positions = [];
  const normals = [];
  const colors = [];
  const tracks = [];
  const list = feeds || [];
  // Tracks are centred on the origin in both directions so the default camera can look
  // at (0, y, 0) without knowing how many symbols or candles there are.
  const zOffset = -((list.length - 1) * spacingZ) / 2;

  list.forEach((feed, trackIndex) => {
    const candles = (feed.candles || []).slice(-maxCandles);
    if (!candles.length) {
      tracks.push({ symbol: feed.symbol, z: trackIndex * spacingZ + zOffset, count: 0,
                    width: 0, min: 0, max: 0, color: feed.color || paletteFor(trackIndex) });
      return;
    }
    let low = Infinity;
    let high = -Infinity;
    candles.forEach((candle) => {
      low = Math.min(low, candle.low);
      high = Math.max(high, candle.high);
    });
    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) {
      high = low + 1;
    }
    const span = high - low;
    const scaleY = height / span;
    const z = trackIndex * spacingZ + zOffset;
    const xOffset = -((candles.length - 1) * spacingX) / 2;
    const base = hexToRgb(feed.color || paletteFor(trackIndex));
    const first = candles[0].open;

    candles.forEach((candle, index) => {
      const up = candle.close >= candle.open;
      const tone = up ? 1.12 : 0.78;
      const color = [Math.min(1, base[0] * tone), Math.min(1, base[1] * tone),
                     Math.min(1, base[2] * tone)];
      const x = xOffset + index * spacingX;
      const yOpen = (candle.open - low) * scaleY;
      const yClose = (candle.close - low) * scaleY;
      const yLow = (candle.low - low) * scaleY;
      const yHigh = (candle.high - low) * scaleY;
      const yMin = Math.min(yOpen, yClose);
      const yMax = Math.max(yOpen, yClose);
      // Body: a box from yMin to yMax, centred on x.
      box(positions, normals, colors, x - bodyWidth / 2, yMin, z - depth / 2,
          x + bodyWidth / 2, yMax, z + depth / 2, color);
      // Wick: a thin column from low to high.
      box(positions, normals, colors, x - bodyWidth / 6, yLow, z - depth / 6,
          x + bodyWidth / 6, yHigh, z + depth / 6, [
        color[0] * 0.8, color[1] * 0.8, color[2] * 0.8]);
      // A faint floor tick under each candle anchors it to the grid.
      box(positions, normals, colors, x - bodyWidth / 2, 0, z - depth / 2,
          x + bodyWidth / 2, 0.012, z + depth / 2,
          [color[0] * 0.25, color[1] * 0.25, color[2] * 0.25]);
    });

    const last = candles[candles.length - 1].close;
    tracks.push({
      symbol: feed.symbol,
      provider: feed.provider,
      quality: feed.quality,
      status: feed.status,
      z,
      count: candles.length,
      min: low,
      max: high,
      last,
      first,
      changePct: first ? ((last / first - 1) * 100) : 0,
      color: feed.color || paletteFor(trackIndex),
      width: candles.length * spacingX,
      // Enough to place a label on the newest close without rebuilding the vertices.
      scaleY,
      xOffset,
      lastX: xOffset + (candles.length - 1) * spacingX,
    });
  });

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    colors: new Float32Array(colors),
    tracks,
    vertexCount: positions.length / 3,
  };
}

function box(positions, normals, colors, x0, y0, z0, x1, y1, z1, color) {
  // Six faces, each two triangles, each with its own normal.
  const faces = [
    [[1, 0, 0], [[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]]],
    [[-1, 0, 0], [[x0, y0, z1], [x0, y0, z0], [x0, y1, z0], [x0, y1, z1]]],
    [[0, 1, 0], [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]],
    [[0, -1, 0], [[x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]]],
    [[0, 0, 1], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]],
    [[0, 0, -1], [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
  ];
  faces.forEach((face) => {
    const normal = face[0];
    const quad = face[1];
    const order = [0, 1, 2, 0, 2, 3];
    order.forEach((index) => {
      const point = quad[index];
      positions.push(point[0], point[1], point[2]);
      normals.push(normal[0], normal[1], normal[2]);
      colors.push(color[0], color[1], color[2]);
    });
  });
}

/* The floor grid and the vertical "wall" that gives the scene a horizon. */
function buildGrid(options) {
  const opts = options || {};
  const lines = opts.lines || 14;
  const step = opts.step || 0.9;
  const half = (lines * step) / 2;
  const positions = [];
  const colors = [];
  const addLine = (a, b, color) => {
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    colors.push(color[0], color[1], color[2], color[0], color[1], color[2]);
  };
  const zCenter = opts.zCenter || 0;
  const xCenter = opts.xCenter || 0;
  for (let i = 0; i <= lines; i += 1) {
    const offset = -half + i * step;
    const fade = 0.42 - Math.abs(offset / (half * 2.2)) * 0.3;
    const color = [0.18, 0.26, 0.34].map((c) => c * (0.6 + fade));
    addLine([xCenter - half, 0, zCenter + offset], [xCenter + half, 0, zCenter + offset], color);
    addLine([xCenter + offset, 0, zCenter - half], [xCenter + offset, 0, zCenter + half], color);
  }
  return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
}

const PALETTE = ['#6cc6f5', '#f2b134', '#67d18a', '#e86a92', '#b28df0', '#4fd1c5'];

function paletteFor(index) {
  return PALETTE[index % PALETTE.length];
}

const api = {
  identity, multiply, perspective, lookAt, orbitEye, project, buildTracks, buildGrid,
  paletteFor, hexToRgb, TRIANGLES_PER_CANDLE,
};

if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof window !== 'undefined') window.SceneMath = api;
