/*
 * gta6-twitch-forecast — hourly Twitch concurrent-viewer quantiles for a game's
 * release day, forecast from its first observed hour.
 *
 * Model, in one paragraph: every comparable launch is re-expressed as log
 * ratios to its own hour-1 count, which cancels scale and leaves the day's
 * shape. Those shapes are pooled with a gaussian kernel on log hour-1 viewers,
 * because launch size is what actually separates profiles — big launches decay
 * far less in relative terms than small ones (leave-one-out pinball 0.39 vs
 * 0.65 for a pre-launch-jump kernel). The pooled centre is the weighted median
 * of the log ratios per hour, smoothed across neighbouring hours; the spread is
 * the weighted MAD, shrunk slightly for the homogeneous pool and widened when
 * the effective sample is thin or the case sits above every comparable. A
 * clamped slope on the case's own pre-launch jump nudges the whole curve (never
 * its individual hours), 30% of pooled empirical quantiles are blended into the
 * gaussian form to keep the fat tails real, and the five levels are forced
 * non-decreasing. Deterministic: no clock, no network, no randomness.
 *
 * Node 22 strips the types, so this file runs as-is: `node main.ts`.
 */

"use strict";

const fs = require("node:fs");
const path = require("node:path");

/* ---------------------------------------------------------------- constants */

const HOURS = [];
for (let h = 2; h <= 24; h++) HOURS.push(h);

const TAUS = [0.05, 0.25, 0.5, 0.75, 0.95];
const ZS = [
  -1.6448536269514722,
  -0.6744897501960817,
  0.0,
  0.6744897501960817,
  1.6448536269514722,
];

const MAX_LR = 6.0; // log-ratio clamps: e^6 = 403x, e^-6 = 1/403
const MIN_SAMPLED = 6; // a comparable needs this many of hours 2..24 to be used
const SIZE_BW = 1.0; // kernel width on ln(hour-1 viewers)
const MIN_EFF = 6.0; // effective-sample target before the spread is widened
const SHRINK = 0.85; // the sized pool is more homogeneous than the raw one
const SD_FLOOR = 0.18;
const SD_CAP = 1.9;
const B_SHRINK = 0.5; // trust in the pre-launch-jump slope
const B_CAP = 0.35;
const SHIFT_CAP = 0.3;
const W_EMP = 0.3; // weight of pooled empirical quantiles in the blend
const Q_MIN = 1e-3;
const Q_MAX = 1e9;

/*
 * Prior shape, used when a case brings no usable comparables: geometric mean
 * log ratio (hour h / hour 1) and its log-sd, measured over the launches in the
 * kit with at least 50k hour-1 viewers (hours 2..24).
 */
const PRIOR_RATIOS = [
  0.9672, 0.8074, 0.6994, 0.7158, 0.7543, 0.7205, 0.7187, 0.6887, 0.6737,
  0.6857, 0.7217, 0.8252, 0.9398, 1.0472, 1.1223, 1.2408, 1.297, 1.3905,
  1.4031, 1.414, 1.3827, 1.3185, 1.2123,
];
const PRIOR_SD = [
  0.193, 0.245, 0.289, 0.295, 0.415, 0.485, 0.457, 0.459, 0.473, 0.474, 0.466,
  0.43, 0.425, 0.432, 0.431, 0.41, 0.401, 0.407, 0.422, 0.414, 0.436, 0.421,
  0.432,
];

/* ------------------------------------------------------------------ helpers */

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  return x < lo ? lo : x > hi ? hi : x;
}

function num(v) {
  const x = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(x) ? x : null;
}

/** ISO string or epoch number to milliseconds, or null. */
function timeMs(v) {
  if (typeof v === "number" && Number.isFinite(v)) {
    return v > 1e11 ? v : v * 1000;
  }
  if (typeof v === "string") {
    const t = Date.parse(v);
    return Number.isFinite(t) ? t : null;
  }
  return null;
}

function median(values) {
  if (!values.length) return null;
  const a = values.slice().sort((x, y) => x - y);
  const n = a.length;
  return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

/** Linearly interpolated percentile, q in [0,1]. */
function percentile(values, q) {
  if (!values.length) return null;
  const a = values.slice().sort((x, y) => x - y);
  if (a.length === 1) return a[0];
  const i = q * (a.length - 1);
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, a.length - 1);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

/** Weighted median of {v, w} pairs. */
function wMedian(pairs) {
  if (!pairs.length) return null;
  const a = pairs.slice().sort((p, q) => p.v - q.v);
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.max(a[i].w, 0);
  if (!(total > 0)) return a[a.length >> 1].v;
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const w = Math.max(a[i].w, 0);
    if (w > 0) acc += w;
    if (acc >= total * 0.5) return a[i].v;
  }
  return a[a.length - 1].v;
}

/** Weighted quantile of {v, w} pairs, tau in (0,1). */
function wQuantile(pairs, tau) {
  if (!pairs.length) return null;
  const a = pairs.slice().sort((p, q) => p.v - q.v);
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.max(a[i].w, 0);
  if (!(total > 0)) return a[0].v;
  const target = tau * total;
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const w = Math.max(a[i].w, 0);
    if (w > 0 && acc + w >= target) return a[i].v;
    acc += w;
  }
  return a[a.length - 1].v;
}

/** Weighted least-squares slope of y on x; null when x cannot vary. */
function wSlope(xs, ys, ws) {
  let sw = 0;
  for (let i = 0; i < ws.length; i++) sw += ws[i];
  if (!(sw > 0)) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < xs.length; i++) {
    mx += ws[i] * xs[i];
    my += ws[i] * ys[i];
  }
  mx /= sw;
  my /= sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  if (!(sxx > 1e-9)) return null;
  return sxy / sxx;
}

/* ------------------------------------------------------------- comparables */

/**
 * Turn one comparable launch into log ratios for hours 2..24 relative to its
 * own hour 1 (offset 0 from its day_start), plus its pre-launch jump.
 * Returns null when it cannot carry a shape.
 */
function buildComparable(comp) {
  if (!comp || typeof comp !== "object") return null;
  const hourly = Array.isArray(comp.hourly) ? comp.hourly : [];
  const t0 = timeMs(comp.day_start);
  if (t0 === null || hourly.length < 6) return null;

  // mean of the points that land in the same hour
  const sum = new Map();
  const cnt = new Map();
  for (let i = 0; i < hourly.length; i++) {
    const row = hourly[i];
    if (!row) continue;
    const t = timeMs(row.from);
    const v = num(row.viewers);
    if (t === null || v === null || !(v >= 0)) continue;
    const off = Math.round((t - t0) / 3600000);
    if (!Number.isFinite(off) || off < -400 || off > 400) continue;
    sum.set(off, (sum.get(off) || 0) + v);
    cnt.set(off, (cnt.get(off) || 0) + 1);
  }
  const at = new Map();
  sum.forEach((s, off) => at.set(off, s / Math.max(cnt.get(off) || 1, 1)));

  let off0 = 0;
  if (!(at.get(0) > 0)) {
    // tolerate a day_start that is off by one hour either way
    if (at.get(-1) > 0) off0 = -1;
    else if (at.get(1) > 0) off0 = 1;
    else return null;
  }
  const v0 = at.get(off0);
  if (!(v0 > 0)) return null;

  const ratios = new Map();
  let sampled = 0;
  for (let i = 0; i < HOURS.length; i++) {
    const h = HOURS[i];
    // hour h of the release day is the (h-1)-th hour after the day's first
    // hour, which is offset 0 — exactly what the case's `known` row measures.
    const v = at.get(off0 + h - 1);
    if (!(v > 0)) continue;
    ratios.set(h, clamp(Math.log(v / v0), -MAX_LR, MAX_LR));
    sampled++;
  }
  if (sampled < MIN_SAMPLED) return null;

  // pre-launch jump: hour 1 against the same hours of the previous 14 days
  const base = [];
  for (let off = -336; off <= -1; off += 24) {
    const v = at.get(off0 + off);
    if (v > 0) base.push(v);
  }
  const b0 = base.length >= 3 ? median(base) : null;
  const k = b0 && b0 > 0 ? clamp(v0 / b0, 1e-9, 1e12) : null;

  return { v0: v0, ratios: ratios, k: k, name: String(comp.name || "") };
}

/** The case's own pre-launch jump, on the same footing as the comparables'. */
function caseJump(c, anchor) {
  const hh = Array.isArray(c && c.history_hourly) ? c.history_hourly : [];
  const cp = timeMs(c && c.checkpoint);
  const use = [];
  for (let i = 0; i < hh.length; i++) {
    const row = hh[i];
    if (!row) continue;
    const v = num(row.viewers);
    if (v === null || !(v > 0)) continue;
    const t = timeMs(row.from);
    if (cp !== null && t !== null && (cp - t) / 3600000 > 73) continue;
    use.push({ t: t === null ? 0 : t, v: v });
  }
  use.sort((a, b) => a.t - b.t);
  const tail = use.slice(-72).map((r) => r.v);
  if (tail.length < 6) return null;
  const b = percentile(tail, 0.2);
  if (!(b > 0)) return null;
  return clamp(Math.log(anchor / b), -3.0, 8.5);
}

/**
 * Small clamped correction from the last minutes before the checkpoint: if the
 * known hour is still climbing (or falling) inside itself, the next few hours
 * follow it a little. Fades out by hour 5 and can never move more than 15%.
 */
function hourOneTrend(c) {
  const hist = Array.isArray(c && c.history) ? c.history : [];
  const pts = [];
  for (let i = 0; i < hist.length; i++) {
    const row = hist[i];
    if (!row) continue;
    const t = timeMs(row.at);
    const v = num(row.viewers);
    if (t === null || v === null || !(v > 0)) continue;
    pts.push({ t: t, v: v });
  }
  if (pts.length < 6) return 0;
  pts.sort((a, b) => a.t - b.t);
  const span = pts[pts.length - 1].t - pts[0].t;
  if (!(span > 0)) return 0;
  const half = pts[pts.length - 1].t - span / 2;
  const early = [];
  const late = [];
  for (let i = 0; i < pts.length; i++) {
    (pts[i].t <= half ? early : late).push(Math.log(pts[i].v));
  }
  if (early.length < 2 || late.length < 2) return 0;
  let me = 0;
  for (let i = 0; i < early.length; i++) me += early[i];
  me /= early.length;
  let ml = 0;
  for (let i = 0; i < late.length; i++) ml += late[i];
  ml /= late.length;
  const hours = span / 3600000;
  if (!(hours > 0)) return 0;
  return clamp(((ml - me) / hours) * 0.5, -0.15, 0.15);
}

/* --------------------------------------------------------------------- fit */

/** No usable comparables: the prior shape, with its own measured spread. */
function fallbackSolve(anchor) {
  const med = new Map();
  const sd = new Map();
  const emp = new Map();
  for (let i = 0; i < HOURS.length; i++) {
    const h = HOURS[i];
    const m = Math.log(PRIOR_RATIOS[i]);
    const s = Math.max(PRIOR_SD[i], SD_FLOOR);
    med.set(h, m);
    sd.set(h, s);
    emp.set(
      h,
      TAUS.map((tau) => m + ZS[TAUS.indexOf(tau)] * s)
    );
  }
  return { p: anchor, med: med, sd: sd, emp: emp, nudge: 0 };
}

/**
 * Fit the pooled shape to one case. Returns {p, med, sd, emp, nudge} where med
 * and sd are log ratios to p, per hour.
 */
function solve(c) {
  // --- hour-1 anchor, degrading through the other fields --------------------
  let p = null;
  const known = Array.isArray(c && c.known) ? c.known : [];
  for (let i = 0; i < known.length && p === null; i++) {
    const v = num(known[i] && known[i].viewers);
    if (v !== null && v > 0) p = v;
  }
  if (p === null) {
    const hist = Array.isArray(c && c.history) ? c.history : [];
    for (let i = hist.length - 1; i >= 0 && p === null; i--) {
      const v = num(hist[i] && hist[i].viewers);
      if (v !== null && v > 0) p = v;
    }
  }
  if (p === null) {
    const hh = Array.isArray(c && c.history_hourly) ? c.history_hourly : [];
    for (let i = hh.length - 1; i >= 0 && p === null; i--) {
      const v = num(hh[i] && hh[i].viewers);
      if (v !== null && v > 0) p = v;
    }
  }
  const anchor = p === null ? 1 : clamp(p, 1e-3, 1e12);

  // --- pool the comparable launches ----------------------------------------
  const rawComps = Array.isArray(c && c.comparables) ? c.comparables : [];
  const comps = [];
  for (let i = 0; i < rawComps.length; i++) {
    const cc = buildComparable(rawComps[i]);
    if (cc) comps.push(cc);
  }
  if (comps.length < 2) return fallbackSolve(anchor);

  const lnp = clamp(Math.log(anchor), 0, Math.log(1e9));
  let maxLn = -Infinity;
  const ws = [];
  for (let i = 0; i < comps.length; i++) {
    const lv = comps[i].v0 > 0 ? Math.log(comps[i].v0) : 0;
    if (lv > maxLn) maxLn = lv;
    const d = (lv - lnp) / SIZE_BW;
    ws.push(Math.max(Math.exp(-0.5 * d * d), 1e-12));
  }

  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < ws.length; i++) {
    s1 += ws[i];
    s2 += ws[i] * ws[i];
  }
  const neff = s2 > 0 ? (s1 * s1) / s2 : 0;
  // thin pool: widen, because a handful of launches cannot pin a spread down
  let widen =
    neff < MIN_EFF ? Math.min(1.7, Math.sqrt(MIN_EFF / Math.max(neff, 0.5))) : 1.0;
  // the case is bigger than anything in the pool: the curve is being extrapolated
  if (lnp > maxLn) widen *= 1 + Math.min(0.35, 0.3 * (lnp - maxLn));

  const lnkCase = caseJump(c, anchor);

  const med = new Map();
  const sd = new Map();
  const emp = new Map();
  for (let hi = 0; hi < HOURS.length; hi++) {
    const h = HOURS[hi];
    const pairs = [];
    const lk = [];
    const lr = [];
    const lw = [];
    let lkSum = 0;
    let lkWeight = 0;
    for (let i = 0; i < comps.length; i++) {
      const r = comps[i].ratios.get(h);
      if (r === undefined) continue;
      pairs.push({ v: r, w: ws[i] });
      if (comps[i].k !== null) {
        const x = clamp(Math.log(comps[i].k), -3.0, 8.5);
        lk.push(x);
        lr.push(r);
        lw.push(ws[i]);
        lkSum += ws[i] * x;
        lkWeight += ws[i];
      }
    }
    if (!pairs.length) continue;

    let m = wMedian(pairs);
    if (m === null) continue;

    // one clamped, curve-wide correction from the case's own pre-launch jump
    if (lnkCase !== null && lk.length >= 5 && lkWeight > 0) {
      const slope = wSlope(lk, lr, lw);
      if (slope !== null) {
        const b = clamp(B_SHRINK * slope, -B_CAP, B_CAP);
        const shift = clamp(
          b * (lnkCase - lkSum / lkWeight),
          -SHIFT_CAP,
          SHIFT_CAP
        );
        m += shift;
      }
    }
    m = clamp(m, -MAX_LR, MAX_LR);
    med.set(h, m);

    const dev = [];
    for (let i = 0; i < pairs.length; i++) {
      dev.push({ v: Math.abs(pairs[i].v - m), w: pairs[i].w });
    }
    const mad = wMedian(dev);
    sd.set(
      h,
      clamp(1.4826 * (mad === null ? 0.5 : mad) * SHRINK * widen, SD_FLOOR, SD_CAP)
    );
    emp.set(h, TAUS.map((tau) => wQuantile(pairs, tau)));
  }
  if (med.size === 0) return fallbackSolve(anchor);

  // --- smooth the centre across neighbouring hours -------------------------
  const used = HOURS.filter((h) => med.has(h));
  const raw = used.map((h) => med.get(h));
  for (let i = 1; i < used.length - 1; i++) {
    med.set(used[i], 0.25 * raw[i - 1] + 0.5 * raw[i] + 0.25 * raw[i + 1]);
  }

  // --- the spread only ever grows with the hour ----------------------------
  let run = SD_FLOOR;
  for (let i = 0; i < used.length; i++) {
    const s = sd.get(used[i]);
    run = Math.max(run, s === undefined ? SD_FLOOR : s);
    sd.set(used[i], run);
  }

  return { p: anchor, med: med, sd: sd, emp: emp, nudge: hourOneTrend(c) };
}

/* ------------------------------------------------------------------- output */

/** Build the 23 output rows from a fit. Never throws. */
function emit(fit) {
  const rows = [];
  for (let hi = 0; hi < HOURS.length; hi++) {
    const h = HOURS[hi];
    let m = fit.med.get(h);
    let s = fit.sd.get(h);
    if (m === undefined) m = Math.log(PRIOR_RATIOS[hi]);
    if (s === undefined) s = Math.max(PRIOR_SD[hi], SD_FLOOR);
    const e = fit.emp.get(h);
    // the trend nudge is one number for the whole curve, fading by hour 5
    const nudge = (fit.nudge || 0) * Math.exp(-(h - 2) / 2.5);

    const q = [];
    for (let j = 0; j < TAUS.length; j++) {
      const param = m + ZS[j] * s;
      const ratio =
        e && Number.isFinite(e[j]) ? (1 - W_EMP) * param + W_EMP * e[j] : param;
      const v = fit.p * Math.exp(clamp(ratio + nudge, -20.0, 20.0));
      q.push(clamp(v, Q_MIN, Q_MAX));
    }
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    for (let j = 0; j < q.length; j++) q[j] = clamp(q[j], Q_MIN, Q_MAX);
    rows.push({ hour: h, q: q });
  }
  return { hours: rows };
}

/* --------------------------------------------------------------------- main */

function safeId(caseObj, stem) {
  const raw = caseObj && typeof caseObj.id === "string" ? caseObj.id : stem;
  const clean = String(raw).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120);
  return clean || stem || "case";
}

function runFile(file, inDir, outDir) {
  const stem = file.replace(/\.in$/, "");
  let caseObj = null;
  try {
    const text = fs.readFileSync(path.join(inDir, file), "utf8");
    caseObj = JSON.parse(text);
  } catch (err) {
    caseObj = null;
  }
  let out = null;
  try {
    out = emit(caseObj ? solve(caseObj) : fallbackSolve(1000));
  } catch (err) {
    try {
      out = emit(fallbackSolve(1000));
    } catch (err2) {
      out = { hours: HOURS.map((h) => ({ hour: h, q: [1, 1, 1, 1, 1] })) };
    }
  }
  fs.writeFileSync(path.join(outDir, safeId(caseObj, stem) + ".out"), JSON.stringify(out));
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  try {
    fs.mkdirSync(outDir, { recursive: true });
  } catch (err) {
    /* the directory may already exist or be unwritable; the write will report it */
  }

  let files = [];
  try {
    files = fs
      .readdirSync(inDir)
      .filter((f) => typeof f === "string" && f.slice(-3) === ".in")
      .sort();
  } catch (err) {
    files = [];
  }

  for (let i = 0; i < files.length; i++) {
    try {
      runFile(files[i], inDir, outDir);
    } catch (err) {
      /* one bad case must never stop the others */
    }
  }
}

main();
