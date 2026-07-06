/**
 * Monte Carlo race simulator.
 *
 * The deployed site is fully static and the trained model cannot run in the
 * browser, but every prediction JSON ships each driver's continuous
 * `predictedPosition` score. We treat that score as the centre of a driver's
 * race-day pace and sample many noisy outcomes to produce win / podium / points
 * probabilities, the way bookmakers quote F1 odds.
 */

// Standard normal sample via Box-Muller. (Math.random is fine in the app; it is
// only forbidden inside Workflow orchestration scripts.)
export function gaussian() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Base spread of race-day pace, in "positions", before the chaos multiplier.
const BASE_SIGMA = 1.5;

// Per-run probability that a driver retires when DNFs are enabled.
const DNF_CHANCE = 0.08;

export const CHAOS_PRESETS = [
  { key: 'dry', label: 'Dry', multiplier: 1.0, hint: 'Clean, predictable race' },
  { key: 'damp', label: 'Damp', multiplier: 1.6, hint: 'Mixed conditions' },
  { key: 'wet', label: 'Wet', multiplier: 2.4, hint: 'Rain shakes up the order' },
  { key: 'chaos', label: 'Chaos', multiplier: 3.2, hint: 'Safety cars, mayhem' },
];

export function getChaosMultiplier(chaosKey) {
  const preset = CHAOS_PRESETS.find((p) => p.key === chaosKey);
  return preset ? preset.multiplier : 1.0;
}

/**
 * Run the simulation.
 *
 * @param {Array<{driverName:string, predictedPosition:number, driverNumber?:number}>} drivers
 * @param {Object} opts
 * @param {number} opts.runs        number of simulated races (e.g. 10000)
 * @param {string} opts.chaos       chaos preset key (dry|damp|wet|chaos)
 * @param {boolean} opts.dnfEnabled whether reliability failures occur
 * @param {Object<string,number>} opts.paceOffsets  driverName -> pace nudge (negative = faster)
 * @returns {{ runs:number, results: Array }} per-driver aggregated probabilities,
 *          sorted by win probability descending.
 */
export function runMonteCarlo(drivers, opts = {}) {
  const {
    runs = 10000,
    chaos = 'dry',
    dnfEnabled = false,
    paceOffsets = {},
  } = opts;

  const sigma = BASE_SIGMA * getChaosMultiplier(chaos);
  const n = drivers.length;
  const fieldSize = n;

  // Tallies indexed by driver array position.
  const wins = new Array(n).fill(0);
  const podiums = new Array(n).fill(0);
  // Top 5 rather than points (top 10): the field is the model's predicted top
  // 10, so a top-10 finish is trivially guaranteed and uninformative.
  const top5 = new Array(n).fill(0);
  const finishSum = new Array(n).fill(0);
  const best = new Array(n).fill(Infinity);
  const worst = new Array(n).fill(0);
  // histogram[i][pos-1] = how often driver i finished in position `pos`
  const histogram = Array.from({ length: n }, () => new Array(fieldSize).fill(0));

  const order = new Array(n);
  const scratch = new Array(n);

  for (let r = 0; r < runs; r++) {
    for (let i = 0; i < n; i++) {
      const offset = paceOffsets[drivers[i].driverName] || 0;
      let score = drivers[i].predictedPosition + offset + gaussian() * sigma;
      // A retirement drops the driver to the back of the field for this run.
      if (dnfEnabled && Math.random() < DNF_CHANCE) {
        score += 100 + Math.random() * 10;
      }
      scratch[i] = score;
      order[i] = i;
    }

    // Sort driver indices by sampled score ascending (lower = better finish).
    order.sort((a, b) => scratch[a] - scratch[b]);

    for (let pos = 0; pos < n; pos++) {
      const di = order[pos];
      const finish = pos + 1;
      finishSum[di] += finish;
      histogram[di][pos] += 1;
      if (finish < best[di]) best[di] = finish;
      if (finish > worst[di]) worst[di] = finish;
      if (pos === 0) wins[di] += 1;
      if (pos < 3) podiums[di] += 1;
      if (pos < 5) top5[di] += 1;
    }
  }

  const results = drivers.map((driver, i) => ({
    driverName: driver.driverName,
    driverNumber: driver.driverNumber,
    predictedPosition: driver.predictedPosition,
    winPct: (wins[i] / runs) * 100,
    podiumPct: (podiums[i] / runs) * 100,
    top5Pct: (top5[i] / runs) * 100,
    expectedFinish: finishSum[i] / runs,
    best: best[i] === Infinity ? null : best[i],
    worst: worst[i],
    histogram: histogram[i].map((c) => c / runs), // probability per finishing slot
  }));

  results.sort((a, b) => b.winPct - a.winPct || a.expectedFinish - b.expectedFinish);

  return { runs, results };
}
