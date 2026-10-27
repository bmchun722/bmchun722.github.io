// Murray's reaction-diffusion model for animal coat patterns (Thomas substrate-inhibition kinetics).
//   u_t = f(u,v) + lap(u),   v_t = g(u,v) + d lap(v)      (lengths rescaled by sqrt(gamma))
//   f = a - u - h(u,v),  g = alpha (b - v) - h(u,v),  h = rho u v / (1 + u + K u^2)
// Numerics: method of lines, 5-point finite-difference Laplacian, explicit forward Euler.

const MURRAY_KINETICS = { a: 92, b: 64, alpha: 1.5, rho: 18.5, K: 0.1 };

function reactionTerms(p, u, v) {
  const h = p.rho * u * v / (1 + u + p.K * u * u);
  return [p.a - u - h, p.alpha * (p.b - v) - h];
}

// Uniform steady state: f = g = 0. From f = g, v = b - (a - u)/alpha; bisect f(u, v(u)) = 0.
function steadyState(p) {
  const vOf = u => p.b - (p.a - u) / p.alpha;
  let lo = Math.max(0, p.a - p.alpha * p.b) + 1e-12, hi = p.a;
  for (let i = 0; i < 200; i++) {
    const m = 0.5 * (lo + hi);
    if (reactionTerms(p, m, vOf(m))[0] > 0) lo = m; else hi = m;
  }
  const u = 0.5 * (lo + hi);
  return { u, v: vOf(u) };
}

// Linear stability: Jacobian at the steady state, critical diffusion ratio d_c and
// the band of unstable wavenumbers for a given d.
function linearStability(p, d) {
  const s = steadyState(p), e = 1e-6;
  const fu = (reactionTerms(p, s.u + e, s.v)[0] - reactionTerms(p, s.u - e, s.v)[0]) / (2 * e);
  const fv = (reactionTerms(p, s.u, s.v + e)[0] - reactionTerms(p, s.u, s.v - e)[0]) / (2 * e);
  const gu = (reactionTerms(p, s.u + e, s.v)[1] - reactionTerms(p, s.u - e, s.v)[1]) / (2 * e);
  const gv = (reactionTerms(p, s.u, s.v + e)[1] - reactionTerms(p, s.u, s.v - e)[1]) / (2 * e);
  const det = fu * gv - fv * gu;
  // Onset condition (d fu + gv)^2 = 4 d det -> fu^2 d^2 + (2 fu gv - 4 det) d + gv^2 = 0, larger root.
  const B = 2 * fu * gv - 4 * det;
  const dc = (-B + Math.sqrt(B * B - 4 * fu * fu * gv * gv)) / (2 * fu * fu);
  const kc2 = (dc * fu + gv) / (2 * dc);
  // Unstable band for this d: det(J - k^2 diag(1,d)) < 0.
  const qa = d, qb = -(d * fu + gv), qc = det, disc = qb * qb - 4 * qa * qc;
  let band = null;
  if (d > dc && disc > 0) {
    band = [(-qb - Math.sqrt(disc)) / (2 * qa), (-qb + Math.sqrt(disc)) / (2 * qa)].map(Math.sqrt);
  }
  return { steady: s, fu, fv, gu, gv, det, trace: fu + gv, dc, lambdaC: 2 * Math.PI / Math.sqrt(kc2), band };
}

// Grid made of rows (index j) of possibly different widths W[j]. Precomputes neighbour indices so that
// a flat rectangle (Neumann in x) and an unrolled tapering tail (periodic in x) share one stepper.
//   geometry: { kind: 'body', Lx, Ly }  or  { kind: 'tail', length, cRoot, cTip }
function buildGrid(geometry, hTarget) {
  const periodic = geometry.kind === 'tail';
  const lenY = periodic ? geometry.length : geometry.Ly;
  const widthAt = periodic
    ? (y => geometry.cRoot + (geometry.cTip - geometry.cRoot) * y / lenY)
    : (() => geometry.Lx);
  // Keep at least ~8 cells across the smallest dimension so tiny domains are still resolved
  // (finer grids would make dt ~ h^2 prohibitively small).
  const minDim = periodic ? lenY : Math.min(geometry.Lx, geometry.Ly);
  const h = Math.min(hTarget, minDim / 8);
  const ny = Math.max(2, Math.round(lenY / h)), hy = lenY / ny;
  const W = new Int32Array(ny), start = new Int32Array(ny + 1), hxRow = new Float64Array(ny);
  for (let j = 0; j < ny; j++) {
    const c = widthAt((j + 0.5) * hy);
    W[j] = Math.max(1, Math.round(c / h));
    hxRow[j] = c / W[j];
    start[j + 1] = start[j] + W[j];
  }
  const n = start[ny];
  const L = new Int32Array(n), R = new Int32Array(n), U = new Int32Array(n), D = new Int32Array(n);
  const cx = new Float64Array(n), row = new Int32Array(n);
  // Map cell i of row j onto the row jj with the same fractional position (exact for equal widths).
  const mapRow = (j, i, jj) => start[jj] + Math.min(W[jj] - 1, Math.floor((i + 0.5) * W[jj] / W[j]));
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < W[j]; i++) {
      const k = start[j] + i;
      row[k] = j;
      if (periodic) {
        L[k] = start[j] + (i - 1 + W[j]) % W[j];
        R[k] = start[j] + (i + 1) % W[j];
      } else {
        // Neumann via reflected ghost cell: u_{-1} = u_{1}.
        L[k] = start[j] + (i > 0 ? i - 1 : Math.min(1, W[j] - 1));
        R[k] = start[j] + (i < W[j] - 1 ? i + 1 : Math.max(W[j] - 2, 0));
      }
      U[k] = mapRow(j, i, j > 0 ? j - 1 : Math.min(1, ny - 1));
      D[k] = mapRow(j, i, j < ny - 1 ? j + 1 : Math.max(ny - 2, 0));
      cx[k] = W[j] > 1 ? 1 / (hxRow[j] * hxRow[j]) : 0;
    }
  }
  // Rows with a single cell have no x-diffusion (cx = 0), so they do not constrain dt.
  const hMin = Math.min(hy, ...Array.from(hxRow).filter((_, j) => W[j] > 1));
  return { geometry, periodic, n, ny, hy, W, start, row, L, R, U, D, cx, cy: 1 / (hy * hy), hMin };
}

class MurraySimulation {
  constructor(params, geometry, { hTarget = 0.7, noise = 0.02, seed = 1 } = {}) {
    this.p = { ...MURRAY_KINETICS, ...params };
    this.grid = buildGrid(geometry, hTarget);
    this.noise = noise;
    this.reset(seed);
  }

  reset(seed = 1) {
    const n = this.grid.n, s = steadyState(this.p);
    let state = (seed * 2654435761) % 2147483647 || 1;
    const rnd = () => (state = (state * 16807) % 2147483647) / 2147483647;
    this.u = new Float64Array(n); this.v = new Float64Array(n);
    this.u2 = new Float64Array(n); this.v2 = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.u[k] = s.u * (1 + this.noise * (rnd() - 0.5));
      this.v[k] = s.v * (1 + this.noise * (rnd() - 0.5));
    }
    this.t = 0;
  }

  // Forward Euler is stable for dt < h^2 / (4 d) (diffusion); 0.02 bounds the stiff reaction part.
  timeStep() { return Math.min(0.9 * this.grid.hMin * this.grid.hMin / (4 * Math.max(1, this.p.d)), 0.02); }

  step(nSteps) {
    const { a, b, alpha, rho, K, d } = this.p;
    const { n, L, R, U, D, cx, cy } = this.grid;
    const dt = this.timeStep();
    let u = this.u, v = this.v, un = this.u2, vn = this.v2;
    for (let s = 0; s < nSteps; s++) {
      for (let k = 0; k < n; k++) {
        const uk = u[k], vk = v[k];
        const h = rho * uk * vk / (1 + uk + K * uk * uk);
        const lapU = (u[L[k]] + u[R[k]] - 2 * uk) * cx[k] + (u[U[k]] + u[D[k]] - 2 * uk) * cy;
        const lapV = (v[L[k]] + v[R[k]] - 2 * vk) * cx[k] + (v[U[k]] + v[D[k]] - 2 * vk) * cy;
        const uNew = uk + dt * (a - uk - h + lapU);
        const vNew = vk + dt * (alpha * (b - vk) - h + d * lapV);
        un[k] = uNew > 0 ? uNew : 0;
        vn[k] = vNew > 0 ? vNew : 0;
      }
      const tu = u; u = un; un = tu;
      const tv = v; v = vn; vn = tv;
    }
    this.u = u; this.v = v; this.u2 = un; this.v2 = vn;
    this.t += nSteps * dt;
  }
}

if (typeof module !== 'undefined') {
  module.exports = { MURRAY_KINETICS, reactionTerms, steadyState, linearStability, buildGrid, MurraySimulation };
}
