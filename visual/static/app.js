/* SynapseOS Visual Simulator — frontend (free-tier friendly) */

const COLORS = {
  processing: "#e74c3c",
  memory: "#3498db",
  routing: "#2ecc71",
  sensing: "#f39c12",
  decision: "#9b59b6",
};

let state = null;
let nodes = {};
let animId = null;
let busy = false;
let pollTimer = null;
let speedTimer = null;

const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const spark = document.getElementById("spark");
const sparkCtx = spark.getContext("2d");

function resize() {
  const wrap = canvas.parentElement;
  canvas.width = wrap.clientWidth * devicePixelRatio;
  canvas.height = wrap.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
window.addEventListener("resize", resize);
resize();

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

async function refresh() {
  try {
    state = await api("/api/state");
    updateUI();
    ensureNodes();
  } catch (e) {
    /* ignore transient network blips */
  }
}

function setBusy(on) {
  busy = on;
  document.querySelectorAll("button").forEach((b) => {
    if (b.id === "btn-stop") return; // stop always allowed
    b.disabled = on;
    b.style.opacity = on ? "0.6" : "1";
  });
}

function updateUI() {
  if (!state) return;
  document.getElementById("gen").textContent = state.generation;
  document.getElementById("ncells").textContent = state.n_cells;
  document.getElementById("nlinks").textContent = state.n_links;
  document.getElementById("acc").textContent = state.accuracy.toFixed(2);
  document.getElementById("energy").textContent = state.energy.toFixed(2);
  document.getElementById("fit").textContent = state.fitness.toFixed(2);
  if (document.getElementById("maturity")) {
    document.getElementById("maturity").textContent =
      state.avg_maturity != null ? state.avg_maturity.toFixed(2) : "—";
  }
  document.getElementById("message").textContent = state.message;

  const bar = document.getElementById("roles-bar");
  bar.innerHTML = "";
  const order = ["processing", "memory", "routing", "sensing", "decision"];
  order.forEach((k) => {
    const v = state.roles[k] || 0;
    const s = document.createElement("span");
    s.style.width = v * 100 + "%";
    s.style.background = COLORS[k];
    bar.appendChild(s);
  });

  const list = document.getElementById("role-list");
  list.innerHTML = order
    .map(
      (k) =>
        `<div><span class="swatch ${k}"></span>${k}: ${((state.roles[k] || 0) * 100).toFixed(0)}%</div>`
    )
    .join("");

  drawSpark();
}

function drawSpark() {
  const w = spark.width;
  const h = spark.height;
  sparkCtx.clearRect(0, 0, w, h);
  if (!state || !state.history || state.history.length < 2) return;
  const hist = state.history;
  sparkCtx.beginPath();
  sparkCtx.strokeStyle = "#3dd6c6";
  sparkCtx.lineWidth = 2;
  hist.forEach((p, i) => {
    const x = (i / (hist.length - 1)) * (w - 4) + 2;
    const y = h - 4 - (p.acc / 1.0) * (h - 8);
    if (i === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
  });
  sparkCtx.stroke();
}

function ensureNodes() {
  if (!state) return;
  const ids = new Set(state.cells.map((c) => c.id));
  Object.keys(nodes).forEach((id) => {
    if (!ids.has(id)) delete nodes[id];
  });
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  state.cells.forEach((c, i) => {
    if (!nodes[c.id]) {
      const angle = (i / state.cells.length) * Math.PI * 2;
      const r = Math.min(w, h) * 0.28;
      nodes[c.id] = {
        x: w / 2 + Math.cos(angle) * r + (Math.random() - 0.5) * 20,
        y: h / 2 + Math.sin(angle) * r + (Math.random() - 0.5) * 20,
        vx: 0,
        vy: 0,
      };
    }
  });
}

function tickPhysics() {
  if (!state) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const ids = state.cells.map((c) => c.id);

  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = nodes[ids[i]];
      const b = nodes[ids[j]];
      if (!a || !b) continue;
      let dx = a.x - b.x;
      let dy = a.y - b.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
      const force = 800 / (dist * dist);
      dx = (dx / dist) * force;
      dy = (dy / dist) * force;
      a.vx += dx;
      a.vy += dy;
      b.vx -= dx;
      b.vy -= dy;
    }
  }

  state.links.forEach((l) => {
    const a = nodes[l.source];
    const b = nodes[l.target];
    if (!a || !b) return;
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
    const force = (dist - 90) * 0.012 * l.strength;
    dx = (dx / dist) * force;
    dy = (dy / dist) * force;
    a.vx += dx;
    a.vy += dy;
    b.vx -= dx;
    b.vy -= dy;
  });

  ids.forEach((id) => {
    const n = nodes[id];
    if (!n) return;
    n.vx += (w / 2 - n.x) * 0.0015;
    n.vy += (h / 2 - n.y) * 0.0015;
    n.vx *= 0.85;
    n.vy *= 0.85;
    n.x += n.vx;
    n.y += n.vy;
    n.x = Math.max(20, Math.min(w - 20, n.x));
    n.y = Math.max(20, Math.min(h - 20, n.y));
  });
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!state) return;

  state.links.forEach((l) => {
    const a = nodes[l.source];
    const b = nodes[l.target];
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = `rgba(120, 160, 190, ${0.15 + 0.55 * l.strength})`;
    ctx.lineWidth = 0.6 + 2.4 * l.strength;
    ctx.stroke();
  });

  state.cells.forEach((c) => {
    const n = nodes[c.id];
    if (!n) return;
    const color = COLORS[c.dominant] || "#888";
    const r = 6 + 9 * Math.min(1, c.activity);
    const mat = c.maturity || 0;

    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 3 + mat * 3, 0, Math.PI * 2);
    ctx.fillStyle = color + (mat > 0.5 ? "44" : "28");
    ctx.fill();

    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = mat > 0.6 ? "#e6edf5" : "#0b0f14";
    ctx.lineWidth = mat > 0.6 ? 2 : 1.5;
    ctx.stroke();

    ctx.fillStyle = "#e6edf5";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(c.id.replace("C", ""), n.x, n.y + r + 11);
  });
}

function loop() {
  tickPhysics();
  draw();
  animId = requestAnimationFrame(loop);
}

// ------------------------------------------------------------------
// Controls — Start/Stop are instant; heavy actions show busy state
// ------------------------------------------------------------------
document.getElementById("btn-start").onclick = async () => {
  await api("/api/start", "POST");
  if (state) {
    state.running = true;
    state.message = "Running…";
    updateUI();
  }
  poll();
};

document.getElementById("btn-stop").onclick = async () => {
  await api("/api/stop", "POST");
  if (state) {
    state.running = false;
    state.message = "Paused";
    updateUI();
  }
};

document.getElementById("btn-step").onclick = async () => {
  if (busy) return;
  setBusy(true);
  try {
    state = await api("/api/step", "POST");
    updateUI();
    ensureNodes();
  } finally {
    setBusy(false);
  }
};

document.getElementById("btn-reset").onclick = async () => {
  if (busy) return;
  setBusy(true);
  try {
    state = await api("/api/reset", "POST", { n_cells: 20 });
    nodes = {};
    updateUI();
    ensureNodes();
  } finally {
    setBusy(false);
  }
};

document.getElementById("btn-damage").onclick = async () => {
  if (busy) return;
  setBusy(true);
  try {
    state = await api("/api/damage", "POST", { n: 2 });
    updateUI();
    ensureNodes();
  } finally {
    setBusy(false);
  }
};

document.getElementById("btn-killer").onclick = async () => {
  if (busy) return;
  if (!confirm("Destroy ~30% of cells and start recovery?")) return;
  setBusy(true);
  try {
    state = await api("/api/killer_demo", "POST", { frac: 0.3 });
    updateUI();
    ensureNodes();
    await api("/api/start", "POST");
    if (state) state.running = true;
    poll();
  } finally {
    setBusy(false);
  }
};

document.getElementById("task-select").onchange = async (e) => {
  await api("/api/task", "POST", { task: e.target.value });
};

// Debounce speed slider — was flooding the server on every tick
document.getElementById("speed").oninput = (e) => {
  clearTimeout(speedTimer);
  speedTimer = setTimeout(() => {
    api("/api/speed", "POST", { speed: parseFloat(e.target.value) });
  }, 200);
};

function poll() {
  if (pollTimer) clearInterval(pollTimer);
  // 600ms — much less load than 200ms on free CPU
  pollTimer = setInterval(async () => {
    await refresh();
    if (state && !state.running) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 600);
}

refresh().then(() => {
  loop();
});