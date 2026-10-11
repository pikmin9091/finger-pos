/* Dashboard client: REST actions + WS state feed with reconnect. */
const $ = (id) => document.getElementById(id);
const seenLog = new Set();
let wsTimer = null;

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...opts,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}
const post = (p, body) => api(p, {method: "POST", body: body ? JSON.stringify(body) : undefined});

function toast(t) {
  const el = document.getElementById("toast");
  el.textContent = t;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.classList.remove("show"); }, 3000);
}

function pill(id, mode) { // mode: on|bad|off
  const el = $(id);
  el.classList.remove("on", "bad");
  if (mode === "on") el.classList.add("on");
  if (mode === "bad") el.classList.add("bad");
}

function badge(el, text, cls) {
  el.textContent = text;
  el.className = "badge" + (cls ? " " + cls : "");
}

function render(s) {
  const cam = s.camera || {};
  pill("h-cam", cam.running ? "on" : "bad");
  pill("h-back", "on");
  const g = s.gesture || {};
  $("g-name").textContent = g.gesture || "—";
  $("g-conf").style.width = `${Math.round((g.confidence || 0) * 100)}%`;
  const st = g.state || "UNKNOWN";
  badge($("g-state"), st, st === "CONFIRMED" ? "ok" : st === "CANDIDATE" ? "warn" : "");
  $("g-hands").textContent = s.hands ?? 0;
  $("g-faces").textContent = s.faces ?? 0;
  const c = s.control || {};
  $("m-track").textContent = c.now_playing || "no player";
  $("m-ctl").textContent = c.enabled ? "gestures ON" : "gestures OFF";
  const pct = c.volume_pct;
  $("v-pct").textContent = pct === null || pct === undefined ? "—" : pct;
  if (document.activeElement !== $("v-slider") && pct !== null && pct !== undefined)
    $("v-slider").value = pct;
  $("v-muted").textContent = c.muted ? "yes" : "no";
  badge($("v-mode"), c.volume_mode ? (c.volume_state?.state || "ACTIVE") : "manual",
        c.volume_mode ? "ok" : "");
  for (const e of (s.events || [])) {
    if (seenLog.has(e.id)) continue;
    seenLog.add(e.id);
    const li = document.createElement("li");
    li.innerHTML = `<b>${new Date(e.t * 1000).toLocaleTimeString()}</b> ${e.text}`;
    const log = $("log");
    log.prepend(li);
    while (log.children.length > 60) log.lastChild.remove();
  }
  if (seenLog.size > 400) seenLog.clear();
}

async function refreshSettings() {
  try {
    const s = await api("/api/settings");
    $("s-smooth").value = Math.round(s.smoothing * 100);
    $("s-smooth-v").textContent = Number(s.smoothing).toFixed(2);
    $("s-debounce").value = s.debounce;
    $("s-debounce-v").textContent = s.debounce;
    $("s-strength").value = s.strength;
    $("s-strength-v").textContent = s.strength;
    $("s-kind").value = s.blur_kind;
    $("s-privacy").value = s.privacy_mode;
    $("s-sens").value = s.sensitivity;
  } catch (e) { toast("settings: " + e.message); }
}

function connectWS() {
  clearTimeout(wsTimer);
  let ws;
  try {
    ws = new WebSocket(`ws://${location.host}/ws`);
  } catch (e) {
    pill("h-ws", "bad");
    wsTimer = setTimeout(connectWS, 2000);
    return;
  }
  ws.onopen = () => pill("h-ws", "on");
  ws.onmessage = (ev) => {
    try { render(JSON.parse(ev.data)); }
    catch (e) { /* ignore one bad frame */ }
  };
  ws.onclose = () => {
    pill("h-ws", "bad");
    wsTimer = setTimeout(connectWS, 2000);
  };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
}

function bind(id, fn) {
  $(id).addEventListener("click", async () => {
    try { await fn(); } catch (e) { toast(e.message); }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  connectWS();
  refreshSettings();
  const reloadVideo = () => {
    $("video").src = "/video?" + Date.now();
  };
  $("video").addEventListener("error", () => {
    $("video").removeAttribute("src");
  });
  bind("b-start", async () => { await post("/api/camera/start", {}); reloadVideo(); });
  bind("b-stop", () => post("/api/camera/stop"));
  bind("b-gon", () => post("/api/gestures/enable"));
  bind("b-goff", () => post("/api/gestures/disable"));
  bind("m-toggle", () => post("/api/media/play_pause"));
  bind("m-next", () => post("/api/media/next"));
  bind("m-prev", () => post("/api/media/previous"));
  bind("v-up", () => post("/api/volume/up"));
  bind("v-down", () => post("/api/volume/down"));
  bind("v-mute", () => post("/api/volume/mute"));
  let sliderT = null;
  $("v-slider").addEventListener("change", () => {
    clearTimeout(sliderT);
    sliderT = setTimeout(async () => {
      try { await post("/api/volume/set", {pct: Number($("v-slider").value)}); }
      catch (e) { toast(e.message); }
    }, 150);
  });
  for (const [id, out] of [["s-smooth", "s-smooth-v"], ["s-debounce", "s-debounce-v"], ["s-strength", "s-strength-v"]]) {
    $(id).addEventListener("input", () => { $(out).textContent = $(id).value; });
  }
  bind("s-apply", async () => {
    await api("/api/settings", {method: "PUT", body: JSON.stringify({
      smoothing: Number($("s-smooth").value) / 100,
      debounce: Number($("s-debounce").value),
      strength: Number($("s-strength").value),
      blur_kind: $("s-kind").value,
      privacy_mode: $("s-privacy").value,
      sensitivity: $("s-sens").value,
    })});
    toast("settings applied");
    refreshSettings();
  });
  document.querySelectorAll("#sidebar nav a").forEach((a) =>
    a.addEventListener("click", () => {
      document.querySelectorAll("#sidebar nav a").forEach((x) => x.classList.remove("active"));
      a.classList.add("active");
    }));
});
