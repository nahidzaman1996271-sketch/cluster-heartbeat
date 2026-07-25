(() => {
  "use strict";

  // Same-origin by default (served by the FastAPI app itself). Falls back
  // to localhost:8000 if this file is opened directly from disk (file://).
  const API_BASE = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
  const POLL_MS = 15000;

  let state = {
    dashboard: null,
    expandedNode: null,
    activeSeverity: null, // null = all
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function fmt(n, d = 1) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toFixed(d);
  }

  function riskColor(risk) {
    if (risk > 0.6) return "var(--critical)";
    if (risk > 0.3) return "var(--warning)";
    return "var(--healthy)";
  }

  function healthColor(pct) {
    if (pct < 50) return "var(--critical)";
    if (pct < 80) return "var(--warning)";
    return "var(--healthy)";
  }

  function sevColor(sev) {
    return { CRITICAL: "var(--critical)", HIGH: "var(--warning)", MEDIUM: "var(--brand)", LOW: "var(--healthy)" }[sev] || "var(--text-dim)";
  }

  function timeAgo(iso) {
    if (!iso) return "—";
    const t = new Date(iso.replace(" ", "T") + (iso.includes("Z") || iso.includes("+") ? "" : "Z"));
    if (Number.isNaN(t.getTime())) return iso;
    const diffMs = Date.now() - t.getTime();
    const mins = Math.round(diffMs / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 48) return `${hrs}h ago`;
    return `${Math.round(hrs / 24)}d ago`;
  }

  // ---------------- connection state ----------------
  function setConn(ok, msg) {
    const dot = $("#connDot");
    const txt = $("#connText");
    dot.className = "conn-dot " + (ok ? "on" : "off");
    txt.textContent = msg || (ok ? "live" : "offline");
  }

  function showError(msg) {
    const el = $("#errorBanner");
    if (!msg) { el.classList.remove("show"); return; }
    el.textContent = "⚠ " + msg;
    el.classList.add("show");
  }

  // ---------------- fetch ----------------
  async function fetchJSON(path) {
    const res = await fetch(API_BASE + path, { headers: { accept: "application/json" } });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} on ${path} ${body ? "— " + body.slice(0, 140) : ""}`);
    }
    return res.json();
  }

  async function loadDashboard() {
    const icon = $("#refreshIcon");
    icon.classList.add("spin");
    try {
      const dash = await fetchJSON("/api/v1/dashboard");
      state.dashboard = dash;
      setConn(true, "live");
      showError(null);
      render();
    } catch (err) {
      setConn(false, "offline");
      showError(
        `Can't reach the Cluster Heartbeat API${API_BASE ? " at " + API_BASE : ""}. ` +
        `Make sure it's running (uvicorn cluster_heartbeat.api.main:app) and reachable. (${err.message})`
      );
    } finally {
      icon.classList.remove("spin");
      $("#loadingOverlay").style.opacity = "0";
      setTimeout(() => $("#loadingOverlay").style.display = "none", 300);
    }
  }

  // ---------------- sparkline ----------------
  function sparklineSVG(points, color) {
    if (!points || points.length < 2) return "";
    const vals = points.map(p => p.value);
    const min = Math.min(...vals), max = Math.max(...vals);
    const range = max - min || 1;
    const w = 1000, h = 54, pad = 4;
    const stepX = (w - pad * 2) / (points.length - 1);
    const coords = vals.map((v, i) => {
      const x = pad + i * stepX;
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      return [x, y];
    });
    const linePath = coords.map((c, i) => (i === 0 ? "M" : "L") + c[0].toFixed(1) + "," + c[1].toFixed(1)).join(" ");
    const areaPath = linePath + ` L${coords[coords.length - 1][0].toFixed(1)},${h} L${coords[0][0].toFixed(1)},${h} Z`;
    const last = coords[coords.length - 1];
    return `
      <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${color}" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path d="${areaPath}" fill="url(#sparkfill)" stroke="none"/>
        <path d="${linePath}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
        <circle cx="${last[0]}" cy="${last[1]}" r="3" fill="${color}"/>
      </svg>`;
  }

  // ---------------- hero ----------------
  function renderHero() {
    const d = state.dashboard;
    const c = d.cluster;
    const score = c.health_score;
    const color = healthColor(score);

    $("#clusterModelTag").textContent = `cluster heartbeat api · backend ${d.model?.backend ?? "—"} · v${d.model?.version ?? "—"}`;
    $("#genAtPill").textContent = "gen " + timeAgo(d.generated_at);

    const spark = sparklineSVG(d.timeseries?.cluster_health, color);

    $("#heroSection").innerHTML = `
      <div class="hero-score">
        <div class="eyebrow">Cluster Health</div>
        <div class="score-figure">
          <span class="num" style="color:${color}">${fmt(score, 0)}</span>
          <span class="unit">/ 100</span>
        </div>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${Math.max(2, score)}%; background:${color}"></div></div>
        <div class="score-caption">${c.nodes_healthy} of ${c.nodes_total} nodes healthy</div>
      </div>
      <div class="hero-stats">
        <div class="stat-cell">
          <span class="label">Nodes at risk</span>
          <span class="value ${c.nodes_at_risk > 0 ? "warn" : "ok"}">${c.nodes_at_risk}</span>
          <span class="subtext">of ${c.nodes_total} total</span>
        </div>
        <div class="stat-cell">
          <span class="label">Active alerts</span>
          <span class="value ${c.active_alerts > 0 ? "warn" : "ok"}">${c.active_alerts}</span>
          <span class="subtext">${c.critical_alerts} critical</span>
        </div>
        <div class="stat-cell">
          <span class="label">Avg GPU util</span>
          <span class="value">${fmt(c.avg_gpu_utilization_pct, 1)}<span style="font-size:15px">%</span></span>
          <span class="subtext">across cluster</span>
        </div>
        <div class="stat-cell">
          <span class="label">Reclaimable</span>
          <span class="value ok">$${fmt(c.estimated_reclaimable_usd, 2)}</span>
          <span class="subtext">per day, est.</span>
        </div>
        <div class="hero-spark">
          <div class="label">Cluster health — trailing window</div>
          ${spark || '<div style="color:var(--text-dim); font-family:var(--font-mono); font-size:11px;">not enough history yet</div>'}
        </div>
      </div>
    `;
  }

  // ---------------- rack ----------------
  function renderRack() {
    const nodes = state.dashboard.nodes || {};
    const ids = Object.keys(nodes).sort();
    $("#rackCount").textContent = `${ids.length} nodes`;

    const rack = $("#rackGrid");
    rack.innerHTML = "";

    ids.forEach(id => {
      const n = nodes[id];
      const risk = n.failure_risk ?? 0;
      const color = riskColor(risk);
      const cls = n.classification?.label ?? "unknown";
      const pulseDur = Math.max(0.8, 2.8 - risk * 2); // faster pulse = higher risk
      const ttf = n.ttf_hours;

      const row = document.createElement("div");
      row.className = "blade";
      row.dataset.node = id;
      row.innerHTML = `
        <div class="node-id">${id}<span class="sub">${n.embedding ? n.embedding.length + "d fp" : ""}</span></div>
        <div class="heartbeat-dot pulse" style="background:${color}; color:${color}; animation:heartbeatPulse ${pulseDur}s ease-in-out infinite;"></div>
        <div class="class-tag">${cls}</div>
        <div class="health-bar-wrap">
          <div class="health-bar-track"><div class="health-bar-fill" style="width:${Math.max(2, n.gpu_health)}%; background:${healthColor(n.gpu_health)}"></div></div>
          <span class="pct">${fmt(n.gpu_health, 0)}%</span>
        </div>
        <div class="risk-val" style="color:${color}">${fmt(risk, 2)}</div>
        <div class="ttf-val" style="color:${ttf < 4 ? "var(--critical)" : "var(--text-mid)"}">${ttf === null || ttf === undefined ? "—" : fmt(ttf, 1) + "h"}</div>
        <svg class="chevron" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6 3l6 5-6 5"/></svg>
      `;
      row.addEventListener("click", () => {
        state.expandedNode = state.expandedNode === id ? null : id;
        renderRack();
      });
      rack.appendChild(row);

      const detailWrap = document.createElement("div");
      detailWrap.className = "blade-detail" + (state.expandedNode === id ? " open" : "");
      if (state.expandedNode === id) {
        row.classList.add("expanded");
        const dp = n.demand_prediction || {};
        const probs = n.classification?.probabilities || {};
        const reasons = n.reasons || [];
        detailWrap.innerHTML = `
          <div class="blade-detail-inner">
            <div class="detail-block">
              <div class="label">Why</div>
              <ul>${reasons.length ? reasons.map(r => `<li>${escapeHTML(r)}</li>`).join("") : "<li>no explanatory signals recorded</li>"}</ul>
            </div>
            <div class="detail-block">
              <div class="label">Predicted demand</div>
              <div class="demand-row"><span>GPU utilization</span><span>${fmt(dp.gpu_utilization, 1)}%</span></div>
              <div class="demand-row"><span>Memory utilization</span><span>${fmt(dp.memory_utilization, 1)}%</span></div>
              <div class="demand-row"><span>Power draw</span><span>${fmt(dp.power_consumption, 1)}W</span></div>
            </div>
            <div class="detail-block">
              <div class="label">Workload classification</div>
              ${Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([k, v]) => `
                <div class="prob-row">
                  <span class="prob-label">${k}</span>
                  <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
                  <span style="font-family:var(--font-mono); color:var(--text-mid); width:34px; text-align:right;">${(v * 100).toFixed(0)}%</span>
                </div>
              `).join("")}
            </div>
          </div>
        `;
      }
      rack.appendChild(detailWrap);
    });
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------- alerts ----------------
  function renderAlertsToolbar() {
    const sevs = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
    const counts = {};
    (state.dashboard.alerts || []).forEach(a => counts[a.severity] = (counts[a.severity] || 0) + 1);

    const bar = $("#alertsToolbar");
    bar.innerHTML = "";
    const allBtn = document.createElement("button");
    allBtn.className = "sev-filter" + (state.activeSeverity === null ? " active" : "");
    allBtn.textContent = `All (${(state.dashboard.alerts || []).length})`;
    allBtn.addEventListener("click", () => { state.activeSeverity = null; renderAlerts(); renderAlertsToolbar(); });
    bar.appendChild(allBtn);

    sevs.forEach(sev => {
      const btn = document.createElement("button");
      btn.className = "sev-filter" + (state.activeSeverity === sev ? " active" : "");
      btn.dataset.sev = sev;
      btn.textContent = `${sev} (${counts[sev] || 0})`;
      btn.addEventListener("click", () => { state.activeSeverity = sev; renderAlerts(); renderAlertsToolbar(); });
      bar.appendChild(btn);
    });
  }

  function renderAlerts() {
    const all = state.dashboard.alerts || [];
    const items = state.activeSeverity ? all.filter(a => a.severity === state.activeSeverity) : all;
    $("#alertsCount").textContent = `${all.length} total`;

    const list = $("#alertsList");
    if (!items.length) {
      list.innerHTML = `<div class="empty-state">no alerts in this filter — cluster is quiet</div>`;
      return;
    }
    const sorted = [...items].sort((a, b) => (a.ts < b.ts ? 1 : -1));
    list.innerHTML = sorted.map(a => `
      <div class="alert-row">
        <div class="alert-bar" style="background:${sevColor(a.severity)}"></div>
        <div class="alert-sev" style="color:${sevColor(a.severity)}">${a.severity}<br><span style="color:var(--text-dim); font-weight:400;">${a.type}</span></div>
        <div class="alert-body">
          <div class="msg">${escapeHTML(a.message)}</div>
          <div class="action"><b>Action —</b> ${escapeHTML(a.recommended_action)}</div>
        </div>
        <div class="alert-meta">
          <span class="node-link">${a.node_id}</span>
          <span>${timeAgo(a.ts)}</span>
        </div>
      </div>
    `).join("");
  }

  // ---------------- cost ----------------
  function renderCost() {
    const rec = state.dashboard.recommendations?.cost;
    const box = $("#costList");
    if (!rec || !rec.idle_or_wasteful_nodes?.length) {
      box.innerHTML = `<div class="empty-state">no wasteful allocations detected</div>`;
      return;
    }
    box.innerHTML = rec.idle_or_wasteful_nodes.map(n => `
      <div class="cost-row">
        <div>
          <div class="cid">${n.node_id}</div>
          <div class="ckind">${n.kind.replace(/_/g, " ")} · ${fmt(n.idle_hours, 1)}h idle · ${fmt(n.mean_gpu_util_pct, 1)}% util</div>
        </div>
        <div class="cval">$${fmt(n.estimated_savings_usd_per_day, 2)}<span class="sub">per day</span></div>
      </div>
    `).join("") + `
      <div class="cost-total">
        <span class="tlabel">Total reclaimable / day</span>
        <span class="tval">$${fmt(rec.estimated_reclaimable_usd, 2)}</span>
      </div>
    `;
  }

  // ---------------- scheduling ----------------
  function renderSchedList(items) {
    const box = $("#schedList");
    if (!items || !items.length) {
      box.innerHTML = `<div class="empty-state">no eligible nodes for this request</div>`;
      return;
    }
    box.innerHTML = items.slice(0, 8).map((n, i) => `
      <div class="sched-row">
        <span class="sched-rank">${String(i + 1).padStart(2, "0")}</span>
        <div class="sched-node">${n.node_id}<span class="sub">${n.complementary_placement || n.current_class || ""}</span></div>
        <span class="sched-score">${fmt(n.score, 2)}</span>
      </div>
    `).join("");
  }

  async function runSchedulingQuery() {
    const gpu = Number($("#gpuReq").value) || 0;
    const mem = Number($("#memReq").value) || 0;
    try {
      const res = await fetchJSON(`/api/v1/recommendations/scheduling?gpu_request=${gpu}&mem_request=${mem}`);
      renderSchedList(res.recommended);
    } catch (err) {
      $("#schedList").innerHTML = `<div class="empty-state">couldn't fetch a ranking (${escapeHTML(err.message)})</div>`;
    }
  }

  // ---------------- explainer ----------------
  async function loadExplainerData() {
    try {
      console.log("Loading explainer data from /api/v1/explainer/dashboard...");
      const data = await fetchJSON("/api/v1/explainer/dashboard");
      console.log("✓ Explainer data received:", data);
      renderExplainer(data);
    } catch (err) {
      console.error("❌ Explainer endpoint failed:", err.message);
      console.log("Trying fallback /api/v1/explainer/cluster endpoint...");
      try {
        const data = await fetchJSON("/api/v1/explainer/cluster");
        console.log("✓ Cluster explainer data received (legacy):", data);
        renderExplainerLegacy(data);
      } catch (err2) {
        console.warn("Both explainer endpoints unavailable:", err2.message);
        $("#explainerSection").style.display = "none";
        $("#suggestedActionsPanel").style.display = "none";
      }
    }
  }

  function renderExplainerLegacy(data) {
    const explainerSec = $("#explainerSection");
    const actionsSec = $("#suggestedActionsPanel");
    
    if (!data) {
      explainerSec.style.display = "none";
      actionsSec.style.display = "none";
      return;
    }
    
    if (data.anomalies && data.anomalies.length > 0) {
      explainerSec.style.display = "none";
      actionsSec.style.display = "block";
      const actionsBox = $("#suggestedActionsList");
      actionsBox.innerHTML = `<div class="empty-state">Explainer shape/movers unavailable. Check browser console for errors.</div>`;
    }
  }

  function renderExplainer(data) {
    const explainerSec = $("#explainerSection");
    const actionsSec = $("#suggestedActionsPanel");
    
    if (!data || !data.shape_view) {
      explainerSec.style.display = "none";
      actionsSec.style.display = "none";
      return;
    }
    
    explainerSec.style.display = "grid";
    actionsSec.style.display = "block";

    // Shape view
    const shapeBox = $("#shapeViewList");
    const shapes = (data.shape_view?.shapes || []).slice(0, 5);
    shapeBox.innerHTML = shapes.map(s => `
      <div class="metric-row" style="margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--line-soft);">
        <div style="font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:6px;">${s.metric}</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-family:var(--font-mono); font-size:11px;">
          <div>
            <div style="color:var(--text-dim); margin-bottom:4px;">Healthy</div>
            <div style="font-size:16px; font-weight:600; color:var(--healthy);">${fmt(s.healthy, 1)}<span style="font-size:11px; margin-left:4px;">${s.unit}</span></div>
          </div>
          <div>
            <div style="color:var(--text-dim); margin-bottom:4px;">Current</div>
            <div style="font-size:16px; font-weight:600; color:${s.change_percent > 10 ? "var(--critical)" : s.change_percent > 5 ? "var(--warning)" : "var(--text-hi)"};">${fmt(s.current, 1)}<span style="font-size:11px; margin-left:4px;">${s.unit}</span></div>
            <div style="font-size:10px; color:var(--text-dim); margin-top:2px;">${s.change_percent > 0 ? "↑" : "↓"} ${Math.abs(s.change_percent).toFixed(0)}%</div>
          </div>
        </div>
      </div>
    `).join("");

    // Key movers
    const moversBox = $("#keyMoversList");
    const movers = (data.key_movers?.top_movers || []).slice(0, 5);
    moversBox.innerHTML = movers.map(m => `
      <div class="mover-row" style="margin-bottom:12px; padding:10px; background:var(--surface-raised); border-radius:var(--radius-sm); border-left:3px solid ${m.change_percent > 0 ? "var(--critical)" : "var(--healthy)"};">
        <div style="font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:4px;">${m.direction} ${m.metric}</div>
        <div style="font-size:11px; color:var(--text-mid); margin-bottom:6px;">Healthy: ${fmt(m.healthy_value, 1)} → Current: ${fmt(m.current_value, 1)}</div>
        <div style="font-size:10px; color:var(--text-dim);">Impact: ${fmt(m.impact_score, 1)}% | ${m.reason}</div>
      </div>
    `).join("") || `<div class="empty-state" style="font-size:12px;">No significant metric changes detected.</div>`;

    // Suggested actions
    const actionsBox = $("#suggestedActionsList");
    const actions = (data.suggested_actions || []);
    actionsBox.innerHTML = actions.map(a => {
      const bgColor = a.priority === 1 ? "var(--critical-dim)" : a.priority === 2 ? "var(--warning-dim)" : "var(--surface-raised)";
      const borderColor = a.priority === 1 ? "var(--critical)" : a.priority === 2 ? "var(--warning)" : "var(--line)";
      return `
        <div style="margin-bottom:12px; padding:12px; background:${bgColor}; border:1px solid ${borderColor}; border-radius:var(--radius-sm);">
          <div style="font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:4px;">
            P${a.priority} · ${a.action}
          </div>
          <div style="font-size:11px; color:var(--text-mid);">${a.reason}</div>
          ${a.affected_nodes?.length ? `<div style="font-size:10px; color:var(--text-dim); margin-top:6px;">Nodes: ${a.affected_nodes.slice(0,3).join(", ")}${a.affected_nodes.length > 3 ? " +" + (a.affected_nodes.length - 3) : ""}</div>` : ""}
          ${a.estimated_ttf_hours ? `<div style="font-size:10px; color:var(--text-dim);">TTF: ~${fmt(a.estimated_ttf_hours, 1)}h</div>` : ""}
        </div>
      `;
    }).join("");
  }

  // ---------------- master render ----------------
  function render() {
    renderHero();
    renderRack();
    renderAlertsToolbar();
    renderAlerts();
    renderCost();
    loadExplainerData();
    const initialSched = state.dashboard.recommendations?.scheduling?.recommended;
    if (initialSched) renderSchedList(initialSched);
  }

  // inject keyframes for heartbeat pulse (variable duration set per-node via inline style)
  const styleTag = document.createElement("style");
  styleTag.textContent = `
    @keyframes heartbeatPulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      15% { transform: scale(1.35); opacity: 1; }
      30% { transform: scale(0.9); opacity: 0.85; }
      45% { transform: scale(1.15); opacity: 1; }
      60% { transform: scale(1); opacity: 1; }
    }
  `;
  document.head.appendChild(styleTag);

  // ---------------- wiring ----------------
  document.addEventListener("DOMContentLoaded", () => {
    $("#refreshBtn").addEventListener("click", loadDashboard);
    $("#schedBtn").addEventListener("click", runSchedulingQuery);
    loadDashboard();
    setInterval(loadDashboard, POLL_MS);
  });
})();
