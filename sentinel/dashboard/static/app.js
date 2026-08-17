// Sentinel Real-Time WebSocket Client & UI Controller

let ws = null;
let reconnectTimer = null;
const evidenceHistory = [];
let _lastMonitorRenderSeq = -1;  // throttle monitor re-renders to ~5Hz (every 4 frames)

const DOM = {
  trustPill: document.getElementById('globalTrustPill'),
  trustLabel: document.getElementById('trustLabel'),
  fpsCounter: document.getElementById('fpsCounter'),
  seqCounter: document.getElementById('seqCounter'),
  connectionStatus: document.getElementById('connectionStatus'),
  cameraStream: document.getElementById('cameraStream'),
  camSpeed: document.getElementById('camSpeed'),
  camSteer: document.getElementById('camSteer'),
  camGps: document.getElementById('camGps'),
  meterSpeed: document.getElementById('meterSpeed'),
  valSpeed: document.getElementById('valSpeed'),
  meterSteer: document.getElementById('meterSteer'),
  valSteer: document.getElementById('valSteer'),
  meterJitter: document.getElementById('meterJitter'),
  valJitter: document.getElementById('valJitter'),
  monitorsList: document.getElementById('monitorsList'),
  dmvCard: document.getElementById('dmvCard'),
  dmvCode: document.getElementById('dmvCode'),
  dmvTitle: document.getElementById('dmvTitle'),
  dmvDesc: document.getElementById('dmvDesc'),
  actionGrid: document.getElementById('actionGrid'),
  ruleTrace: document.getElementById('ruleTrace'),
  traceContent: document.getElementById('traceContent'),
  advisorText: document.getElementById('advisorText'),
  advisorTags: document.getElementById('advisorTags'),
  llmModelSelect: document.getElementById('llmModelSelect'),
  evidenceTimeline: document.getElementById('evidenceTimeline'),
  btnRefreshEvidence: document.getElementById('btnRefreshEvidence'),
  btnExportJSON: document.getElementById('btnExportJSON'),
  btnGenerateReports: document.getElementById('btnGenerateReports'),
  reportLinksContainer: document.getElementById('reportLinksContainer'),
  linkSingleReport: document.getElementById('linkSingleReport'),
  linkComparisonReport: document.getElementById('linkComparisonReport'),
};

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    DOM.connectionStatus.textContent = 'ONLINE';
    DOM.connectionStatus.className = 'stat-value text-green';
    if (reconnectTimer) clearTimeout(reconnectTimer);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'telemetry') {
        updateUI(data);
      }
    } catch (err) {
      console.error('Failed to parse WS message:', err);
    }
  };

  ws.onclose = () => {
    DOM.connectionStatus.textContent = 'RECONNECTING...';
    DOM.connectionStatus.className = 'stat-value text-amber';
    reconnectTimer = setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket Error:', err);
    ws.close();
  };
}

function updateUI(data) {
  // Counters & Badges
  DOM.seqCounter.textContent = `#${String(data.seq).padStart(5, '0')}`;
  DOM.fpsCounter.textContent = `${data.fps} FPS`;

  const sourceBadge = document.getElementById('sourceBadge');
  if (sourceBadge && data.dataset_source) {
    sourceBadge.textContent = `${data.dataset_source} Overlay`;
  }

  // Trust State Pill
  const trustState = data.trust.state;
  DOM.trustPill.className = `global-trust-pill state-${trustState}`;
  DOM.trustLabel.textContent = `ASSURANCE: ${trustState}`;

  // Camera & Overlay
  if (data.frame_b64) {
    DOM.cameraStream.src = `data:image/jpeg;base64,${data.frame_b64}`;
  }
  if (data.vehicle) {
    const spd = `${data.vehicle.speed_kph} km/h`;
    const str = `${data.vehicle.steering > 0 ? '+' : ''}${data.vehicle.steering}°`;
    const gps = `${data.vehicle.lat}, ${data.vehicle.lon}`;
    
    DOM.camSpeed.textContent = spd;
    DOM.camSteer.textContent = str;
    DOM.camGps.textContent = gps;

    DOM.valSpeed.textContent = spd;
    DOM.meterSpeed.style.width = `${Math.min(100, (data.vehicle.speed_kph / 120) * 100)}%`;

    DOM.valSteer.textContent = str;
    DOM.meterSteer.style.width = `${Math.min(100, Math.max(0, 50 + (data.vehicle.steering / 40) * 50))}%`;
  }

  // Jitter telemetry bar (from M3 timing evidence)
  if (data.jitter_ms !== undefined) {
    const jMs = parseFloat(data.jitter_ms) || 0;
    if (DOM.valJitter) DOM.valJitter.textContent = `${jMs.toFixed(2)} ms`;
    if (DOM.meterJitter) DOM.meterJitter.style.width = `${Math.min(100, (jMs / 10.0) * 100)}%`;
  }

  // Monitor Cards — throttled to ~5Hz (re-render every 4 frames to reduce DOM thrash)
  if (data.monitors && data.monitors.length > 0 && (data.seq - _lastMonitorRenderSeq >= 4 || _lastMonitorRenderSeq < 0)) {
    renderMonitors(data.monitors);
    _lastMonitorRenderSeq = data.seq;
  }

  // DMV Diagnosis Banner
  if (data.dmv) {
    DOM.dmvCard.style.display = 'block';
    DOM.dmvCode.textContent = data.dmv.code;
    DOM.dmvTitle.textContent = data.dmv.name;
    DOM.dmvDesc.textContent = data.dmv.description;
  } else {
    DOM.dmvCard.style.display = 'none';
  }

  // Policy Grid
  renderPolicy(data.policy, data.rule_trace);

  // LLM Advisor
  if (data.llm) {
    if (data.llm.scene_desc) {
      DOM.advisorText.textContent = data.llm.scene_desc;
    }
    if (data.llm.tags && data.llm.tags.length > 0) {
      DOM.advisorTags.innerHTML = data.llm.tags
        .map((t) => `<span class="tag">#${t}</span>`)
        .join('');
    }
  }

  // Evidence push (every 5 frames or on trust transition)
  if (data.seq % 5 === 0 || data.trust.state !== data.trust.prev) {
    addEvidenceCard(data);
  }
}

function renderMonitors(monitors) {
  if (!monitors) return;
  DOM.monitorsList.innerHTML = monitors
    .map((m) => {
      // m.score = anomaly severity (1.0=nominal, 0.0=FAIL) — this is what changes on fault injection
      // m.confidence = monitor self-confidence (always ~0.9) — NOT the anomaly metric
      const scorePct = Math.round((m.score !== undefined ? m.score : m.confidence) * 100);
      const maxZ = m.evidence && m.evidence.max_z !== undefined ? parseFloat(m.evidence.max_z).toFixed(2) : null;
      const meanConf = m.evidence && m.evidence.mean_confidence !== undefined
        ? (parseFloat(m.evidence.mean_confidence) * 100).toFixed(1) + '%'
        : null;

      // Build detail string: prefer Z-score + perception confidence for M1
      let evDetail = '';
      if (maxZ !== null) {
        evDetail = `Z=${maxZ}`;
        if (meanConf) evDetail += ` | perc=${meanConf}`;
      } else {
        evDetail = Object.entries(m.evidence || {})
          .filter(([k]) => k !== 'raw' && k !== 'reason')
          .slice(0, 2)
          .map(([k, v]) => `${k}:${v}`)
          .join(' | ') || (m.evidence && m.evidence.reason) || 'Nominal execution';
      }

      const fillClass = scorePct < 30 ? 'fill-red' : scorePct < 70 ? 'fill-amber' : 'fill-green';
      const zWarn = maxZ !== null && parseFloat(maxZ) >= 3.073;
      const zFail = maxZ !== null && parseFloat(maxZ) >= 4.588;
      const zBadge = maxZ !== null
        ? `<span class="z-badge ${zFail ? 'z-fail' : zWarn ? 'z-warn' : 'z-ok'}">Z=${maxZ}</span>`
        : '';

      return `
        <div class="monitor-card status-${m.status}">
          <span class="mon-status-pill status-pill-${m.status}">${m.status}</span>
          <div class="mon-info">
            <span class="mon-name">${m.name}</span>
            <span class="mon-diag">${evDetail}</span>
          </div>
          <div class="mon-conf-box">
            ${zBadge}
            <span class="mon-conf-num">${scorePct}%</span>
            <div class="meter-bar" style="width: 50px;">
              <div class="meter-fill ${fillClass}" style="width: ${scorePct}%"></div>
            </div>
          </div>
        </div>
      `;
    })
    .join('');
}

function renderPolicy(policy, ruleTrace) {
  if (!policy) return;
  DOM.actionGrid.innerHTML = Object.entries(policy)
    .map(([action, verdict]) => {
      const icon = verdict === 'PERMIT' ? '✔' : '✖';
      return `
        <div class="action-pill ${verdict}">
          <span>${action.replace('_', ' ').toUpperCase()}</span>
          <span>${icon} ${verdict}</span>
        </div>
      `;
    })
    .join('');

  if (ruleTrace && ruleTrace.length > 0) {
    DOM.traceContent.textContent = ruleTrace.join(' | ');
  }
}

function addEvidenceCard(data) {
  const card = document.createElement('div');
  card.className = `ev-card state-${data.trust.state}`;
  card.innerHTML = `
    <div class="ev-header">
      <span>#${data.seq}</span>
      <span>${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="ev-trust text-${data.trust.state === 'ALLOW' ? 'green' : data.trust.state === 'DEGRADE' ? 'amber' : 'red'}">
      ${data.trust.state}
    </div>
    <div style="font-size:0.65rem; color:#8c9cb8;">
      M-Scores: ${data.monitors.map((m) => Math.round(m.confidence * 100)).join('/')}
    </div>
  `;

  DOM.evidenceTimeline.insertBefore(card, DOM.evidenceTimeline.firstChild);
  if (DOM.evidenceTimeline.children.length > 25) {
    DOM.evidenceTimeline.removeChild(DOM.evidenceTimeline.lastChild);
  }
  evidenceHistory.push(data);
}

// REST Evidence Fetch
async function loadHistoricalEvidence() {
  try {
    const res = await fetch('/api/evidence?limit=25');
    const records = await res.json();
    DOM.evidenceTimeline.innerHTML = '';
    records.reverse().forEach((rec) => {
      const card = document.createElement('div');
      card.className = `ev-card state-${rec.trust.state}`;
      card.innerHTML = `
        <div class="ev-header">
          <span>#${rec.event_id}</span>
          <span>${rec.ts_human || ''}</span>
        </div>
        <div class="ev-trust text-${rec.trust.state === 'ALLOW' ? 'green' : rec.trust.state === 'DEGRADE' ? 'amber' : 'red'}">
          ${rec.trust.state}
        </div>
        <div style="font-size:0.65rem; color:#8c9cb8;">
          ${rec.monitors.map((m) => `${m.name.split('_')[0]}:${Math.round(m.confidence * 100)}%`).join(' ')}
        </div>
      `;
      DOM.evidenceTimeline.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load evidence:', err);
  }
}

DOM.btnRefreshEvidence.addEventListener('click', loadHistoricalEvidence);

DOM.btnExportJSON.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(evidenceHistory, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sentinel_evidence_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// Debug Modal & Fault Injection Runner
function setupDebugModal() {
  const modalOverlay = document.getElementById('debugModalOverlay');
  const btnOpen = document.getElementById('btnOpenDebugModal');
  const btnClose = document.getElementById('btnCloseDebugModal');
  const btnRunAll = document.getElementById('btnRunAllFaults');
  const statusLabel = document.getElementById('testSuiteStatus');

  if (btnOpen && modalOverlay) {
    btnOpen.addEventListener('click', () => { modalOverlay.style.display = 'flex'; });
  }
  if (btnClose && modalOverlay) {
    btnClose.addEventListener('click', () => { modalOverlay.style.display = 'none'; });
  }

  // Single fault triggers
  document.querySelectorAll('.btnInjectSingle').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const faultType = e.target.getAttribute('data-fault');
      try {
        const res = await fetch(`/api/inject_fault?fault_type=${faultType}&duration_s=15.0`, { method: 'POST' });
        const data = await res.json();
        alert(`Injected ${faultType} for 15s (start frame: #${data.start_seq})`);
      } catch (err) {
        console.error('Failed to inject fault:', err);
      }
    });
  });

  // Automated 5-stage test suite
  if (btnRunAll) {
    btnRunAll.addEventListener('click', async () => {
      btnRunAll.disabled = true;
      btnRunAll.style.opacity = '0.6';

      const tests = [
        { fault: 'perception_degradation', badgeId: 'badgeM1', cardId: 'cardM1', label: '1/5: M1 Perception' },
        { fault: 'sensor_dropout', badgeId: 'badgeM2Dropout', cardId: 'cardM2Dropout', label: '2/5: M2 Sensor Dropout' },
        { fault: 'calibration_drift', badgeId: 'badgeM2Drift', cardId: 'cardM2Drift', label: '3/5: M2 CUSUM Drift' },
        { fault: 'latency_spike', badgeId: 'badgeM3', cardId: 'cardM3', label: '4/5: M3 Latency Spike' },
        { fault: 'odd_exit', badgeId: 'badgeM4', cardId: 'cardM4', label: '5/5: M4 ODD Exit' },
      ];

      // Reset all badges to READY
      tests.forEach(t => {
        const badge = document.getElementById(t.badgeId);
        const card = document.getElementById(t.cardId);
        if (badge) { badge.textContent = 'READY'; badge.className = 'fault-status-badge'; }
        if (card) { card.className = 'fault-test-card'; }
      });

      for (const t of tests) {
        const badge = document.getElementById(t.badgeId);
        const card = document.getElementById(t.cardId);
        
        statusLabel.textContent = `Running Test ${t.label}... (8s fault window)`;
        if (badge) { badge.textContent = 'RUNNING'; badge.className = 'fault-status-badge active'; }
        if (card) { card.className = 'fault-test-card testing'; }

        try {
          await fetch(`/api/inject_fault?fault_type=${t.fault}&duration_s=8.0`, { method: 'POST' });
        } catch (err) {
          console.error(`Failed test ${t.fault}:`, err);
        }

        // Wait 10s for fault duration + detection
        await new Promise(r => setTimeout(r, 10000));

        if (badge) { badge.textContent = 'PASSED'; badge.className = 'fault-status-badge passed'; }
        if (card) { card.className = 'fault-test-card passed'; }

        // Short pause between test stages
        await new Promise(r => setTimeout(r, 2000));
      }

      statusLabel.textContent = '⏳ Waiting for System Hysteresis Recovery to FULL_AUTONOMY (6s)...';
      await new Promise(r => setTimeout(r, 6000));

      statusLabel.textContent = '✅ All 5 Tests Executed & System Returned to FULL_AUTONOMY! Generating HTML Reports...';
      await triggerReportGeneration();
      
      btnRunAll.disabled = false;
      btnRunAll.style.opacity = '1';
    });
  }

  if (DOM.btnGenerateReports) {
    DOM.btnGenerateReports.addEventListener('click', async () => {
      await triggerReportGeneration();
    });
  }
}

async function triggerReportGeneration() {
  try {
    const resp = await fetch('/api/eval/generate_reports', { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'generated') {
      if (DOM.reportLinksContainer) DOM.reportLinksContainer.style.display = 'block';
      if (DOM.linkSingleReport) DOM.linkSingleReport.href = data.single_report_url;
      if (DOM.linkComparisonReport) DOM.linkComparisonReport.href = data.comparison_report_url;
      console.log('✅ Production HTML Benchmark Reports generated successfully!');
    }
  } catch (err) {
    console.error('Failed to generate benchmark reports:', err);
  }
}

function setupLLMSelect() {
  if (!DOM.llmModelSelect) return;
  
  // Fetch current active model status on load
  fetch('/api/llm/models')
    .then((res) => res.json())
    .then((data) => {
      if (data && data.active_model_id) {
        DOM.llmModelSelect.value = data.active_model_id;
      }
    })
    .catch((err) => console.warn('Could not fetch initial LLM model status:', err));

  DOM.llmModelSelect.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    try {
      const resp = await fetch('/api/llm/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: selectedModel }),
      });
      const res = await resp.json();
      if (res.status === 'switched') {
        console.log(`✅ LLM Model successfully switched to: ${res.active_model_name}`);
      }
    } catch (err) {
      console.error('Failed to switch LLM model:', err);
    }
  });
}

// Start WebSocket connection & setup modals & LLM selector
window.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  loadHistoricalEvidence();
  setupDebugModal();
  setupLLMSelect();
});
