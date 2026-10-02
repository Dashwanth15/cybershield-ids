document.addEventListener('DOMContentLoaded', () => {
  // State
  let metadata = null;
  let presets = {};
  let currentLoadedPreset = null;
  let optimalThreshold = 0.50;
  let streamRecords = [];
  let streamIndex = 0;
  let streamTimer = null;
  let streamStats = { total: 0, normal: 0, attack: 0, correct: 0 };
  let batchData = null;
  let whatifOriginalRecord = null;

  // DOM Elements
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const presetSelect = document.getElementById('presetSelect');
  const btnLoadPreset = document.getElementById('btnLoadPreset');
  const presetInfoBanner = document.getElementById('presetInfoBanner');
  const trafficForm = document.getElementById('trafficForm');
  const fieldService = document.getElementById('field_service');
  const predictionResultCard = document.getElementById('predictionResultCard');
  const headerThreshold = document.getElementById('headerThreshold');

  // Initialize
  initTabs();
  loadResearchEvolution();
  loadMetadataAndPresets();
  loadPerformanceMetrics();
  loadNovelAttacks();
  initWhatIfLab();
  initLiveStream();
  initBatchPrediction();

  // Tab Navigation
  function initTabs() {
    navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        navTabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        const targetId = tab.getAttribute('data-tab');
        const targetContent = document.getElementById(targetId);
        if (targetContent) {
          targetContent.classList.add('active');
        }
      });
    });
  }

  // Load Metadata & Presets
  async function loadMetadataAndPresets() {
    try {
      const metaRes = await fetch('/api/metadata');
      metadata = await metaRes.json();
      optimalThreshold = metadata.optimal_threshold || 0.50;
      headerThreshold.textContent = `τ = ${optimalThreshold.toFixed(2)}`;

      // Populate service dropdown
      if (metadata.features && metadata.features.service && metadata.features.service.options) {
        fieldService.innerHTML = '';
        metadata.features.service.options.forEach(svc => {
          const opt = document.createElement('option');
          opt.value = svc;
          opt.textContent = svc;
          if (svc === 'http') opt.selected = true;
          fieldService.appendChild(opt);
        });
      }

      // Load presets
      const presetRes = await fetch('/api/presets');
      presets = await presetRes.json();
      presetSelect.innerHTML = '<option value="">-- Select a Real Test Connection --</option>';
      for (const name in presets) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = `${name} (${presets[name].attack_type || 'normal'})`;
        presetSelect.appendChild(opt);
      }
    } catch (e) {
      console.error('Error loading metadata/presets:', e);
    }
  }

  // Preset Loading
  btnLoadPreset.addEventListener('click', () => {
    const selectedName = presetSelect.value;
    if (!selectedName || !presets[selectedName]) return;

    const record = presets[selectedName];
    currentLoadedPreset = record;
    populateFormWithRecord(record);

    // Show info banner
    presetInfoBanner.classList.remove('hidden');
    document.getElementById('presetLabelBadge').textContent = `Label: ${record.attack_type === 'normal' ? 'Normal' : 'Attack'}`;
    document.getElementById('presetCategoryBadge').textContent = `Category: ${record.attack_category || 'normal'}`;
    document.getElementById('presetAttackBadge').textContent = `Attack Type: ${record.attack_type || 'normal'}`;
  });

  function populateFormWithRecord(record) {
    for (const key in record) {
      const field = document.getElementById(`field_${key}`);
      if (field) {
        field.value = record[key];
      }
    }
  }

  // Form Submission / Single Prediction
  trafficForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(trafficForm);
    const payload = {};

    formData.forEach((value, key) => {
      payload[key] = value;
    });

    const btnAnalyze = document.getElementById('btnAnalyze');
    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = '<span>⏳</span> Analyzing with XGBoost...';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const result = await res.json();
      if (result.error) {
        alert('Analysis Error: ' + result.error);
        return;
      }

      displayPredictionResult(result);
    } catch (err) {
      alert('Failed to analyze traffic: ' + err);
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<span class="btn-icon">⚡</span> Analyze Traffic with XGBoost';
    }
  });

  function displayPredictionResult(result) {
    predictionResultCard.classList.remove('hidden');
    const isAttack = result.is_attack;

    // Card styling
    if (isAttack) {
      predictionResultCard.classList.add('attack');
      predictionResultCard.classList.remove('normal');
    } else {
      predictionResultCard.classList.add('normal');
      predictionResultCard.classList.remove('attack');
    }

    // Header Status Badge
    const resStatusBadge = document.getElementById('resStatusBadge');
    resStatusBadge.textContent = result.status;
    resStatusBadge.className = `status-badge ${isAttack ? 'attack' : 'normal'}`;

    // Metrics
    const resPred = document.getElementById('resPrediction');
    resPred.textContent = result.prediction;
    resPred.className = `res-value ${isAttack ? 'attack-text' : 'normal-text'}`;

    document.getElementById('resAttackProb').textContent = `${result.attack_probability}%`;
    document.getElementById('resConfidence').textContent = `${result.confidence}%`;
    document.getElementById('resThreshold').textContent = result.threshold_used;

    // Ground Truth Verification
    const verBox = document.getElementById('presetVerificationBox');
    if (currentLoadedPreset) {
      verBox.classList.remove('hidden');
      document.getElementById('verActualLabel').textContent = currentLoadedPreset.attack_type;
      document.getElementById('verAttackCategory').textContent = currentLoadedPreset.attack_category || 'normal';
      
      const isActualAttack = currentLoadedPreset.attack_type !== 'normal';
      const isCorrect = isAttack === isActualAttack;
      const verStatus = document.getElementById('verStatus');
      if (isCorrect) {
        verStatus.textContent = '✓ Prediction Matches Ground Truth';
        verStatus.className = 'ver-correct';
      } else {
        verStatus.textContent = '✗ Ground Truth Mismatch (Generalization Limit)';
        verStatus.className = 'ver-incorrect';
      }
    } else {
      verBox.classList.add('hidden');
    }

    // SHAP Explainability
    const attackList = document.getElementById('shapAttackList');
    const normalList = document.getElementById('shapNormalList');
    attackList.innerHTML = '';
    normalList.innerHTML = '';

    const expl = result.explanation;
    if (expl.attack_contributors && expl.attack_contributors.length > 0) {
      expl.attack_contributors.forEach(item => {
        const li = document.createElement('li');
        li.className = 'attack-contributor';
        li.innerHTML = `<strong>${item.feature} = ${item.feature_value}</strong><br><span style="color: var(--accent-red);">+${item.shap_value} toward Attack</span> &mdash; ${item.explanation}`;
        attackList.appendChild(li);
      });
    } else {
      attackList.innerHTML = '<li class="empty-notice">No significant attack contributors.</li>';
    }

    if (expl.normal_contributors && expl.normal_contributors.length > 0) {
      expl.normal_contributors.forEach(item => {
        const li = document.createElement('li');
        li.className = 'normal-contributor';
        li.innerHTML = `<strong>${item.feature} = ${item.feature_value}</strong><br><span style="color: var(--accent-green);">${item.shap_value} toward Normal</span> &mdash; ${item.explanation}`;
        normalList.appendChild(li);
      });
    } else {
      normalList.innerHTML = '<li class="empty-notice">No significant normal contributors.</li>';
    }

    // Scroll to results
    predictionResultCard.scrollIntoView({ behavior: 'smooth' });
  }

  // Reset form
  document.getElementById('btnResetForm').addEventListener('click', () => {
    trafficForm.reset();
    currentLoadedPreset = null;
    presetInfoBanner.classList.add('hidden');
    predictionResultCard.classList.add('hidden');
  });

  // ========================================================
  // "WHAT-IF?" SENSITIVITY LAB
  // ========================================================
  function initWhatIfLab() {
    const btnWhatifNormal = document.getElementById('btnWhatifLoadNormal');
    const btnWhatifNeptune = document.getElementById('btnWhatifLoadNeptune');
    const btnWhatifProbe = document.getElementById('btnWhatifLoadProbe');
    const btnRunWhatIf = document.getElementById('btnRunWhatIf');

    btnWhatifNormal.addEventListener('click', () => loadWhatifPreset('Normal HTTP Traffic'));
    btnWhatifNeptune.addEventListener('click', () => loadWhatifPreset('DoS / Neptune Attack'));
    btnWhatifProbe.addEventListener('click', () => loadWhatifPreset('Probe / Portscan Attack'));

    btnRunWhatIf.addEventListener('click', async () => {
      if (!whatifOriginalRecord) {
        loadWhatifPreset('Normal HTTP Traffic');
      }
      
      // Clone original and apply modified values
      const modifiedRecord = { ...whatifOriginalRecord };
      modifiedRecord.flag = document.getElementById('whatif_flag').value;
      modifiedRecord.serror_rate = parseFloat(document.getElementById('whatif_serror_rate').value);
      modifiedRecord.dst_host_serror_rate = parseFloat(document.getElementById('whatif_dst_host_serror_rate').value);
      modifiedRecord.count = parseInt(document.getElementById('whatif_count').value);
      modifiedRecord.logged_in = parseInt(document.getElementById('whatif_logged_in').value);
      modifiedRecord.src_bytes = parseInt(document.getElementById('whatif_src_bytes').value);

      // Run prediction on modified record
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(modifiedRecord)
        });
        const afterResult = await res.json();
        displayWhatIfComparison(afterResult, modifiedRecord);
      } catch (err) {
        alert('Error evaluating What-If: ' + err);
      }
    });
  }

  async function loadWhatifPreset(presetName) {
    if (!presets[presetName]) return;
    whatifOriginalRecord = { ...presets[presetName] };

    // Update controls to match
    document.getElementById('whatif_flag').value = whatifOriginalRecord.flag || 'SF';
    document.getElementById('whatif_serror_rate').value = whatifOriginalRecord.serror_rate || 0.0;
    document.getElementById('whatif_dst_host_serror_rate').value = whatifOriginalRecord.dst_host_serror_rate || 0.0;
    document.getElementById('whatif_count').value = whatifOriginalRecord.count || 10;
    document.getElementById('whatif_logged_in').value = whatifOriginalRecord.logged_in || 1;
    document.getElementById('whatif_src_bytes').value = whatifOriginalRecord.src_bytes || 300;

    // Run baseline prediction
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(whatifOriginalRecord)
    });
    const beforeResult = await res.json();
    whatifOriginalRecord._beforeResult = beforeResult;

    // Display before card
    document.getElementById('whatifBeforePred').textContent = beforeResult.prediction;
    document.getElementById('whatifBeforePred').className = beforeResult.is_attack ? 'attack-text' : 'normal-text';
    document.getElementById('whatifBeforeProb').textContent = `${beforeResult.attack_probability}%`;
    document.getElementById('whatifBeforeDetails').innerHTML = `
      <div><strong>Flag:</strong> ${whatifOriginalRecord.flag}</div>
      <div><strong>serror_rate:</strong> ${whatifOriginalRecord.serror_rate}</div>
      <div><strong>count:</strong> ${whatifOriginalRecord.count}</div>
      <div><strong>logged_in:</strong> ${whatifOriginalRecord.logged_in}</div>
    `;

    document.getElementById('whatifResultsCard').classList.remove('hidden');
  }

  function displayWhatIfComparison(afterResult, modifiedRecord) {
    const beforeResult = whatifOriginalRecord._beforeResult;
    const delta = afterResult.attack_probability - beforeResult.attack_probability;

    document.getElementById('whatifAfterPred').textContent = afterResult.prediction;
    document.getElementById('whatifAfterPred').className = afterResult.is_attack ? 'attack-text' : 'normal-text';
    document.getElementById('whatifAfterProb').textContent = `${afterResult.attack_probability}%`;

    const deltaElem = document.getElementById('whatifProbDelta');
    deltaElem.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%`;
    deltaElem.className = delta >= 0 ? 'attack-text' : 'normal-text';

    document.getElementById('whatifAfterDetails').innerHTML = `
      <div><strong>Flag:</strong> ${modifiedRecord.flag}</div>
      <div><strong>serror_rate:</strong> ${modifiedRecord.serror_rate}</div>
      <div><strong>count:</strong> ${modifiedRecord.count}</div>
      <div><strong>logged_in:</strong> ${modifiedRecord.logged_in}</div>
    `;
  }

  // ========================================================
  // SIMULATED LIVE TRAFFIC STREAM
  // ========================================================
  function initLiveStream() {
    const btnStart = document.getElementById('btnStartStream');
    const btnPause = document.getElementById('btnPauseStream');
    const btnReset = document.getElementById('btnResetStream');
    const speedSelect = document.getElementById('streamSpeedSelect');

    btnStart.addEventListener('click', async () => {
      if (streamRecords.length === 0) {
        btnStart.disabled = true;
        btnStart.textContent = 'Loading Sample...';
        const res = await fetch('/api/stream_sample');
        streamRecords = await res.json();
        streamIndex = 0;
        document.getElementById('streamTableBody').innerHTML = '';
      }

      btnStart.disabled = true;
      btnPause.disabled = false;
      const speed = parseInt(speedSelect.value);
      streamTimer = setInterval(processNextStreamItem, speed);
    });

    btnPause.addEventListener('click', () => {
      clearInterval(streamTimer);
      btnStart.disabled = false;
      btnStart.textContent = '▶ Resume Simulation';
      btnPause.disabled = true;
    });

    btnReset.addEventListener('click', () => {
      clearInterval(streamTimer);
      streamRecords = [];
      streamIndex = 0;
      streamStats = { total: 0, normal: 0, attack: 0, correct: 0 };
      updateStreamCounters();
      document.getElementById('streamTableBody').innerHTML = '<tr><td colspan="10" class="empty-notice">Click "Start Simulation" to begin processing connections from KDDTest+.</td></tr>';
      btnStart.disabled = false;
      btnStart.textContent = '▶ Start Simulation';
      btnPause.disabled = true;
    });
  }

  async function processNextStreamItem() {
    if (streamIndex >= streamRecords.length) {
      clearInterval(streamTimer);
      document.getElementById('btnStartStream').disabled = false;
      document.getElementById('btnStartStream').textContent = '▶ Restart Simulation';
      document.getElementById('btnPauseStream').disabled = true;
      return;
    }

    const rec = streamRecords[streamIndex];
    streamIndex++;

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rec)
      });
      const result = await res.json();

      const isActualAttack = rec.attack_type !== 'normal';
      const isCorrect = result.is_attack === isActualAttack;

      streamStats.total++;
      if (result.is_attack) streamStats.attack++;
      else streamStats.normal++;
      if (isCorrect) streamStats.correct++;

      updateStreamCounters();
      appendStreamRow(rec, result, isCorrect);
    } catch (e) {
      console.error('Stream item error:', e);
    }
  }

  function updateStreamCounters() {
    document.getElementById('streamCountTotal').textContent = streamStats.total;
    document.getElementById('streamCountNormal').textContent = streamStats.normal;
    document.getElementById('streamCountAttack').textContent = streamStats.attack;
    
    const rate = streamStats.total > 0 ? (streamStats.attack / streamStats.total * 100).toFixed(1) : '0.0';
    document.getElementById('streamAttackRate').textContent = `${rate}%`;

    const acc = streamStats.total > 0 ? (streamStats.correct / streamStats.total * 100).toFixed(1) : '--';
    document.getElementById('streamAccuracy').textContent = acc !== '--' ? `${acc}%` : '--';
  }

  function appendStreamRow(rec, result, isCorrect) {
    const tbody = document.getElementById('streamTableBody');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>#${streamStats.total}</td>
      <td>${rec.protocol_type}</td>
      <td>${rec.service}</td>
      <td>${rec.flag}</td>
      <td>${rec.src_bytes}</td>
      <td>${rec.dst_bytes}</td>
      <td>${result.attack_probability}%</td>
      <td class="${result.is_attack ? 'attack-text' : 'normal-text'}">${result.prediction}</td>
      <td>${rec.attack_type}</td>
      <td><span class="${isCorrect ? 'ver-correct' : 'ver-incorrect'}">${isCorrect ? '✓ Match' : '✗ Miss'}</span></td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);
  }

  // ========================================================
  // BATCH CSV PREDICTION
  // ========================================================
  function initBatchPrediction() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('batchFileInput');
    const fileInfo = document.getElementById('batchFileInfo');
    const fileNameSpan = document.getElementById('batchFileName');
    const btnRunBatch = document.getElementById('btnRunBatch');

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--accent-blue)';
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
      if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });

    function handleFileSelection(file) {
      if (!file.name.endsWith('.csv')) {
        alert('Please select a valid CSV file.');
        return;
      }
      fileInfo.classList.remove('hidden');
      fileNameSpan.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      fileInfo._selectedFile = file;
    }

    btnRunBatch.addEventListener('click', async () => {
      const file = fileInfo._selectedFile;
      if (!file) return;

      btnRunBatch.disabled = true;
      btnRunBatch.textContent = 'Processing with XGBoost...';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/batch_predict', {
          method: 'POST',
          body: formData
        });
        const result = await res.json();
        if (result.error) {
          alert('Batch Error: ' + result.error);
          return;
        }

        displayBatchResults(result);
      } catch (err) {
        alert('Failed to process batch CSV: ' + err);
      } finally {
        btnRunBatch.disabled = false;
        btnRunBatch.textContent = 'Process Batch with XGBoost';
      }
    });
  }

  function displayBatchResults(data) {
    batchData = data;
    const container = document.getElementById('batchResultsContainer');
    container.classList.remove('hidden');

    document.getElementById('batchTotalRows').textContent = data.total_rows.toLocaleString();
    document.getElementById('batchAttacksFlagged').textContent = data.total_attacks_flagged.toLocaleString();
    document.getElementById('batchNormalFlagged').textContent = data.total_normal_flagged.toLocaleString();

    renderBatchTable(data.sample_predictions);

    // Setup filter buttons
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.getAttribute('data-filter');
        filterBatchTable(filter);
      });
    });
  }

  function renderBatchTable(rows) {
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    rows.forEach(r => {
      const tr = document.createElement('tr');
      const isAttack = r.prediction === 'MALICIOUS';
      tr.innerHTML = `
        <td>#${r.row_index}</td>
        <td class="${isAttack ? 'attack-text' : 'normal-text'}">${r.prediction}</td>
        <td>${r.attack_probability}%</td>
        <td>${r.actual_label || '--'}</td>
        <td>${r.is_correct !== undefined ? (r.is_correct ? '<span class="ver-correct">✓ Match</span>' : '<span class="ver-incorrect">✗ Miss</span>') : '--'}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function filterBatchTable(filter) {
    if (!batchData || !batchData.sample_predictions) return;
    let filtered = batchData.sample_predictions;

    if (filter === 'normal') {
      filtered = filtered.filter(r => r.prediction === 'NORMAL');
    } else if (filter === 'attack') {
      filtered = filtered.filter(r => r.prediction === 'MALICIOUS');
    } else if (filter === 'correct') {
      filtered = filtered.filter(r => r.is_correct === true);
    } else if (filter === 'incorrect') {
      filtered = filtered.filter(r => r.is_correct === false);
    }

    renderBatchTable(filtered);
  }

  // ========================================================
  // PERFORMANCE & GENERALIZATION
  // ========================================================
  async function loadPerformanceMetrics() {
    try {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      if (!data || !data.test_metrics) return;

      const vBase = data.validation_metrics.baseline;
      const vTuned = data.validation_metrics.tuned_threshold_optimal;
      const tBase = data.test_metrics.baseline;
      const tTuned = data.test_metrics.tuned_threshold_optimal;

      const tbody = document.getElementById('perfTableBody');
      const metricsList = [
        ['Accuracy', 'accuracy', '%'],
        ['Precision', 'precision', '%'],
        ['Recall (Sensitivity)', 'recall', '%'],
        ['F1-Score', 'f1_score', 'num'],
        ['ROC-AUC', 'roc_auc', 'num'],
        ['PR-AUC', 'pr_auc', 'num'],
        ['Specificity', 'specificity', '%'],
        ['Balanced Accuracy', 'balanced_accuracy', '%'],
        ['False Positive Rate (FPR)', 'false_positive_rate', '%'],
        ['False Negative Rate (FNR)', 'false_negative_rate', '%'],
        ['Inference Time (s)', 'inference_time_seconds', 'sec']
      ];

      tbody.innerHTML = '';
      metricsList.forEach(([label, key, type]) => {
        const tr = document.createElement('tr');
        const formatVal = (v) => {
          if (v === undefined || v === null) return '--';
          if (type === '%') return `${(v * 100).toFixed(2)}%`;
          if (type === 'num') return v.toFixed(4);
          if (type === 'sec') return `${v.toFixed(4)}s`;
          return v;
        };

        tr.innerHTML = `
          <td><strong>${label}</strong></td>
          <td>${formatVal(vBase[key])}</td>
          <td class="highlight">${formatVal(vTuned[key])}</td>
          <td>${formatVal(tBase[key])}</td>
          <td class="highlight"><strong>${formatVal(tTuned[key])}</strong></td>
        `;
        tbody.appendChild(tr);
      });

      // Render Attack Category Cards
      if (data.attack_category_performance) {
        const grid = document.getElementById('categoryCardsGrid');
        grid.innerHTML = '';
        for (const cat in data.attack_category_performance) {
          const info = data.attack_category_performance[cat];
          const card = document.createElement('div');
          card.className = 'category-card';
          
          const rateVal = cat === 'normal' ? info.accuracy_pct : info.detection_rate_pct;
          const rateLabel = cat === 'normal' ? 'Normal Accuracy' : 'Attack Detection Rate';
          
          card.innerHTML = `
            <h4>${cat.toUpperCase()} <span>${info.total_samples.toLocaleString()} samples</span></h4>
            <div class="cat-rate ${rateVal >= 80 ? 'normal-text' : (rateVal >= 50 ? 'highlight' : 'attack-text')}">${rateVal}%</div>
            <small style="color: var(--text-muted);">${rateLabel}</small>
          `;
          grid.appendChild(card);
        }
      }
    } catch (e) {
      console.error('Error loading performance metrics:', e);
    }
  }

  async function loadNovelAttacks() {
    try {
      const res = await fetch('/api/novel_attacks');
      const novelData = await res.json();
      const tbody = document.getElementById('novelTableBody');
      if (!novelData || novelData.length === 0) return;

      tbody.innerHTML = '';
      novelData.forEach(item => {
        const tr = document.createElement('tr');
        const rate = item.detection_rate_pct;
        tr.innerHTML = `
          <td><strong>${item.attack_type}</strong></td>
          <td>${item.attack_category}</td>
          <td>${item.total_samples}</td>
          <td class="normal-text">${item.detected_as_attack}</td>
          <td class="attack-text">${item.missed_as_normal}</td>
          <td class="${rate >= 80 ? 'normal-text' : (rate >= 50 ? 'highlight' : 'attack-text')}"><strong>${rate}%</strong></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error('Error loading novel attacks:', e);
    }
  }

  // ========================================================
  // RESEARCH & MODEL EVOLUTION MODULE
  // ========================================================
  async function loadResearchEvolution() {
    try {
      const res = await fetch('/api/research_evolution');
      if (!res.ok) {
        console.error('Failed to fetch research evolution data:', res.status);
        return;
      }
      const data = await res.json();
      if (!data) return;

      // Section 1: Hero
      if (data.hero) {
        const baseAccEl = document.getElementById('heroBaselineAcc');
        const regAccEl = document.getElementById('heroRegularizedAcc');
        const finalAccEl = document.getElementById('heroFinalAcc');
        if (baseAccEl) baseAccEl.textContent = `${data.experiments.baseline.accuracy.toFixed(2)}%`;
        if (regAccEl) regAccEl.textContent = `${data.experiments.regularized.accuracy.toFixed(2)}%`;
        if (finalAccEl) finalAccEl.textContent = `${data.experiments.final_cost_sensitive.accuracy.toFixed(2)}%`;
      }

      // Section 2: Methodology Timeline
      const timeline = document.getElementById('methodologyTimeline');
      if (timeline && data.methodology) {
        timeline.innerHTML = '';
        data.methodology.forEach((stage, idx) => {
          const item = document.createElement('div');
          item.className = `methodology-stage-item ${idx === 0 ? 'open' : ''}`;
          item.innerHTML = `
            <div class="methodology-stage-header">
              <div class="stage-title-wrap">
                <span class="stage-num-badge">0${stage.id}</span>
                <span class="stage-name">${stage.step}</span>
              </div>
              <span class="stage-toggle-icon">▼</span>
            </div>
            <div class="methodology-stage-body">
              <div class="stage-section-block">
                <span class="block-tag tag-why">WHY</span>
                <p class="block-text">${stage.why}</p>
              </div>
              <div class="stage-section-block">
                <span class="block-tag tag-change">CHANGE</span>
                <p class="block-text">${stage.change}</p>
              </div>
              <div class="stage-section-block">
                <span class="block-tag tag-result">RESULT</span>
                <p class="block-text">${stage.result}</p>
              </div>
            </div>
          `;

          const header = item.querySelector('.methodology-stage-header');
          header.addEventListener('click', () => {
            item.classList.toggle('open');
          });

          timeline.appendChild(item);
        });
      }

      // Section 3: Accuracy Progression Bar Chart (SVG)
      const accChart = document.getElementById('accuracyChartContainer');
      if (accChart && data.experiments) {
        const exp = data.experiments;
        const items = [
          { name: 'Baseline', val: exp.baseline.accuracy, color: 'rgba(255, 255, 255, 0.35)', meta: 'τ=0.41' },
          { name: 'Regularized', val: exp.regularized.accuracy, color: 'var(--accent-blue)', meta: 'τ=0.36' },
          { name: 'Final Cost-Sensitive', val: exp.final_cost_sensitive.accuracy, color: 'var(--accent-green)', meta: 'τ*=0.10' }
        ];

        let barsHtml = '';
        items.forEach(item => {
          barsHtml += `
            <div style="margin-bottom: 1.1rem;">
              <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem; font-size: 0.88rem;">
                <span style="font-weight: 600; color: #fff;">${item.name} <small style="color: var(--text-muted); font-family: var(--font-mono);">(${item.meta})</small></span>
                <span style="font-family: var(--font-mono); font-weight: 800; color: ${item.color === 'var(--accent-green)' ? 'var(--accent-green)' : '#fff'};">${item.val.toFixed(2)}%</span>
              </div>
              <div style="height: 22px; background: rgba(255, 255, 255, 0.05); border-radius: 6px; overflow: hidden; position: relative;">
                <div style="width: ${item.val}%; height: 100%; background: ${item.color}; border-radius: 6px; transition: width 1s ease;"></div>
              </div>
            </div>
          `;
        });
        accChart.innerHTML = barsHtml;
      }

      // Section 4: Metric Evolution (Precision, Recall, F1)
      const evoDisplay = document.getElementById('metricEvolutionDisplay');
      if (evoDisplay && data.experiments) {
        const base = data.experiments.baseline;
        const fin = data.experiments.final_cost_sensitive;
        const metrics = [
          { label: 'Precision', base: base.precision, final: fin.precision, isPct: true, diff: (fin.precision - base.precision).toFixed(2) + ' pp' },
          { label: 'Recall', base: base.recall, final: fin.recall, isPct: true, diff: '+' + (fin.recall - base.recall).toFixed(2) + ' pp' },
          { label: 'F1-Score', base: base.f1, final: fin.f1, isPct: false, diff: '+' + (fin.f1 - base.f1).toFixed(4) }
        ];

        let evoHtml = '';
        metrics.forEach(m => {
          const baseFormatted = m.isPct ? `${m.base.toFixed(2)}%` : m.base.toFixed(4);
          const finalFormatted = m.isPct ? `${m.final.toFixed(2)}%` : m.final.toFixed(4);
          const baseWidth = m.isPct ? m.base : (m.base * 100);
          const finalWidth = m.isPct ? m.final : (m.final * 100);

          evoHtml += `
            <div class="metric-evo-row">
              <div class="evo-header">
                <span class="evo-name">${m.label}</span>
                <span class="evo-values">
                  <span class="val-baseline">${baseFormatted}</span>
                  <span class="val-arrow">→</span>
                  <span class="val-final">${finalFormatted}</span>
                  <span class="diff-badge positive" style="margin-left: 0.6rem; font-size: 0.72rem;">${m.diff}</span>
                </span>
              </div>
              <div class="evo-bar-track">
                <div class="evo-bar-fill-baseline" style="width: ${baseWidth}%;"></div>
                <div class="evo-bar-fill-final" style="width: ${finalWidth}%; background: ${m.label === 'Recall' ? 'var(--accent-green)' : 'var(--accent-blue)'};"></div>
              </div>
            </div>
          `;
        });
        evoDisplay.innerHTML = evoHtml;
      }

      // Section 5: Novel Attack Generalization
      const novelContainer = document.getElementById('novelChartContainer');
      if (novelContainer && data.experiments) {
        const exp = data.experiments;
        const variants = [
          { name: 'Baseline Model', rate: exp.baseline.novel_attack_detection_rate, detected: exp.baseline.novel_attacks_detected, total: exp.baseline.novel_attacks_total, color: 'rgba(255, 255, 255, 0.3)' },
          { name: 'Regularized Model', rate: exp.regularized.novel_attack_detection_rate, detected: exp.regularized.novel_attacks_detected, total: exp.regularized.novel_attacks_total, color: 'var(--accent-blue)' },
          { name: 'Final Cost-Sensitive', rate: exp.final_cost_sensitive.novel_attack_detection_rate, detected: exp.final_cost_sensitive.novel_attacks_detected, total: exp.final_cost_sensitive.novel_attacks_total, color: 'var(--accent-cyan)' },
          { name: 'High-Detection Variant', rate: exp.high_detection_variant.novel_attack_detection_rate, detected: exp.high_detection_variant.novel_attacks_detected, total: exp.high_detection_variant.novel_attacks_total, color: 'var(--accent-purple)' }
        ];

        let novelHtml = '<div style="display: flex; flex-direction: column; gap: 1rem;">';
        variants.forEach(v => {
          novelHtml += `
            <div>
              <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem; font-size: 0.86rem;">
                <span style="font-weight: 600; color: #fff;">${v.name}</span>
                <span style="font-family: var(--font-mono); font-size: 0.82rem;">
                  <span style="color: var(--text-secondary);">${v.detected.toLocaleString()} / ${v.total.toLocaleString()}</span>
                  <strong style="color: ${v.color}; margin-left: 0.5rem; font-size: 0.95rem;">${v.rate.toFixed(2)}%</strong>
                </span>
              </div>
              <div style="height: 14px; background: rgba(255, 255, 255, 0.05); border-radius: 7px; overflow: hidden; position: relative;">
                <div style="width: ${v.rate}%; height: 100%; background: ${v.color}; border-radius: 7px;"></div>
              </div>
            </div>
          `;
        });
        novelHtml += '</div>';
        novelContainer.innerHTML = novelHtml;
      }

      // Section 6: Recall vs False Positive Rate (Interactive Scatter Plot)
      const scatterWrapper = document.getElementById('scatterPlotWrapper');
      if (scatterWrapper && data.experiments) {
        const exp = data.experiments;
        const points = [
          { name: 'Baseline', fpr: exp.baseline.fpr, recall: exp.baseline.recall, threshold: exp.baseline.threshold, color: '#58a6ff' },
          { name: 'Final Balanced', fpr: exp.final_cost_sensitive.fpr, recall: exp.final_cost_sensitive.recall, threshold: exp.final_cost_sensitive.threshold, color: '#3fb950' },
          { name: 'High Detection', fpr: exp.high_detection_variant.fpr, recall: exp.high_detection_variant.recall, threshold: exp.high_detection_variant.threshold, color: '#bc8cff' }
        ];

        // Coordinate scaling: FPR 0-10% -> X (60 to 440), Recall 60-90% -> Y (250 to 40)
        const svgW = 500;
        const svgH = 300;
        const padX = 60;
        const padY = 40;
        const graphW = svgW - padX - 30;
        const graphH = svgH - padY - 50;

        const scaleX = (fpr) => padX + (fpr / 10) * graphW;
        const scaleY = (rec) => (svgH - 50) - ((rec - 60) / 30) * graphH;

        let pointsSvg = '';
        points.forEach((pt, i) => {
          const cx = scaleX(pt.fpr);
          const cy = scaleY(pt.recall);
          pointsSvg += `
            <g class="scatter-node" data-idx="${i}" style="cursor: pointer;">
              <circle cx="${cx}" cy="${cy}" r="9" fill="${pt.color}" stroke="#ffffff" stroke-width="2" />
              <text x="${cx + 12}" y="${cy + 4}" fill="#ffffff" font-size="11" font-family="'Inter', sans-serif" font-weight="600">${pt.name}</text>
            </g>
          `;
        });

        const svgContent = `
          <svg viewBox="0 0 ${svgW} ${svgH}" width="100%" height="280" style="overflow: visible;">
            <!-- Grid lines -->
            <line x1="${padX}" y1="${scaleY(60)}" x2="${padX + graphW}" y2="${scaleY(60)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="3,3" />
            <line x1="${padX}" y1="${scaleY(70)}" x2="${padX + graphW}" y2="${scaleY(70)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="3,3" />
            <line x1="${padX}" y1="${scaleY(80)}" x2="${padX + graphW}" y2="${scaleY(80)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="3,3" />
            <line x1="${padX}" y1="${scaleY(90)}" x2="${padX + graphW}" y2="${scaleY(90)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="3,3" />
            
            <line x1="${scaleX(0)}" y1="${padY}" x2="${scaleX(0)}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.2)" />
            <line x1="${scaleX(2.5)}" y1="${padY}" x2="${scaleX(2.5)}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3" />
            <line x1="${scaleX(5.0)}" y1="${padY}" x2="${scaleX(5.0)}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3" />
            <line x1="${scaleX(7.5)}" y1="${padY}" x2="${scaleX(7.5)}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3" />
            <line x1="${scaleX(10)}" y1="${padY}" x2="${scaleX(10)}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3" />

            <!-- Axes -->
            <line x1="${padX}" y1="${svgH - 50}" x2="${padX + graphW}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
            <line x1="${padX}" y1="${padY}" x2="${padX}" y2="${svgH - 50}" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />

            <!-- Y Axis labels -->
            <text x="${padX - 10}" y="${scaleY(60) + 4}" fill="var(--text-muted)" font-size="10" text-anchor="end" font-family="'JetBrains Mono', monospace">60%</text>
            <text x="${padX - 10}" y="${scaleY(70) + 4}" fill="var(--text-muted)" font-size="10" text-anchor="end" font-family="'JetBrains Mono', monospace">70%</text>
            <text x="${padX - 10}" y="${scaleY(80) + 4}" fill="var(--text-muted)" font-size="10" text-anchor="end" font-family="'JetBrains Mono', monospace">80%</text>
            <text x="${padX - 10}" y="${scaleY(90) + 4}" fill="var(--text-muted)" font-size="10" text-anchor="end" font-family="'JetBrains Mono', monospace">90%</text>

            <!-- X Axis labels -->
            <text x="${scaleX(0)}" y="${svgH - 32}" fill="var(--text-muted)" font-size="10" text-anchor="middle" font-family="'JetBrains Mono', monospace">0%</text>
            <text x="${scaleX(2.5)}" y="${svgH - 32}" fill="var(--text-muted)" font-size="10" text-anchor="middle" font-family="'JetBrains Mono', monospace">2.5%</text>
            <text x="${scaleX(5.0)}" y="${svgH - 32}" fill="var(--text-muted)" font-size="10" text-anchor="middle" font-family="'JetBrains Mono', monospace">5.0%</text>
            <text x="${scaleX(7.5)}" y="${svgH - 32}" fill="var(--text-muted)" font-size="10" text-anchor="middle" font-family="'JetBrains Mono', monospace">7.5%</text>
            <text x="${scaleX(10)}" y="${svgH - 32}" fill="var(--text-muted)" font-size="10" text-anchor="middle" font-family="'JetBrains Mono', monospace">10%</text>

            <!-- Axis Titles -->
            <text x="${padX + graphW / 2}" y="${svgH - 12}" fill="var(--text-secondary)" font-size="11" text-anchor="middle" font-family="'Inter', sans-serif">False Positive Rate (FPR %)</text>
            <text x="16" y="${padY + graphH / 2}" fill="var(--text-secondary)" font-size="11" text-anchor="middle" font-family="'Inter', sans-serif" transform="rotate(-90 16 ${padY + graphH / 2})">Recall (%)</text>

            <!-- Trade-off curve line -->
            <path d="M ${scaleX(points[0].fpr)} ${scaleY(points[0].recall)} L ${scaleX(points[1].fpr)} ${scaleY(points[1].recall)} L ${scaleX(points[2].fpr)} ${scaleY(points[2].recall)}" fill="none" stroke="rgba(88, 166, 255, 0.4)" stroke-width="2" stroke-dasharray="4,4" />

            <!-- Points -->
            ${pointsSvg}
          </svg>
        `;
        scatterWrapper.innerHTML = svgContent + `<div id="scatterTooltip" class="scatter-tooltip hidden"></div>`;

        // Attach hover listeners
        const activeTooltip = scatterWrapper.querySelector('#scatterTooltip');
        scatterWrapper.querySelectorAll('.scatter-node').forEach(node => {
          node.addEventListener('mouseenter', () => {
            const idx = parseInt(node.getAttribute('data-idx'));
            const pt = points[idx];
            activeTooltip.innerHTML = `
              <strong>${pt.name}</strong><br>
              Threshold: τ = ${pt.threshold.toFixed(2)}<br>
              Recall: ${pt.recall.toFixed(2)}%<br>
              FPR: ${pt.fpr.toFixed(2)}%
            `;
            activeTooltip.classList.remove('hidden');
          });

          node.addEventListener('mousemove', (e) => {
            const rect = scatterWrapper.getBoundingClientRect();
            activeTooltip.style.left = (e.clientX - rect.left + 15) + 'px';
            activeTooltip.style.top = (e.clientY - rect.top - 20) + 'px';
          });

          node.addEventListener('mouseleave', () => {
            activeTooltip.classList.add('hidden');
          });
        });
      }

      // Section 7: Attack Category Improvement
      const catCompContainer = document.getElementById('categoryComparisonContainer');
      if (catCompContainer && data.experiments) {
        const baseCats = data.experiments.baseline.categories;
        const finCats = data.experiments.final_cost_sensitive.categories;
        const catOrder = [
          { key: 'DoS', name: 'DoS (Denial of Service)' },
          { key: 'Probe', name: 'Probe (Port Scanning & Recon)' },
          { key: 'R2L', name: 'R2L (Remote to Local Exploits)' },
          { key: 'U2R', name: 'U2R (User to Root Privilege Escalation)' }
        ];

        let catHtml = '';
        catOrder.forEach(cat => {
          const bVal = baseCats[cat.key];
          const fVal = finCats[cat.key];
          const diff = (fVal - bVal).toFixed(2);
          catHtml += `
            <div class="cat-comp-row">
              <div class="cat-comp-labels">
                <span class="cat-comp-name">${cat.name}</span>
                <span class="cat-comp-values">
                  <span style="color: var(--text-muted);">${bVal.toFixed(2)}%</span>
                  <span style="color: var(--text-secondary); margin: 0 0.3rem;">→</span>
                  <strong style="color: var(--accent-cyan);">${fVal.toFixed(2)}%</strong>
                  <span class="diff-badge positive" style="margin-left: 0.5rem; font-size: 0.72rem;">+${diff} pp</span>
                </span>
              </div>
              <div class="cat-comp-track">
                <div class="cat-fill-baseline" style="width: ${bVal}%;"></div>
                <div class="cat-fill-final" style="width: ${fVal}%;"></div>
              </div>
            </div>
          `;
        });
        catCompContainer.innerHTML = catHtml;
      }

      // Section 8: Side-by-Side Confusion Matrices
      const cmContainer = document.getElementById('cmSideBySide');
      if (cmContainer && data.experiments) {
        const bCM = data.experiments.baseline.confusion_matrix;
        const fCM = data.experiments.final_cost_sensitive.confusion_matrix;

        cmContainer.innerHTML = `
          <!-- Baseline Matrix -->
          <div class="cm-matrix-card">
            <div class="cm-matrix-title">Baseline XGBoost (τ = 0.41)</div>
            <div class="cm-grid-2x2">
              <div class="cm-cell cell-tn">
                <span class="cell-label">True Negative (TN)</span>
                <span class="cell-count">${bCM.tn.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-fp">
                <span class="cell-label">False Positive (FP)</span>
                <span class="cell-count">${bCM.fp.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-fn">
                <span class="cell-label">False Negative (FN)</span>
                <span class="cell-count">${bCM.fn.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-tp">
                <span class="cell-label">True Positive (TP)</span>
                <span class="cell-count">${bCM.tp.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <!-- Final Matrix -->
          <div class="cm-matrix-card final-highlight">
            <div class="cm-matrix-title">Final Model (τ* = 0.10)</div>
            <div class="cm-grid-2x2">
              <div class="cm-cell cell-tn">
                <span class="cell-label">True Negative (TN)</span>
                <span class="cell-count">${fCM.tn.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-fp">
                <span class="cell-label">False Positive (FP)</span>
                <span class="cell-count">${fCM.fp.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-fn">
                <span class="cell-label">False Negative (FN)</span>
                <span class="cell-count" style="color: var(--accent-green);">${fCM.fn.toLocaleString()}</span>
              </div>
              <div class="cm-cell cell-tp">
                <span class="cell-label">True Positive (TP)</span>
                <span class="cell-count">${fCM.tp.toLocaleString()}</span>
              </div>
            </div>
          </div>
        `;
      }

      // Section 9: Configuration Rationale Table
      const rationaleBody = document.getElementById('configRationaleBody');
      if (rationaleBody && data.model_config && data.model_config.parameter_changes) {
        rationaleBody.innerHTML = '';
        data.model_config.parameter_changes.forEach(p => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><code>${p.param}</code></td>
            <td style="color: var(--text-muted); font-family: var(--font-mono);">${p.baseline}</td>
            <td style="color: var(--accent-green); font-weight: 700; font-family: var(--font-mono);">${p.final}</td>
            <td>${p.purpose}</td>
          `;
          rationaleBody.appendChild(tr);
        });
      }

      // Section 10: Research Findings Cards
      const findingsGrid = document.getElementById('findingsCardsGrid');
      if (findingsGrid && data.findings) {
        findingsGrid.innerHTML = '';
        data.findings.forEach(f => {
          const card = document.createElement('div');
          card.className = 'finding-card';
          card.innerHTML = `
            <div class="finding-card-header">
              <span class="finding-title">${f.title}</span>
              <span class="finding-diff">${f.diff}</span>
            </div>
            <div class="finding-progression">
              <span class="finding-base">${f.baseline}</span>
              <span class="finding-arrow">→</span>
              <span class="finding-final">${f.final}</span>
            </div>
            <span class="finding-note">${f.note}</span>
          `;
          findingsGrid.appendChild(card);
        });
      }

      // Section 11: Experimental Integrity Checklist
      const integrityGrid = document.getElementById('integrityGrid');
      if (integrityGrid && data.integrity_checklist) {
        integrityGrid.innerHTML = '';
        data.integrity_checklist.forEach(item => {
          const div = document.createElement('div');
          div.className = 'integrity-item';
          div.innerHTML = `
            <span class="integrity-check-icon">✓</span>
            <span class="integrity-item-text">${item}</span>
          `;
          integrityGrid.appendChild(div);
        });
      }

    } catch (err) {
      console.error('Error loading Research Evolution module:', err);
    }
  }
});
