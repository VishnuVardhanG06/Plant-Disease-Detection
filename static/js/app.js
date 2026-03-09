/**
 * app.js — Plant Disease Detection Web App
 *
 * Handles:
 *  - Drag-and-drop and click-to-browse file selection
 *  - Live image preview via FileReader
 *  - POST /predict via Fetch API (multipart/form-data)
 *  - Animated SVG confidence ring reveal
 *  - Tabbed tips (Symptoms / Prevention / Treatment)
 *  - Reset flow
 *  - Graceful error display
 */

/* ── DOM References ──────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const previewWrap    = document.getElementById('preview-wrap');
const previewImg     = document.getElementById('preview-img');
const changeBtn      = document.getElementById('change-btn');
const predictBtn     = document.getElementById('predict-btn');
const resetBtn       = document.getElementById('reset-btn');
const btnText        = document.getElementById('btn-text');
const btnSpinner     = document.getElementById('btn-spinner');

const resultsPanel   = document.getElementById('results-panel');
const errorPanel     = document.getElementById('error-panel');
const errorTitle     = document.getElementById('error-title');
const errorMsg       = document.getElementById('error-msg');
const errorRetryBtn  = document.getElementById('error-retry-btn');

const diseaseName    = document.getElementById('disease-name');
const diseaseRaw     = document.getElementById('disease-raw');
const severityDot    = document.getElementById('severity-dot');
const severityBadge  = document.getElementById('severity-badge');
const ringFill       = document.getElementById('ring-fill');
const confPct        = document.getElementById('conf-pct');
const diseaseDesc    = document.getElementById('disease-description');
const tipList        = document.getElementById('tip-list');
const tabBtns        = document.querySelectorAll('.tab-btn');

/* ── State ───────────────────────────────────────────────────────────────── */
let selectedFile = null;       // The File object chosen by the user
let currentTips  = {};         // Tips data from the last API response

/* ── SVG Ring Constants ──────────────────────────────────────────────────── */
// Circle circumference = 2π × r = 2π × 48 ≈ 301.6
const RING_CIRC = 2 * Math.PI * 48;

/* ═══════════════════════════════════════════════════════════
   FILE SELECTION (Drop Zone + Click)
   ══════════════════════════════════════════════════════════ */

/** Open the hidden file input when the drop zone is clicked or Enter pressed */
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

/** Handle file chosen via the native file picker */
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

/** "Change image" button re-opens the file picker */
changeBtn.addEventListener('click', () => fileInput.click());

/* ── Drag & Drop ─────────────────────────────────────────────────────────── */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

['dragleave', 'dragend'].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'))
);

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

/* ── Process a selected file ─────────────────────────────────────────────── */
/**
 * Validate and preview an uploaded image file.
 * @param {File} file — The image File object.
 */
function handleFile(file) {
  // Validate MIME type
  if (!file.type.startsWith('image/')) {
    showError('Invalid file type', 'Please upload a JPG, PNG, or BMP image.');
    return;
  }

  selectedFile = file;

  // Show live preview using FileReader (no server round-trip)
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropZone.classList.add('hidden');
    previewWrap.classList.remove('hidden');
    predictBtn.disabled = false;        // Enable analyse button
  };
  reader.readAsDataURL(file);

  // Hide previous results
  hideResults();
}

/* ═══════════════════════════════════════════════════════════
   PREDICTION
   ══════════════════════════════════════════════════════════ */

predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!selectedFile) return;

  setLoading(true);
  hideResults();

  // Build multipart form — field name must match Flask's request.files['image']
  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      // Server returned a 4xx/5xx with { error: "..." }
      showError('Prediction Failed', data.error || 'An unexpected error occurred.');
      return;
    }

    displayResults(data);

  } catch (err) {
    // Network/CORS error
    showError('Connection Error', 'Could not reach the server. Make sure app.py is running.');
  } finally {
    setLoading(false);
  }
}

/* ── Show prediction results ─────────────────────────────────────────────── */
/**
 * Populate and reveal the results panel with the API response data.
 * @param {Object} data — JSON response from /predict
 */
function displayResults(data) {
  const sev = data.severity || 'unknown';

  // Disease name + raw label
  diseaseName.textContent = data.display_name || data.disease || 'Unknown';
  diseaseRaw.textContent  = data.disease || '—';

  // Severity dot + badge
  severityDot.className   = `severity-dot ${sev}`;
  severityBadge.textContent = capitalise(sev);
  severityBadge.className  = `severity-badge ${sev}`;

  // Description
  diseaseDesc.textContent = data.tips?.description || '—';

  // Confidence ring animation
  animateRing(data.confidence || 0, sev);

  // Tips — populate all tabs, activate "symptoms" by default
  currentTips = data.tips || {};
  activateTab('symptoms');

  // Reveal results panel
  resultsPanel.classList.remove('hidden');
  errorPanel.classList.add('hidden');

  // Smooth scroll to results on mobile
  if (window.innerWidth < 720) {
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ── Animate the SVG confidence ring ─────────────────────────────────────── */
/**
 * Animate the circular progress ring from 0 → confidence%.
 * @param {number} pct     — Confidence percentage (0-100).
 * @param {string} severity — Severity class name for ring colour.
 */
function animateRing(pct, severity) {
  // Reset ring to 0 instantly before re-animating
  ringFill.style.transition = 'none';
  ringFill.style.strokeDashoffset = RING_CIRC;
  ringFill.className = `ring-fill sev-${severity}`;
  confPct.textContent = '0%';

  // Trigger browser reflow so the transition restart is visible
  void ringFill.getBoundingClientRect();

  // Animate stroke to target value
  ringFill.style.transition = 'stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke .4s ease';
  const offset = RING_CIRC * (1 - pct / 100);
  ringFill.style.strokeDashoffset = offset;

  // Count-up the numeric percentage text
  let current = 0;
  const step  = pct / 60;   // ~60 frames
  const timer = setInterval(() => {
    current = Math.min(current + step, pct);
    confPct.textContent = Math.round(current) + '%';
    if (current >= pct) clearInterval(timer);
  }, 16);
}

/* ═══════════════════════════════════════════════════════════
   TABBED TIPS
   ══════════════════════════════════════════════════════════ */

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

/**
 * Show the tip list for the selected tab.
 * @param {string} tabKey — 'symptoms' | 'prevention' | 'treatment'
 */
function activateTab(tabKey) {
  // Update tab button states
  tabBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabKey);
    btn.setAttribute('aria-selected', btn.dataset.tab === tabKey);
  });

  // Rebuild the tip list
  const items = currentTips[tabKey] || [];
  tipList.innerHTML = '';
  items.forEach((text, i) => {
    const li = document.createElement('li');
    li.textContent = text;
    li.style.animationDelay = `${i * 0.06}s`;
    tipList.appendChild(li);
  });

  if (items.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No information available for this category.';
    li.style.opacity = '.5';
    tipList.appendChild(li);
  }
}

/* ═══════════════════════════════════════════════════════════
   RESET
   ══════════════════════════════════════════════════════════ */

resetBtn.addEventListener('click', resetAll);
errorRetryBtn.addEventListener('click', resetAll);

/** Clear all state and return to the initial upload view. */
function resetAll() {
  selectedFile = null;
  currentTips  = {};

  // Reset file input so the same file can be re-selected
  fileInput.value = '';
  previewImg.src  = '';

  // Restore drop zone, hide preview + results + error
  dropZone.classList.remove('hidden');
  previewWrap.classList.add('hidden');
  hideResults();

  // Reset ring
  ringFill.style.transition = 'none';
  ringFill.style.strokeDashoffset = RING_CIRC;
  confPct.textContent = '0%';

  predictBtn.disabled = true;
}

/* ═══════════════════════════════════════════════════════════
   UI HELPERS
   ══════════════════════════════════════════════════════════ */

/** Toggle the loading state of the Analyze button. */
function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnText.classList.toggle('hidden', isLoading);
  btnSpinner.classList.toggle('hidden', !isLoading);
}

/** Hide both results and error panels. */
function hideResults() {
  resultsPanel.classList.add('hidden');
  errorPanel.classList.add('hidden');
}

/**
 * Show the error panel with a custom title and message.
 * @param {string} title   — Short error heading.
 * @param {string} message — Detailed error message.
 */
function showError(title, message) {
  errorTitle.textContent = title;
  errorMsg.textContent   = message;
  errorPanel.classList.remove('hidden');
  resultsPanel.classList.add('hidden');
}

/** Capitalise the first letter of a string. */
function capitalise(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}
