/* =========================================================================
   FraudShield AI — main.js
   ========================================================================= */

// ---------------------------------------------------------------------------
// Toast notification system
// ---------------------------------------------------------------------------
(function () {
  // Create toast container
  const container = document.createElement('div');
  container.id = 'toastContainer';
  document.body.appendChild(container);

  /**
   * Show a toast notification.
   * @param {string} message  - Text to display
   * @param {string} type     - 'success' | 'danger' | 'warning' | 'info'
   * @param {number} duration - ms before auto-dismiss (0 = permanent)
   */
  window.showToast = function (message, type = 'info', duration = 5000) {
    const icons = {
      success: 'fa-circle-check',
      danger:  'fa-circle-xmark',
      warning: 'fa-triangle-exclamation',
      info:    'fa-circle-info',
    };
    const icon = icons[type] || icons.info;

    const toast = document.createElement('div');
    toast.className = `custom-toast ${type}`;
    toast.innerHTML = `
      <i class="fa-solid ${icon} toast-icon"></i>
      <span class="flex-grow-1 small">${message}</span>
      <button class="btn-close btn-close-white btn-sm ms-2" aria-label="Close"></button>
    `;

    toast.querySelector('.btn-close').addEventListener('click', () => dismissToast(toast));
    container.appendChild(toast);

    if (duration > 0) {
      setTimeout(() => dismissToast(toast), duration);
    }
  };

  function dismissToast(toast) {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'opacity 0.3s, transform 0.3s';
    setTimeout(() => toast.remove(), 320);
  }
})();


// ---------------------------------------------------------------------------
// File upload: drag-and-drop + file input (supports multiple drop zones)
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
  const uploadForm = document.getElementById('uploadForm');
  const sizeWarn   = document.getElementById('sizeWarning');
  const uploadBtn  = document.getElementById('uploadBtn');
  const spinner    = document.getElementById('uploadSpinner');
  const uploadIcon = document.getElementById('uploadIcon');

  // Initialize every .drop-zone on the page. The HTML provides
  // data-input (id of <input>) and data-preview (id of preview container).
  const zones = document.querySelectorAll('.drop-zone');
  zones.forEach(zone => {
    const inputId = zone.dataset.input;
    const previewId = zone.dataset.preview;
    const input = document.getElementById(inputId);
    const preview = previewId ? document.getElementById(previewId) : null;
    if (!input) return;

    // Drag styling
    ['dragenter', 'dragover'].forEach(evt =>
      zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add('drag-over'); })
    );
    ['dragleave', 'drop'].forEach(evt =>
      zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove('drag-over'); })
    );

    zone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFileSelection(input, preview, files[0]);
    });

    zone.addEventListener('click', () => input.click());

    input.addEventListener('change', () => {
      if (input.files.length > 0) handleFileSelection(input, preview, input.files[0]);
    });

    // wire remove button inside preview (if present)
    if (preview) {
      const removeBtn = preview.querySelector('button[data-clear]');
      if (removeBtn) {
        removeBtn.addEventListener('click', () => {
          input.value = '';
          preview.classList.add('d-none');
          if (uploadBtn) uploadBtn.disabled = true;
        });
      }
    }
  });

  function handleFileSelection(input, preview, file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Only CSV files are supported.', 'danger');
      return;
    }

    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;

    if (preview) {
      const nameEl = preview.querySelector('[id$="Name"]');
      const sizeEl = preview.querySelector('[id$="Size"]');
      if (nameEl) nameEl.textContent = file.name;
      if (sizeEl) sizeEl.textContent = formatBytes(file.size);
      preview.classList.remove('d-none');
    }

    if (sizeWarn) sizeWarn.classList.toggle('d-none', file.size <= 100 * 1024 * 1024);
    if (uploadBtn) uploadBtn.disabled = false;
  }

  // Show spinner on submit
  if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
      if (spinner) spinner.classList.remove('d-none');
      if (uploadIcon) uploadIcon.classList.add('d-none');
      if (uploadBtn) uploadBtn.disabled = true;
    });
  }
});


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatBytes(bytes) {
  if (bytes < 1024)          return bytes + ' B';
  if (bytes < 1024 * 1024)   return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 ** 3)     return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 ** 3)).toFixed(2) + ' GB';
}


// ---------------------------------------------------------------------------
// Auto-dismiss Bootstrap flash toasts
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.toast').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.4s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 420);
    }, 6000);
  });
});


// ---------------------------------------------------------------------------
// Progress bar helper (used in EDA page navigation)
// ---------------------------------------------------------------------------
window.updateProgressBar = function (current, total) {
  const bar   = document.getElementById('overallProgress');
  const label = document.getElementById('progressLabel');
  if (!bar) return;
  const pct = Math.round((current / total) * 100);
  bar.style.width = pct + '%';
  bar.setAttribute('aria-valuenow', pct);
  if (label) label.textContent = `Step ${current} of ${total}`;
};


// ---------------------------------------------------------------------------
// Keyboard navigation for EDA steps
// ---------------------------------------------------------------------------
document.addEventListener('keydown', function (e) {
  if (typeof navigateStep !== 'function') return;
  if (typeof currentStep === 'undefined') return;

  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    navigateStep(currentStep + 1);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    navigateStep(currentStep - 1);
  }
});
