document.addEventListener('DOMContentLoaded', () => {
  chrome.runtime.sendMessage({ action: 'getToken' }, (res) => {
    if (res && res.token) {
      document.getElementById('hfToken').value = res.token;
    }
  });
  chrome.runtime.sendMessage({ action: 'getSettings' }, (res) => {
    const s = res && res.settings;
    if (s) {
      const modeEl = document.getElementById('inferenceMode');
      const urlEl = document.getElementById('backendUrl');
      if (modeEl) modeEl.value = s.inference_mode || 'hf';
      if (urlEl) urlEl.value = s.backend_url || 'http://localhost:8000';
    }
  });
});

document.getElementById('saveToken').addEventListener('click', () => {
  const token = document.getElementById('hfToken').value.trim();
  chrome.runtime.sendMessage({ action: 'setToken', token }, (res) => {
    console.log('Token saved', res);
  });
});

// Open review page
const openReview = document.getElementById('openReview');
if (openReview) {
  openReview.addEventListener('click', () => {
    const url = chrome.runtime.getURL('detected.html');
    chrome.tabs.create({ url });
  });
}

// Save inference settings (mode + backend URL)
const saveSettingsBtn = document.getElementById('saveSettings');
if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener('click', () => {
    const mode = document.getElementById('inferenceMode').value;
    const url = document.getElementById('backendUrl').value.trim();
    chrome.runtime.sendMessage({ action: 'setSettings', inference_mode: mode, backend_url: url }, (res) => {
      console.log('Settings saved', res);
    });
  });
}