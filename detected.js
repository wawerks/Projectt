function fmtDate(ts) {
  const d = new Date(ts);
  return d.toLocaleString();
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else e.setAttribute(k, v);
  });
  children.forEach((c) => e.appendChild(c));
  return e;
}

async function loadItems() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'getFlagged' }, (res) => {
      resolve((res && res.items) || []);
    });
  });
}

async function removeItem(id) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'removeFlagged', id }, () => resolve());
  });
}

async function clearAll() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'clearFlagged' }, () => resolve());
  });
}

function render(items) {
  const list = document.getElementById('list');
  const empty = document.getElementById('empty');
  list.innerHTML = '';

  if (!items.length) {
    empty.style.display = '';
    return;
  }
  empty.style.display = 'none';

  items.forEach((it) => {
    const img = el('img', { src: it.thumbnail, class: 'thumb', alt: it.label || 'AI image' });
    const meta = el('div', { class: 'meta' });
    const sc = typeof it.score === 'number' ? it.score : null;
    const pct = sc != null ? Math.round(sc * 100) : null;
    const cleanLabel = (it.label || '').replace(/</g, '&lt;');

    // Confidence block
    const confWrap = el('div', { class: 'conf' });
    const confText = el('div', { class: 'conf-text', text: sc != null ? `Confidence: ${pct}% (${sc.toFixed(2)})` : 'Confidence: n/a' });
    const confBar = el('div', { class: 'conf-bar' });
    const confFill = el('div', { class: 'conf-bar-fill' });
    if (pct != null) confFill.style.width = `${pct}%`;
    confBar.appendChild(confFill);
    confWrap.appendChild(confText);
    confWrap.appendChild(confBar);

    meta.innerHTML = `Label: <b>${cleanLabel}</b><br/>When: <b>${fmtDate(it.timestamp)}</b><br/>From: <a href="${it.pageUrl}" target="_blank">Open page</a>`;

    const openBtn = el('button', { text: 'Open Source Image' });
    openBtn.addEventListener('click', () => {
      if (it.imageUrl) window.open(it.imageUrl, '_blank');
    });

    const removeBtn = el('button', { text: 'Remove' });
    removeBtn.addEventListener('click', async () => {
      await removeItem(it.id);
      await refresh();
    });

    const actions = el('div', { class: 'actions' }, [openBtn, removeBtn]);
    const card = el('div', { class: 'card' }, [img, meta, confWrap, actions]);
    list.appendChild(card);
  });
}

async function refresh() {
  const items = await loadItems();
  render(items);
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('refresh').addEventListener('click', refresh);
  document.getElementById('clear').addEventListener('click', async () => {
    await clearAll();
    await refresh();
  });
  refresh();
});
