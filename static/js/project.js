(() => {
  'use strict';
  const data = window.PAPER_DATA;
  if (!data) return;
  const select = (selector) => document.querySelector(selector);
  const selectAll = (selector) => [...document.querySelectorAll(selector)];
  selectAll('.highlight-number').forEach((element, index) => {
    const table = data.tables[index === 0 ? 'ade' : 'coco'];
    const ours = table.rows.find((row) => row.name === 'S²-RL').mean[0];
    const baseline = table.rows.find((row) => row.name === 'SD3.5').mean[0];
    element.textContent = `+${(ours - baseline).toFixed(2)}`;
  });
  const setActive = (selector, attribute, value) => selectAll(selector).forEach((button) => {
    const active = button.dataset[attribute] === value;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });

  let currentScene = 'bridge';
  let currentView = 'images';
  const renderComparison = () => {
    const scene = data.comparisons.find((item) => item.id === currentScene);
    const masks = currentView === 'masks';
    const baseline = select('#comparison-baseline');
    const ours = select('#comparison-ours');
    baseline.src = masks ? scene.baselineMask : scene.baseline;
    ours.src = masks ? scene.oursMask : scene.ours;
    baseline.alt = `SD3.5 ${scene.name.toLowerCase()} ${masks ? 'predicted segmentation map' : 'generated image'}`;
    ours.alt = `S squared RL ${scene.name.toLowerCase()} ${masks ? 'predicted segmentation map' : 'generated image'}`;
    select('#comparison-prompt').textContent = scene.prompt;
    select('#comparison-note').textContent = masks
      ? 'Predicted segmentation maps from Figure 4. These are not ground-truth annotations.'
      : 'Same text conditioning, different generated scenes. Examples from Figure 4.';
    setActive('[data-scene]', 'scene', currentScene);
    setActive('[data-view]', 'view', currentView);
  };
  selectAll('[data-scene]').forEach((button) => button.addEventListener('click', () => {
    currentScene = button.dataset.scene;
    renderComparison();
  }));
  selectAll('[data-view]').forEach((button) => button.addEventListener('click', () => {
    currentView = button.dataset.view;
    renderComparison();
  }));

  select('#hero-mask-toggle').addEventListener('change', (event) => {
    selectAll('.hero-scene img').forEach((img, index) => {
      const scene = data.comparisons[index];
      img.src = event.target.checked ? scene.oursMask : scene.ours;
      img.alt = `S squared RL ${scene.name.toLowerCase()} ${event.target.checked ? 'predicted segmentation map' : 'generated image'}`;
    });
  });
  selectAll('.hero-scene').forEach((link, index) => link.addEventListener('click', () => {
    currentScene = data.comparisons[index].id;
    currentView = select('#hero-mask-toggle').checked ? 'masks' : 'images';
    renderComparison();
  }));

  selectAll('[data-tsne]').forEach((button) => button.addEventListener('click', () => {
    const category = button.dataset.tsne;
    select('#tsne-image').src = `static/images/figures/tsne-${category}.webp`;
    select('#tsne-image').alt = `${category} class t-SNE: real data as gray dots, uniform selection as blue triangles, S squared RL as red stars`;
    setActive('[data-tsne]', 'tsne', category);
  }));

  const format = (value) => value.toFixed(2);
  const formatRatio = (value) => `${value === 1 ? '1.0' : value}%`;
  const tableTarget = select('#results-tables');
  for (const dataset of Object.values(data.tables)) {
    for (const metric of ['mean', 'macc']) {
      const stdKey = metric === 'mean' ? 'std' : 'maccStd';
      const metricLabel = metric === 'mean' ? 'mIoU' : 'mAcc';
      const wrapper = document.createElement('div');
      wrapper.className = 'table-wrapper';
      const table = document.createElement('table');
      const caption = document.createElement('caption');
      caption.textContent = `${dataset.name} · ${metricLabel} (%)`;
      table.append(caption);
      const header = table.createTHead().insertRow();
      ['Method', ...dataset.ratios.map(formatRatio)].forEach((name) => {
        const th = document.createElement('th');
        th.scope = 'col';
        th.textContent = name;
        header.append(th);
      });
      const body = table.createTBody();
      dataset.rows.forEach((method) => {
        const row = body.insertRow();
        if (method.name === 'S²-RL') row.className = 'our-row';
        const th = document.createElement('th');
        th.scope = 'row';
        th.textContent = method.name;
        row.append(th);
        method[metric].forEach((value, index) => {
          row.insertCell().textContent = `${format(value)} ± ${format(method[stdKey][index])}`;
        });
      });
      const full = body.insertRow();
      full.className = 'whole-row';
      full.insertCell().textContent = 'Full real dataset';
      const valueCell = full.insertCell();
      valueCell.colSpan = 3;
      valueCell.textContent = format(metric === 'mean' ? dataset.fullMiou : dataset.fullMacc);
      wrapper.append(table);
      tableTarget.append(wrapper);
    }
  }

  const ablationTarget = select('#ablation-bars');
  data.ablation.forEach((configuration) => {
    const row = document.createElement('div');
    row.className = 'ablation-row';
    const name = document.createElement('span');
    name.className = 'ablation-name';
    name.textContent = configuration.name;
    const track = document.createElement('div');
    track.className = 'ablation-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div');
    fill.className = 'ablation-fill';
    track.append(fill);
    const value = document.createElement('span');
    value.className = 'ablation-value';
    row.append(name, track, value);
    ablationTarget.append(row);
  });
  const renderAblation = () => {
    const budget = Number(select('#ablation-budget').value);
    // Common zero baseline and fixed 30 mIoU maximum across all budget choices.
    selectAll('.ablation-row').forEach((row, index) => {
      const item = data.ablation[index];
      row.querySelector('.ablation-fill').style.width = `${item.mean[budget] / 30 * 100}%`;
      const value = row.querySelector('.ablation-value');
      value.replaceChildren(document.createTextNode(format(item.mean[budget])));
      const error = document.createElement('span');
      error.textContent = ` ± ${format(item.std[budget])}`;
      value.append(error);
    });
  };
  select('#ablation-budget').addEventListener('change', renderAblation);
  renderAblation();

  select('#copy-bibtex').addEventListener('click', async () => {
    const code = select('#bibtex').textContent;
    try {
      await navigator.clipboard.writeText(code);
      select('#copy-status').textContent = 'Citation copied to clipboard.';
      select('#copy-bibtex').textContent = 'Copied';
      window.setTimeout(() => { select('#copy-bibtex').textContent = 'Copy BibTeX'; }, 2200);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(select('#bibtex'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      select('#copy-status').textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
})();
