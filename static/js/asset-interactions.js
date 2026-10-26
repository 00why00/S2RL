(() => {
  'use strict';
  const data = window.ASSET_DATA;
  if (!data) return;
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const active = (attribute, value) => $$(`[data-${attribute}]`).forEach((button) => {
    const on = button.getAttribute(`data-${attribute}`) === value;
    button.classList.toggle('active', on);
    button.setAttribute('aria-pressed', String(on));
  });
  const option = (value, label) => {
    const element = document.createElement('option');
    element.value = value;
    element.textContent = label;
    return element;
  };
  const legend = (target, classes) => {
    target.replaceChildren(...classes.map((item) => {
      const chip = document.createElement('span');
      const swatch = document.createElement('i');
      swatch.style.background = item.color;
      swatch.setAttribute('aria-hidden', 'true');
      chip.append(swatch, document.createTextNode(`${item.name} · ${(item.area * 100).toFixed(1)}%`));
      return chip;
    }));
  };

  let gallery = 'ade';
  let budgetIndex = 0;
  let view = 'rgb';
  let visible = 24;
  let dialogIndex = 0;
  let dialogView = 'rgb';
  const dialog = $('#image-dialog');
  const budget = () => data.datasets[gallery].budgets[budgetIndex];

  const renderDialogView = () => {
    const mask = $('#dialog-mask');
    mask.hidden = dialogView === 'rgb';
    mask.style.opacity = dialogView === 'mask' ? '1' : String(Number($('#dialog-opacity').value) / 100);
    $('#dialog-opacity-control').hidden = dialogView !== 'overlay';
    $('#dialog-opacity-value').textContent = `${$('#dialog-opacity').value}%`;
    active('dialog-view', dialogView);
  };
  const renderDialog = () => {
    const sample = budget().samples[dialogIndex];
    $('#dialog-image').src = sample.image;
    $('#dialog-image').alt = `${data.datasets[gallery].name} generated image ${sample.id}, ${budget().label} data budget`;
    $('#dialog-mask').src = sample.mask;
    $('#dialog-mask').alt = 'Teacher-generated semantic pseudo-label for this image';
    $('#dialog-caption').textContent = `${data.datasets[gallery].name} · ${budget().label} · Sample ${sample.id} · 1024 × 1024`;
    $('#dialog-position').textContent = `${dialogIndex + 1} / ${budget().count}`;
    legend($('#dialog-legend'), sample.classes);
    renderDialogView();
  };
  const openSample = (index) => {
    dialogIndex = index;
    dialogView = view;
    renderDialog();
    dialog.showModal();
  };
  const renderGallery = () => {
    const selected = budget();
    const number = Math.min(selected.count, visible);
    const fragment = document.createDocumentFragment();
    selected.samples.slice(0, number).forEach((sample, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'sample-card';
      button.setAttribute('aria-label', `Inspect ${data.datasets[gallery].name} sample ${sample.id}, ${selected.label}`);
      const rgb = document.createElement('img');
      rgb.src = sample.thumb;
      rgb.width = 320;
      rgb.height = 320;
      rgb.loading = 'lazy';
      rgb.alt = `${data.datasets[gallery].name} generated sample ${sample.id}`;
      const mask = document.createElement('img');
      if (view !== 'rgb') mask.src = sample.maskThumb;
      mask.width = 320;
      mask.height = 320;
      mask.loading = 'lazy';
      mask.alt = '';
      mask.className = 'sample-mask';
      mask.hidden = view === 'rgb';
      mask.style.opacity = view === 'mask' ? '1' : '.45';
      const label = document.createElement('span');
      label.className = 'sample-index';
      label.textContent = String(sample.id).padStart(3, '0');
      button.append(rgb, mask, label);
      button.addEventListener('click', () => openSample(index));
      fragment.append(button);
    });
    $('#sample-grid').replaceChildren(fragment);
    $('#gallery-status').textContent = `${selected.count} image–label pairs · ${selected.label} data budget`;
    $('#gallery-count').textContent = `Showing ${number} of ${selected.count}`;
    $('#load-more').hidden = number === selected.count;
    $('#gallery-protocol').textContent = `Pseudo-label teacher: ${data.datasets[gallery].teacher}. Labels are model predictions on synthetic training images. All samples in this stored set are included; the same teacher labels the synthetic baselines.`;
    active('gallery', gallery);
    active('gallery-view', view);
  };
  const renderBudgets = () => {
    $('#gallery-budget').replaceChildren(...data.datasets[gallery].budgets.map((item, index) => option(index, `${item.label} · ${item.count} images`)));
    $('#gallery-budget').value = String(budgetIndex);
  };
  $$('[data-gallery]').forEach((button) => button.addEventListener('click', () => {
    gallery = button.dataset.gallery;
    visible = 24;
    renderBudgets();
    renderGallery();
  }));
  $('#gallery-budget').addEventListener('change', (event) => {
    budgetIndex = Number(event.target.value);
    visible = 24;
    renderGallery();
  });
  $$('[data-gallery-view]').forEach((button) => button.addEventListener('click', () => {
    view = button.dataset.galleryView;
    renderGallery();
  }));
  $('#load-more').addEventListener('click', () => { visible += 24; renderGallery(); });
  $$('[data-dialog-view]').forEach((button) => button.addEventListener('click', () => {
    dialogView = button.dataset.dialogView;
    renderDialogView();
  }));
  $('#dialog-opacity').addEventListener('input', renderDialogView);
  $('#close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  const move = (delta) => {
    dialogIndex = (dialogIndex + delta + budget().count) % budget().count;
    renderDialog();
  };
  $('#prev-image').addEventListener('click', () => move(-1));
  $('#next-image').addEventListener('click', () => move(1));
  dialog.addEventListener('keydown', (event) => {
    if (event.target.matches('input,select')) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
  });
  renderBudgets();
  renderGallery();

  let evidenceDataset = 'ade';
  let metric = 'occupancy';
  let exampleIndex = 0;
  const currentEvidence = () => data.semantic[evidenceDataset];
  const currentExample = () => currentEvidence().examples[exampleIndex];
  const takeaways = {
    ade: {
      occupancy: 'S²-RL covers more class-occupancy configurations than the compared methods; its precision is comparable to Uniform.',
      pairs: 'S²-RL recovers more category pairs, with lower pair precision. Broader coverage comes with a trade-off.',
      geometry: 'S²-RL has the highest pair-macro coverage and precision for the measured relative-geometry descriptors.'
    },
    coco: {
      occupancy: 'S²-RL improves occupancy coverage over the synthetic baselines. Uniform remains slightly ahead in coverage.',
      pairs: 'S²-RL has the highest category-pair recall, with lower precision than the compared methods.',
      geometry: 'S²-RL has the highest pair-macro coverage and precision for the measured relative-geometry descriptors.'
    }
  };
  const renderMetric = () => {
    const item = currentEvidence().metrics.find((entry) => entry.id === metric);
    $('#metric-title').textContent = item.name;
    $('#metric-description').textContent = item.description;
    $('#semantic-chart').src = item.chart;
    $('#semantic-chart').alt = `${currentEvidence().name} ${currentEvidence().budget}: ${item.name}, ${item.firstLabel.toLowerCase()} and precision for Uniform, SD3.5, UniDD* and S²-RL, with sample standard deviations.`;
    $('#metric-takeaway').textContent = takeaways[evidenceDataset][metric];
    const table = document.createElement('table');
    const caption = document.createElement('caption');
    caption.textContent = `${currentEvidence().name} · ${currentEvidence().budget} · ${item.name}`;
    table.append(caption);
    const header = table.createTHead().insertRow();
    ['Method', item.firstLabel, 'Precision'].forEach((name) => {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = name; header.append(th);
    });
    const tbody = table.createTBody();
    item.rows.forEach((entry) => {
      const row = tbody.insertRow();
      if (entry.method === 'S²-RL') row.className = 'our-row';
      const th = document.createElement('th'); th.scope = 'row'; th.textContent = entry.method; row.append(th);
      ['first', 'second'].forEach((field) => { row.insertCell().textContent = `${entry[field].mean.toFixed(3)} ± ${entry[field].sd.toFixed(3)}`; });
    });
    $('#semantic-table').replaceChildren(table);
    active('evidence', evidenceDataset);
    active('metric', metric);
  };

  const heatmap = (side, item) => {
    $(`#occupancy-name-${side}`).textContent = `${side.toUpperCase()} · ${item.name}`;
    $(`#occupancy-${side}`).replaceChildren(...item.occupancy.map((value, index) => {
      const cell = document.createElement('span');
      // Values are fractions of the whole image, so a full 4x4 cell is 1/16.
      const strength = Math.min(1, value * 16);
      cell.style.background = `color-mix(in srgb, ${item.color} ${Math.round(strength * 80 + 5)}%, white)`;
      cell.textContent = value === 0 ? '·' : `${(value * 100).toFixed(1)}`;
      cell.title = `Row ${Math.floor(index / 4) + 1}, column ${index % 4 + 1}: ${(value * 100).toFixed(2)}% of the image`;
      return cell;
    }));
  };
  const renderDescriptors = () => {
    const classes = currentExample().classes;
    const first = classes.find((item) => item.id === Number($('#layout-class-a').value));
    const second = classes.find((item) => item.id === Number($('#layout-class-b').value));
    if (!first || !second) return;
    heatmap('a', first); heatmap('b', second);
    [first, second].forEach((item, index) => {
      const side = index ? 'b' : 'a';
      const [x, y] = item.centroid.map((value) => value * 100);
      const dot = $(`#centroid-${side}`); dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('fill', item.color);
      const text = $(`#centroid-label-${side}`); text.setAttribute('x', Math.min(94, x + 3.6)); text.setAttribute('y', Math.max(7, y - 2.5));
    });
    const line = $('#centroid-line');
    line.setAttribute('x1', first.centroid[0] * 100); line.setAttribute('y1', first.centroid[1] * 100);
    line.setAttribute('x2', second.centroid[0] * 100); line.setAttribute('y2', second.centroid[1] * 100);
    const dx = second.centroid[0] - first.centroid[0];
    const dy = second.centroid[1] - first.centroid[1];
    const signed = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
    const items = [
      ['Category pair', `${first.name} + ${second.name}`],
      ['Horizontal offset · B − A', signed(dx)],
      ['Vertical offset · B − A', signed(dy)],
      ['Area ratio · A / B', `${(first.area / second.area).toFixed(2)}×`]
    ];
    $('#geometry-readout').replaceChildren(...items.map(([name, value]) => {
      const row = document.createElement('div'); const label = document.createElement('span'); const result = document.createElement('strong');
      label.textContent = name; result.textContent = value; row.append(label, result); return row;
    }));
  };
  const renderExample = () => {
    const item = currentExample();
    $('#layout-rgb').src = item.image;
    $('#layout-rgb').alt = `${currentEvidence().name} evaluated S²-RL sample ${exampleIndex + 1}`;
    $('#layout-mask').src = item.mask;
    $('#layout-mask').alt = 'Independent OneFormer semantic annotation for this generated image';
    for (const side of ['a', 'b']) $(`#layout-class-${side}`).replaceChildren(...item.classes.map((entry) => option(entry.id, entry.name)));
    $('#layout-class-a').value = String(item.classes[0].id);
    $('#layout-class-b').value = String(item.classes[Math.min(1, item.classes.length - 1)].id);
    renderDescriptors();
  };
  $$('[data-evidence]').forEach((button) => button.addEventListener('click', () => {
    evidenceDataset = button.dataset.evidence;
    renderMetric(); renderExample();
  }));
  $$('[data-metric]').forEach((button) => button.addEventListener('click', () => { metric = button.dataset.metric; renderMetric(); }));
  $('#layout-example').addEventListener('change', (event) => { exampleIndex = Number(event.target.value); renderExample(); });
  for (const side of ['a', 'b']) $(`#layout-class-${side}`).addEventListener('change', () => {
    const other = side === 'a' ? 'b' : 'a';
    if ($('#layout-class-a').value === $('#layout-class-b').value) {
      const replacement = currentExample().classes.find((entry) => entry.id !== Number($(`#layout-class-${side}`).value));
      if (replacement) $(`#layout-class-${other}`).value = String(replacement.id);
    }
    renderDescriptors();
  });
  $('#layout-opacity').addEventListener('input', (event) => {
    $('#layout-mask').style.opacity = String(Number(event.target.value) / 100);
    $('#layout-opacity-value').textContent = `${event.target.value}%`;
  });
  renderMetric(); renderExample();
})();
