/**
 * Dashboard frontend logic for InverBot.
 * ------------------------------------------------------------
 * Este archivo maneja la interacción con la API `/api/*` para
 * cargar predicciones de precios, renderizar gráficas con Chart.js,
 * descargar bases de datos y mostrar métricas de rendimiento
 * (MAE, RMSE, MAPE, R²) junto a la próxima predicción horaria.
 *
 * Todas las funciones están documentadas con bloques JSDoc en español
 * para facilitar el mantenimiento y la comprensión del código.
 */
/* ---------- util ---------- */
/**
 * Hace una petición `fetch` y devuelve la respuesta parseada como JSON.
 * Lanza un `Error` con el `statusText` cuando la respuesta no es `ok`.
 *
 * @param {string} url - Ruta absoluta o relativa a la API.
 * @returns {Promise<any>} - Objeto JSON de la respuesta.
 */
async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Fetch ${url} → ${r.statusText}`);
    return r.json();
  }
  
  /* ---------- elementos DOM ---------- */
  const tickerSelect   = document.getElementById('tickerSelect');   // <select>
  const datasetSelect  = document.getElementById('datasetSelect');  // <select>
  const updateBtn      = document.getElementById('updateBtn');      // <button>
  const downloadDbBtn  = document.getElementById('downloadDbBtn');  // <a>
  const downloadUpdatedBtn = document.getElementById('downloadUpdatedBtn'); // <a> nuevo
  if (!downloadUpdatedBtn) {
    console.warn('downloadUpdatedBtn not found in DOM');
  }
// Puede que en simulacion.html no exista #chartCanvas.
const chartCanvasElem = document.getElementById('chartCanvas');
// Sólo obtenemos el contexto si el canvas existe.
const canvasCtx = chartCanvasElem ? chartCanvasElem.getContext('2d') : null;
  
  // Canvas exclusivo para la simulación (sólo existe en simulacion.html)
  const simCanvas = document.getElementById('simChartCanvas');
  let   simChart  = null;
  /* ---------- tooltips ---------- */
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(t => new bootstrap.Tooltip(t));
  
  let chart;  // referencia global al Chart.js
  
// === series globales para la simulación ===
let gLabels = [];
let gReal   = [];
let gPred   = [];

  /* ---------- carga inicial de tickers ---------- */
/**
 * Obtiene el listado de tickers disponibles desde `/api/models`
 * y llena el `<select id="tickerSelect">` con ellos.
 * También inicializa el botón de descarga de la BD del último mes
 * y su listener una vez que los tickers están listos.
 */
  async function loadTickers() {
  const tickers = await fetchJSON('/api/models');               // ['AAPL','AMZN',...]
  tickerSelect.innerHTML = tickers
    .map(t => `<option value="${t}">${t}</option>`)
    .join('');
  refreshUpdatedHref();   // ensure href populated once tickers are loaded
  initUpdatedBtn();      // ensure button listener exists once tickers are ready
  }


  function createSimChart(labels) {
    if (!simCanvas) return;
    if (simChart) simChart.destroy();

    simChart = new Chart(simCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label:'Precio Real', data:[], borderColor:'#ffffff',
            backgroundColor:'#ffffff', pointRadius:1, tension:0.4, fill:false },
          { label:'Predicción IA', data:[], borderColor:'#60a5fa',
            borderDash:[4,4], pointRadius:0, tension:0.4, fill:false }
        ]
      },
      options: {
        responsive:true,
        plugins:{
          legend:{labels:{color:'#ffffff'}},
          tooltip:{
            mode:'index',
            intersect:false,
            filter: ctx => !['Compra','Venta','Punto Real','Punto Pred'].includes(ctx.dataset.label)
          }
        },
        scales:{
          x:{ticks:{color:'#8b949e'},grid:{color:'#1f2933'}},
          y:{ticks:{color:'#8b949e'},grid:{color:'#1f2933'}}
        }
      }
    });
  }
  
  /* ---------- carga (o recarga) de datos y gráfica ---------- */
/**
 * Descarga precios reales y predicciones para el ticker y dataset
 * actualmente seleccionados, renderiza el gráfico y actualiza el
 * enlace de descarga de la base de datos.
 *
 * Extrae `labels`, `real`, `pred`, métricas y la próxima predicción
 * a partir de la respuesta JSON de `/api/predict/:ticker/:ds`.
 */
  async function loadData() {
    // Si había una simulación corriendo, deténla y limpia datasets
    if (sim.running) {
      clearInterval(sim.timer);
      sim.running = false;
    }
    sim.buyDs = null;
    sim.sellDs = null;

    const ticker  = tickerSelect.value;          // p.ej. 'AMZN'
    const ds      = datasetSelect.value;         // '2y' o '1mo'

    const resp    = await fetchJSON(`/api/predict/${ticker}/${ds}`);
    const labels  = resp.data.map(d => d.date);
    const real    = resp.data.map(d => d.real);
    const pred    = resp.data.map(d => d.pred);
    const metrics = resp.metrics;
    const nextPred = resp.next_pred;


    // 🔑 Actualiza las globales para la simulación
    gLabels = labels;
    gReal   = real;
    gPred   = pred;

    // destruye gráfico anterior si existe
    if (chart) chart.destroy();

    chart = new Chart(canvasCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Precio Real',
            data: real,
            borderColor: '#ffffff',
            backgroundColor: '#ffffff',
            borderWidth: 1,
            pointRadius: 1,
            tension: 0.4,
            fill: false
          },
          {
            label: 'Predicción IA',
            data: pred,
            borderColor: '#60a5fa',
            backgroundColor: '#60a5fa',
            borderDash: [4, 4],
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.4,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: '#ffffff' } },
          tooltip: {
            mode: 'index',
            intersect: false,
            filter: (ctx) => !['Compra', 'Venta'].includes(ctx.dataset.label)
          }
        },
        scales: {
          x: { type: 'category',ticks: { color: '#8b949e' }, grid: { color: '#1f2933' } },
          y: { ticks: { color: '#8b949e' }, grid: { color: '#1f2933' } }
        }
      }
    });

    // Fuerza el resize robusto del gráfico
    ensureChartResize();

    // actualiza link de descarga de BD
    if (downloadDbBtn) {
      downloadDbBtn.href = `/download/db/${ticker}/${ds}`;
    }

    // keep updated‑db link in sync with current ticker / ds
    refreshUpdatedHref();

    renderMetrics(metrics, nextPred, real);
  }
  
/* ---------- carga (o recarga) de datos PARA SIMULACIÓN ---------- */
async function loadDataSim() {
  if (sim.running) { clearInterval(sim.timer); sim.running = false; }
  sim.buyDs = null;
  sim.sellDs = null;

  const ticker = tickerSelect.value;
  const ds     = datasetSelect.value;

  const resp   = await fetchJSON(`/api/predict-sim/${ticker}/${ds}`);
  const labels = resp.data.map(d => d.date);
  const real   = resp.data.map(d => d.real);
  const pred   = resp.data.map(d => d.pred);
  const metrics  = resp.metrics;
  const nextPred = resp.next_pred;

  gLabels = labels;
  gReal   = real;
  gPred   = pred;

  // Si existe chartCanvas (otras páginas reutilizan la lógica),
  // dibujamos también el gráfico blanco‑azul.  En simulacion.html
  // chartCanvas no existe y se omite sin lanzar error.
  if (canvasCtx) {
    if (chart) chart.destroy();
    chart = new Chart(canvasCtx, {
      type: 'line',
      data: { labels,
        datasets: [
          { label:'Precio Real', data:real,
            borderColor:'#ffffff', backgroundColor:'#ffffff',
            pointRadius:1, tension:0.4, fill:false },
          { label:'Predicción IA', data:pred,
            borderColor:'#60a5fa', backgroundColor:'#60a5fa',
            borderDash:[4,4], pointRadius:0, tension:0.4, fill:false }
        ] },
      options: {
        responsive:true,
        plugins:{
          legend:{labels:{color:'#ffffff'}},
          tooltip:{
            mode:'index',
            intersect:false,
            filter: ctx => !['Compra','Venta','Punto Real','Punto Pred'].includes(ctx.dataset.label)
          }
        },
        scales:{
          x:{ticks:{color:'#8b949e'},grid:{color:'#1f2933'}},
          y:{ticks:{color:'#8b949e'},grid:{color:'#1f2933'}}
        }
      }
    });
  }

  ensureChartResize();
  refreshUpdatedHref();
  renderMetrics(metrics, nextPred, real);

  if (simCanvas) createSimChart(labels);   // gráfica negra vacía
  // En simulacion.html el gráfico “oficial” será simChart
  if (!canvasCtx && simCanvas) {
    chart = simChart;
  }
}

/* Utilidad: fuerza re‑dimensionado del gráfico (evita “aplastado”) */
function ensureChartResize() {
  if (!chart) return;
  chart.resize();
  // segundo intento tras 200 ms por si los estilos aún no aplican
  setTimeout(() => chart && chart.resize(), 200);
}

  /* ---------- listeners ---------- */

/* Utilidad: decide si estamos en la página de simulación o no */
function reloadData() {
  return simCanvas ? loadDataSim() : loadData();
}

document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadTickers();
    if (simCanvas) {        // página de simulación
      await loadDataSim();
    } else {
      await loadData();     // resto de páginas
    }      // dibuja primer gráfico
    initUpdatedBtn();      // <‑‑ attach listener for updated‑month button
  } catch (err) {
    console.error(err);
    alert('Error cargando datos iniciales');
  }
});

// Asegura tamaño correcto cuando toda la página (CSS, fonts) terminó de cargar
window.addEventListener('load', ensureChartResize);


/* Utilidad: asigna textContent sólo si el elemento existe */
function safeText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/**
 * Muestra las métricas (MAE, RMSE, MAPE, R²) y la variación esperada
 * del siguiente periodo en las cards del dashboard.
 *
 * @param {Object} metrics - Métricas de evaluación.
 * @param {number|null} nextPred - Próxima predicción horaria.
 * @param {number[]} realArr - Serie de precios reales.
 */
function renderMetrics(metrics, nextPred, realArr) {
  if (!metrics) return;

  // MAE y RMSE en dólares, R² en %
  safeText('maeVal',  metrics.mae  != null ? `$${metrics.mae.toFixed(2)}`  : '--');
  safeText('rmseVal', metrics.rmse != null ? `$${metrics.rmse.toFixed(2)}` : '--');
  safeText('mapeVal', metrics.mape != null ? `${metrics.mape.toFixed(2)} %` : '--');
  safeText('r2Val',   metrics.r2   != null ? `${(metrics.r2 * 100).toFixed(2)} %` : '--');

  const nextEl = document.getElementById('nextPredVal');
  const dirEl  = document.getElementById('nextDirection');

  if (!nextEl || !dirEl) return;   // si no existen, termina

  if (nextPred == null || !realArr?.length) {
    nextEl.textContent = '--';
    dirEl.innerHTML = '';
    dirEl.className  = '';
    return;
  }

  const lastReal = realArr[realArr.length - 2];
  nextEl.textContent = `$${nextPred.toFixed(2)}`;

  const diff = ((nextPred - lastReal) / lastReal) * 100;
  const up   = diff >= 0;
  dirEl.innerHTML = `${up ? '▲' : '▼'} ${diff.toFixed(2)} %`;
  dirEl.className = up ? 'text-success' : 'text-danger';
}

/* ---------- init listener for updated-month button ---------- */
/**
 * Asigna, sólo una vez, el listener al botón
 * "BD último mes (preds)". Gestiona el flujo completo:
 *   1. Llama `/api/predict-updated/:ticker` y dibuja la nueva gráfica.
 *   2. Llama `renderMetrics` con los nuevos datos.
 *   3. Dispara la descarga automática del CSV actualizado.
 * Incluye feedback de carga con un spinner para mejorar UX.
 */
function initUpdatedBtn() {
  const btn = document.getElementById('downloadUpdatedBtn');
  if (!btn || btn.dataset.listenerAttached) return;   // evita duplicados

  btn.addEventListener('click', async (e) => {
    e.preventDefault();

    const ticker = tickerSelect.value;
    if (!ticker) return;

    // loading UI
    btn.disabled = true;
    const originalHTML = btn.innerHTML;
    btn.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Generando...';

    try {
      // 1️⃣ pide las predicciones (tarda)
      const resp   = await fetchJSON(`/api/predict-updated/${ticker}`);
      let labels   = resp.data.map(d => d.date);
      let real     = resp.data.map(d => d.real);
      let pred     = resp.data.map(d => d.pred);

      // ── Página de simulación: dibujar en simChart ──
      if (simCanvas) {
        // Destruye gráfica negra previa y recrea
        createSimChart(labels);
        chart = simChart;

        // Rellena datasets
        chart.data.datasets[0].data = real.map((y, i) => ({ x: labels[i], y }));
        chart.data.datasets[1].data = pred.map((y, i) => ({ x: labels[i], y }));
        attachTradeDatasets();
        attachPointDatasets();
      }
      // ── Página normal (dashboard) ──
      else {
        if (chart) chart.destroy();
        chart = new Chart(canvasCtx, {
          type: 'line',
          data: {
            labels,
            datasets: [
              { label:'Precio Real', data:real, borderColor:'#ffffff',
                backgroundColor:'#ffffff', pointRadius:3, tension:0.4, fill:false },
              { label:'Predicción IA', data:pred, borderColor:'#60a5fa',
                borderDash:[6,6], pointRadius:0, tension:0.4, fill:false }
            ]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { labels: { color: '#ffffff' } },
              tooltip: {
                mode: 'index',
                intersect: false,
                filter: (ctx) => !['Compra', 'Venta'].includes(ctx.dataset.label)
              }
            },
            scales: {
              x: { ticks: { color: '#8b949e' }, grid: { color: '#1f2933' } },
              y: { ticks: { color: '#8b949e' }, grid: { color: '#1f2933' } }
            }
          }
        });
      }

      const metrics  = resp.metrics;
      const nextPred = resp.next_pred;

      gLabels = labels;
      gReal   = real;
      gPred   = pred;

      renderMetrics(metrics, nextPred, real);

      // 3️⃣ dispara descarga del CSV
      const dl = document.createElement('a');
      dl.href = `/download/updated-db/${ticker}`;
      dl.download = '';
      document.body.appendChild(dl);
      dl.click();
      dl.remove();

    } catch (err) {
      console.error(err);
      alert('Error generando la predicción del último mes');
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalHTML;
    }
  });

  btn.dataset.listenerAttached = 'true';   // marca que ya tiene listener
}

updateBtn.addEventListener('click', reloadData);
datasetSelect.addEventListener('change', () => {
  reloadData();
  refreshUpdatedHref();
});
  
  /* ---------- “BD último mes (preds)” link logic ---------- */

/**
 * Mantiene sincronizado el atributo `href` del botón de descarga de la
 * BD actualizada con el ticker y dataset seleccionados.  Evita enlaces
 * rotos cuando el usuario cambia de ticker/dataset sin recargar la página.
 */
function refreshUpdatedHref() {
    const btn = document.getElementById('downloadUpdatedBtn');
    if (!btn) return;
    const ticker = tickerSelect.value;
    btn.href = ticker ? `/download/updated-db/${ticker}` : '#';
}

// Set on load + on ticker change
document.addEventListener('DOMContentLoaded', refreshUpdatedHref);
if (tickerSelect) tickerSelect.addEventListener('change', refreshUpdatedHref);

// Si hay un listener sobre tickerSelect que llama loadData, actualízalo para usar reloadData
// (opcional coherencia, por si en algún lugar hay un .addEventListener sobre tickerSelect con loadData)
// En este archivo, sólo refreshUpdatedHref está en tickerSelect, así que nada más que hacer aquí.







/* ========================= SIMULACIÓN (Regla básica) ========================= */

/* --- Estado interno de la simulación --- */
const sim = {
  running: false,
  timer: null,
  idx: 0,          // posición actual en la serie (1 = segunda fila)
  cash: 0,         // efectivo disponible
  shares: 0,       // acciones poseídas
  buyDs: null,     // dataset Chart.js para compras
  sellDs: null,    // dataset Chart.js para ventas
  realPointDs: null,   // nuevo: puntos blancos de precio real
  predPointDs: null,   // nuevo: puntos azules de predicción
  lastTradePrice: null 
};

/* Agrega (o limpia) datasets de compras/ventas al gráfico */
function attachTradeDatasets() {
  // si el gráfico fue recreado, los datasets previos ya no existen
  if (sim.buyDs && chart.data.datasets.includes(sim.buyDs)) {
    sim.buyDs.data.length = 0;
    sim.sellDs.data.length = 0;
    return;
  }
  sim.buyDs = {
    type: 'scatter',
    label: 'Compra',
    data: [],
    backgroundColor: '#16a34a',
    pointStyle: 'rectRounded',
    radius: 6,
    showLine: false
  };
  sim.sellDs = {
    type: 'scatter',
    label: 'Venta',
    data: [],
    backgroundColor: '#dc2626',
    pointStyle: 'rectRounded',
    radius: 6,
    showLine: false
  };
  chart.data.datasets.push(sim.buyDs, sim.sellDs);
}

/* Agrega (o limpia) datasets de puntos en cada paso (real y pred) */
function attachPointDatasets() {
  if (sim.realPointDs && chart.data.datasets.includes(sim.realPointDs)) {
    sim.realPointDs.data.length = 0;
    sim.predPointDs.data.length = 0;
    return;
  }
  sim.realPointDs = {
    type: 'scatter',
    label: 'Punto Real',
    data: [],
    backgroundColor: '#ffffff',
    borderColor: '#ffffff',
    radius: 3,
    showLine: false,
    pointStyle: 'circle'
  };
  sim.predPointDs = {
    type: 'scatter',
    label: 'Punto Pred',
    data: [],
    backgroundColor: '#60a5fa',
    borderColor: '#60a5fa',
    radius: 3,
    showLine: false,
    pointStyle: 'circle'
  };
  chart.data.datasets.push(sim.realPointDs, sim.predPointDs);
}

/* Actualiza el panel de estado (efectivo, acciones, equity) */
function updateSimPanel(price) {
  const equity = sim.cash + sim.shares * price;
  document.getElementById('simCash').textContent   = `$${sim.cash.toFixed(2)}`;
  document.getElementById('simShares').textContent = sim.shares.toFixed(4);
  document.getElementById('simEquity').textContent = `$${equity.toFixed(2)}`;
}

/* Un paso de simulación = 1 hora en los datos */
function stepSimulation() {
  // Fin de datos
  if (sim.idx >= gReal.length-1) {
    clearInterval(sim.timer);
    sim.running = false;
    document.getElementById('startSimBtn').disabled = false;

    // --- Popup con resultado final -----------------------------
    const finalPrice = sim.lastPrice ?? gReal[gReal.length - 2];
    const finalEquity = sim.cash + sim.shares * finalPrice;
    const diff = finalEquity - sim.initEquity;
    const msg =
      `Capital inicial: $${sim.initEquity.toFixed(2)}\n` +
      `Capital final:   $${finalEquity.toFixed(2)}\n` +
      `Resultado: ${diff >= 0 ? 'ganancia' : 'pérdida'} de ` +
      `$${Math.abs(diff).toFixed(2)} (${((diff / sim.initEquity) * 100).toFixed(2)} %)`;
    alert(msg);   // Puedes reemplazar por un modal Bootstrap si prefieres
    return;
  }

  // Precio actual (real i-1) y predicción para hora i
  const priceNow  = gReal[sim.idx];
  sim.lastPrice = priceNow;   // guarda el último precio procesado
  const pricePred = gPred[sim.idx];
  if (simCanvas && chart === simChart) {
    // Línea: añade al array de la curva (para mantener trazado)
    chart.data.datasets[0].data.push({ x: gLabels[sim.idx],     y: priceNow });
    chart.data.datasets[1].data.push({ x: gLabels[sim.idx + 1], y: pricePred });

    // Puntos blancos y azules
    sim.realPointDs.data.push({ x: gLabels[sim.idx],     y: priceNow  });
    sim.predPointDs.data.push({ x: gLabels[sim.idx + 1], y: pricePred });
  }
  // --- Regla básica: comprar si la IA predice subida; vender si predice bajada ---
  const investCash = sim.cash * 0.20;      // 20 % del efectivo disponible
  const sellQty    = sim.shares * 0.20;    // 20 % de las acciones

  // COMPRA — la predicción es mayor que el precio actual
  if (pricePred >= priceNow && sim.cash > 0) {
    const qty = investCash / priceNow;
    sim.cash   -= investCash;
    sim.shares += qty;
    sim.buyDs.data.push({ x: gLabels[sim.idx ], y: priceNow });
  }
  // VENTA — la predicción es menor que el precio actual
  else if (pricePred < priceNow && sim.shares > 0) {
    const qty = sellQty;
    sim.shares -= qty;
    sim.cash   += qty * priceNow;
    sim.sellDs.data.push({ x: gLabels[sim.idx ], y: priceNow });
  }

/*  // --- Estrategia basada en último trade ---
  // umbrales (puedes ajustar)
  const buyThreshold  = 0.01;   // +7 % sobre último precio → comprar
  const sellThreshold = 0.01;   // −7 % bajo último precio → vender

  // referencia: último precio de compra/venta (o priceNow si aún no hay)
  const refPrice   = sim.lastTradePrice ?? priceNow;
  const pctFromRef = (refPrice - pricePred) / refPrice;  // diferencia %

  const investCash = sim.cash   * 0.5;   // 20 % efectivo
  const sellQty    = sim.shares * 0.5;   // 20 % acciones

  // COMPRA: predicción ≥ +buyThreshold y hay efectivo
  if (pctFromRef >= buyThreshold && sim.cash > 0) {
    const qty = investCash / priceNow;
    sim.cash   -= investCash;
    sim.shares += qty;
    sim.buyDs.data.push({ x: gLabels[sim.idx], y: priceNow });
    sim.lastTradePrice = priceNow;          // actualiza referencia
  }
  // VENTA: predicción ≤ −sellThreshold y hay acciones
  else if (pctFromRef <= -sellThreshold && sim.shares > 0) {
    const qty = sellQty;
    sim.shares -= qty;
    sim.cash   += qty * priceNow;
    sim.sellDs.data.push({ x: gLabels[sim.idx], y: priceNow });
    sim.lastTradePrice = priceNow;          // actualiza referencia
  }*/

  updateSimPanel(priceNow);
  chart.update('none');   // refresco ligero
  sim.idx += 1;
}


/* ---------- Botón “Comenzar simulación” ---------- */
document.getElementById('startSimBtn')?.addEventListener('click', () => {
  // Evita varias simulaciones simultáneas
  if (sim.running) return;

  // Asegura que haya datos cargados
  if (gReal.length === 0) {
    alert('Primero carga un ticker y dataset.');
    return;
  }

  // Lee valores iniciales del modal
  sim.cash   = parseFloat(document.getElementById('initCash').value)   || 0;
  if (simCanvas && simChart) {
    chart = simChart;    // las compras/ventas se dibujan en la gráfica negra
    // Limpia datos previos de la curva Real/Pred
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    attachPointDatasets();     // añade (o reinicia) puntos blancos/azules
  }
  sim.shares = parseFloat(document.getElementById('initShares').value) || 0;
  if (sim.cash < 0 || sim.shares < 0) return;

  // equity inicial (cash + valor de acciones al primer precio)
  sim.initEquity = sim.cash + sim.shares * gReal[0];

  sim.lastTradePrice = gReal[0];  // primer precio de referencia

  // Reinicia estado
  sim.idx = 0;
  attachTradeDatasets();

  document.getElementById('simPanel').classList.remove('d-none');
  updateSimPanel(gReal[0]);

  // Cierra modal
  bootstrap.Modal.getInstance(document.getElementById('simulateModal'))?.hide();

  // Inicia timer (1 s ≈ 1 h de datos)
  sim.running = true;
  document.getElementById('startSimBtn').disabled = true;
  sim.timer = setInterval(stepSimulation, 500);
});