const startBtn = document.getElementById("startBtn");
const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const exportBtn = document.getElementById("exportBtn");
const clearMeasureBtn = document.getElementById("clearMeasureBtn");
const mmPerPixelInput = document.getElementById("mmPerPixel");
const diagnosisList = document.getElementById("diagnosisList");
const reportBox = document.getElementById("reportBox");
const workLengthOutput = document.getElementById("workLengthOutput");

const sourceCanvas = document.getElementById("sourceCanvas");
const resultCanvas = document.getElementById("resultCanvas");
const sourceCtx = sourceCanvas.getContext("2d");
const resultCtx = resultCanvas.getContext("2d");

let imageReady = false;
let measurePoints = [];
let latestFindings = [];
let latestGray = null;

startBtn.addEventListener("click", () => {
  document.getElementById("workspace").scrollIntoView({ behavior: "smooth" });
});

imageInput.addEventListener("change", handleImageSelect);
analyzeBtn.addEventListener("click", runAnalysis);
resetBtn.addEventListener("click", resetAll);
exportBtn.addEventListener("click", exportResults);
clearMeasureBtn.addEventListener("click", clearMeasurePoints);
resultCanvas.addEventListener("click", captureMeasurePoint);

function handleImageSelect(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const image = new Image();
  image.onload = () => {
    drawImageContain(sourceCtx, image, sourceCanvas.width, sourceCanvas.height);
    drawImageContain(resultCtx, image, resultCanvas.width, resultCanvas.height);
    imageReady = true;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    exportBtn.disabled = true;
    latestGray = null;
    latestFindings = [];
    clearMeasurePoints();
    setDiagnosis(["تم تحميل الصورة بنجاح. اضغط زر المعالجة والتشخيص للمتابعة."], "ok");
    setReportPlaceholder();
  };
  image.src = URL.createObjectURL(file);
}

function runAnalysis() {
  if (!imageReady) return;

  const imageData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const enhanced = enhanceRadiograph(imageData);
  latestGray = enhanced.gray;

  const findings = detectDetailedFindings(enhanced.gray, enhanced.width, enhanced.height);
  latestFindings = findings;

  drawEnhancedWithOverlays(enhanced.gray, enhanced.width, enhanced.height, findings);
  renderFindings(findings);
  renderDetailedReport(findings, enhanced.quality);
  exportBtn.disabled = false;
  clearMeasurePoints();
}

function enhanceRadiograph(imageData) {
  const { data, width, height } = imageData;
  const gray = new Uint8ClampedArray(width * height);

  let min = 255;
  let max = 0;
  let sum = 0;

  for (let i = 0; i < data.length; i += 4) {
    const g = Math.round(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    const idx = i / 4;
    gray[idx] = g;
    min = Math.min(min, g);
    max = Math.max(max, g);
    sum += g;
  }

  const meanRaw = sum / gray.length;
  const span = Math.max(1, max - min);

  for (let i = 0; i < gray.length; i++) {
    const normalized = ((gray[i] - min) / span) * 255;
    gray[i] = Math.max(0, Math.min(255, Math.round((normalized - 128) * 1.32 + 128)));
  }

  const std = standardDeviation(gray);
  const quality = {
    dynamicRange: span,
    mean: meanRaw,
    contrast: std,
    qualityLabel:
      span < 70 || std < 28
        ? "جودة منخفضة نسبيًا (قد تؤثر على دقة القراءة)"
        : "جودة مقبولة للقراءة الأولية"
  };

  return { gray, width, height, quality };
}

function detectDetailedFindings(gray, width, height) {
  const lowThreshold = percentile(gray, 18);
  const strongThreshold = percentile(gray, 10);

  const mask = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    mask[i] = gray[i] < lowThreshold ? 1 : 0;
  }

  const components = connectedComponents(mask, gray, width, height, 40, 7500);
  const findings = [];

  for (const comp of components) {
    const yRatio = comp.cy / height;
    const xRatio = comp.cx / width;
    const aspect = comp.w / Math.max(1, comp.h);
    const density = comp.area / Math.max(1, comp.w * comp.h);
    const darkness = (lowThreshold - comp.meanGray) / Math.max(1, lowThreshold);
    const severeDark = comp.meanGray < strongThreshold;

    const confidenceBase =
      0.42 +
      Math.min(0.22, Math.max(0, darkness) * 0.35) +
      Math.min(0.2, comp.area / 3200) +
      Math.min(0.12, density * 0.2);

    if (yRatio < 0.45 && (xRatio < 0.28 || xRatio > 0.72) && comp.area > 70 && comp.area < 3500) {
      findings.push(
        buildFinding({
          type: "class_ii_caries",
          title: "اشتباه نخر صنف ثاني",
          severity: severeDark ? "alert" : "warn",
          confidence: confidenceBase + (severeDark ? 0.08 : 0),
          box: comp,
          rationale: "منطقة شعاعية منخفضة الكثافة في المنطقة القريبة التاجية مع حدود موضعية.",
          recommendation: "فحص سريري للسطح القريب + اختبار حيوية السن + مقارنة بصورة إضافية bitewing إن أمكن."
        })
      );
      continue;
    }

    if (yRatio > 0.57 && aspect > 0.45 && aspect < 2.9 && comp.area > 80) {
      findings.push(
        buildFinding({
          type: "apical_lesion",
          title: "اشتباه آفة ذروية",
          severity: "alert",
          confidence: confidenceBase + 0.06,
          box: comp,
          rationale: "وجود منطقة منخفضة الكثافة حول الذروة مع شكل بؤري وحدود يمكن تمييزها.",
          recommendation: "ربطها بالأعراض السريرية واختبارات القرع/الجس وحيوية اللب وتقييم متابعة شعاعية."
        })
      );
      continue;
    }

    if (yRatio >= 0.4 && yRatio <= 0.72 && aspect < 0.65 && comp.h > comp.w * 1.35 && comp.area > 55) {
      findings.push(
        buildFinding({
          type: "root_radiolucency",
          title: "اشتباه تبدل شعاعي على مسار الجذر",
          severity: "warn",
          confidence: confidenceBase,
          box: comp,
          rationale: "منطقة طولية منخفضة الكثافة على امتداد الجذر وقد ترتبط بتبدل نسجي موضعي.",
          recommendation: "مقارنة الإسقاطات الشعاعية وتقييم سلامة الجذر والنسج الداعمة سريريًا."
        })
      );
    }
  }

  findings.sort((a, b) => b.confidence - a.confidence);
  return findings.slice(0, 8).map((f, index) => ({ ...f, idx: index + 1 }));
}

function buildFinding(item) {
  const confidence = Math.max(0.3, Math.min(0.96, item.confidence));
  return {
    ...item,
    confidence,
    metrics: {
      areaPx: item.box.area,
      widthPx: item.box.w,
      heightPx: item.box.h,
      centroid: { x: item.box.cx, y: item.box.cy }
    }
  };
}

function connectedComponents(mask, gray, width, height, minArea, maxArea) {
  const visited = new Uint8Array(mask.length);
  const components = [];
  const dirs = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
  ];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const start = y * width + x;
      if (visited[start] || !mask[start]) continue;

      const queue = [start];
      visited[start] = 1;
      let qi = 0;
      let area = 0;
      let sumX = 0;
      let sumY = 0;
      let sumGray = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;

      while (qi < queue.length) {
        const idx = queue[qi++];
        const cx = idx % width;
        const cy = (idx - cx) / width;
        area++;
        sumX += cx;
        sumY += cy;
        sumGray += gray[idx];
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        for (const [dx, dy] of dirs) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const nIdx = ny * width + nx;
          if (visited[nIdx] || !mask[nIdx]) continue;
          visited[nIdx] = 1;
          queue.push(nIdx);
        }
      }

      if (area >= minArea && area <= maxArea) {
        components.push({
          area,
          cx: sumX / area,
          cy: sumY / area,
          meanGray: sumGray / area,
          x: minX,
          y: minY,
          w: maxX - minX + 1,
          h: maxY - minY + 1
        });
      }
    }
  }

  return components;
}

function drawEnhancedWithOverlays(gray, width, height, findings) {
  const out = resultCtx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    const p = i * 4;
    out.data[p] = v;
    out.data[p + 1] = v;
    out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  resultCtx.putImageData(out, 0, 0);

  for (const item of findings) {
    const color = item.severity === "alert" ? "#d53232" : "#d98007";
    resultCtx.strokeStyle = color;
    resultCtx.fillStyle = item.severity === "alert" ? "rgba(213,50,50,0.13)" : "rgba(217,128,7,0.12)";
    resultCtx.lineWidth = 2;
    resultCtx.setLineDash([6, 4]);
    roundedRect(resultCtx, item.box.x, item.box.y, item.box.w, item.box.h, 8);
    resultCtx.fill();
    resultCtx.stroke();
    resultCtx.setLineDash([]);

    const tagY = Math.max(16, item.box.y - 6);
    resultCtx.fillStyle = color;
    resultCtx.font = "bold 12px Arial";
    resultCtx.fillText(`#${item.idx} ${item.title}`, item.box.x + 2, tagY);
  }
}

function renderFindings(findings) {
  diagnosisList.innerHTML = "";
  if (findings.length === 0) {
    const li = document.createElement("li");
    li.className = "ok";
    li.textContent = "لم يتم رصد مؤشرات مرضية واضحة عبر هذه القراءة الآلية. يوصى بالمطابقة مع الفحص السريري.";
    diagnosisList.appendChild(li);
    return;
  }

  for (const item of findings) {
    const li = document.createElement("li");
    li.className = item.severity;
    li.textContent = `[#${item.idx}] ${item.title} — ثقة تقديرية ${Math.round(item.confidence * 100)}%`;
    diagnosisList.appendChild(li);
  }
}

function renderDetailedReport(findings, quality) {
  const severityCount = findings.reduce(
    (acc, f) => {
      acc[f.severity] = (acc[f.severity] || 0) + 1;
      return acc;
    },
    { alert: 0, warn: 0 }
  );

  if (findings.length === 0) {
    reportBox.innerHTML = `
      <p class="report-summary">ملخص القراءة: لا توجد آفات شعاعية مؤكدة آلياً.</p>
      <p>جودة الصورة: ${quality.qualityLabel} (مجال ديناميكي: ${Math.round(quality.dynamicRange)}، تباين: ${quality.contrast.toFixed(1)}).</p>
      <p>التوصية: المتابعة السريرية واختبارات الحيوية عند وجود أعراض.</p>
    `;
    return;
  }

  const summary = `تم رصد ${findings.length} منطقة مشتبهة (شدة عالية: ${severityCount.alert || 0}، متوسطة: ${severityCount.warn || 0}).`;

  const findingsHtml = findings
    .map((item) => {
      const loc = describeLocation(item.box.cx / sourceCanvas.width, item.box.cy / sourceCanvas.height);
      return `
        <li>
          <span class="report-tag ${item.severity}">#${item.idx}</span>
          <strong>${item.title}</strong><br/>
          <span>الموقع التقريبي: ${loc}</span><br/>
          <span>حدود المنطقة: x=${Math.round(item.box.x)}..${Math.round(item.box.x + item.box.w)} ، y=${Math.round(item.box.y)}..${Math.round(item.box.y + item.box.h)}</span><br/>
          <span>المساحة: ${item.metrics.areaPx} بكسل | الأبعاد: ${item.metrics.widthPx}×${item.metrics.heightPx}</span><br/>
          <span>سبب الاشتباه: ${item.rationale}</span><br/>
          <span>الثقة التقديرية: ${Math.round(item.confidence * 100)}%</span><br/>
          <span>مقترح المتابعة: ${item.recommendation}</span>
        </li>
      `;
    })
    .join("");

  reportBox.innerHTML = `
    <p class="report-summary">ملخص القراءة: ${summary}</p>
    <p>جودة الصورة: ${quality.qualityLabel} (مجال ديناميكي: ${Math.round(quality.dynamicRange)}، تباين: ${quality.contrast.toFixed(1)}).</p>
    <ol class="report-findings">${findingsHtml}</ol>
    <p>تنبيه: هذا التقرير آلي مبدئي ويحتاج تأكيدًا سريريًا وشعاعيًا قبل اتخاذ قرار علاجي.</p>
  `;
}

function describeLocation(xRatio, yRatio) {
  const horizontal = xRatio < 0.33 ? "يسار" : xRatio > 0.66 ? "يمين" : "منتصف";
  const vertical = yRatio < 0.33 ? "تاجي علوي" : yRatio > 0.66 ? "ذروي سفلي" : "منطقة جذرية وسطى";
  return `${horizontal} / ${vertical}`;
}

function setDiagnosis(messages, className) {
  diagnosisList.innerHTML = "";
  messages.forEach((message) => {
    const li = document.createElement("li");
    li.className = className;
    li.textContent = message;
    diagnosisList.appendChild(li);
  });
}

function setReportPlaceholder() {
  reportBox.textContent =
    "لا يوجد تقرير بعد. بعد التحليل ستظهر لك قراءة مفصلة تتضمن ما تم رصده وموقعه وحدوده التقريبية ومبررات الاشتباه.";
}

function captureMeasurePoint(event) {
  if (!imageReady || !latestGray) return;
  const rect = resultCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * resultCanvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * resultCanvas.height;

  if (measurePoints.length === 2) measurePoints = [];
  measurePoints.push({ x, y });
  redrawCanvasWithMeasurements();
  updateWorkLengthText();
}

function redrawCanvasWithMeasurements() {
  if (!latestGray) return;
  drawEnhancedWithOverlays(latestGray, resultCanvas.width, resultCanvas.height, latestFindings);
  if (measurePoints.length === 0) return;

  resultCtx.strokeStyle = "#0a6db5";
  resultCtx.fillStyle = "#0a6db5";
  resultCtx.lineWidth = 2;
  measurePoints.forEach((p) => {
    resultCtx.beginPath();
    resultCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    resultCtx.fill();
  });

  if (measurePoints.length === 2) {
    const [a, b] = measurePoints;
    resultCtx.beginPath();
    resultCtx.moveTo(a.x, a.y);
    resultCtx.lineTo(b.x, b.y);
    resultCtx.stroke();
  }
}

function updateWorkLengthText() {
  if (measurePoints.length < 2) {
    workLengthOutput.textContent = "حدّد نقطتين لإظهار الطول العامل.";
    return;
  }

  const [a, b] = measurePoints;
  const pxDistance = Math.hypot(a.x - b.x, a.y - b.y);
  const mmPerPixel = Number(mmPerPixelInput.value);
  if (!Number.isFinite(mmPerPixel) || mmPerPixel <= 0) {
    workLengthOutput.textContent = "قيمة المعايرة غير صحيحة.";
    return;
  }

  const anatomical = pxDistance * mmPerPixel;
  const workingLow = Math.max(0, anatomical - 1);
  const workingHigh = Math.max(0, anatomical - 0.5);
  workLengthOutput.textContent =
    `الطول التشريحي التقريبي: ${anatomical.toFixed(2)} مم | ` +
    `الطول العامل المقترح: بين ${workingLow.toFixed(2)} و ${workingHigh.toFixed(2)} مم`;
}

function clearMeasurePoints() {
  measurePoints = [];
  workLengthOutput.textContent = "لم يتم القياس بعد.";
  if (imageReady && latestGray) {
    drawEnhancedWithOverlays(latestGray, resultCanvas.width, resultCanvas.height, latestFindings);
  }
}

function resetAll() {
  imageInput.value = "";
  imageReady = false;
  latestGray = null;
  latestFindings = [];
  measurePoints = [];
  sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  exportBtn.disabled = true;
  setDiagnosis(["تمت إعادة الضبط. ارفع صورة شعاعية ذروية للبدء مجدداً."], "ok");
  setReportPlaceholder();
  workLengthOutput.textContent = "لم يتم القياس بعد.";
}

function exportResults() {
  const link = document.createElement("a");
  link.download = "endodiag-result.png";
  link.href = resultCanvas.toDataURL("image/png");
  link.click();
}

function drawImageContain(ctx, img, w, h) {
  ctx.clearRect(0, 0, w, h);
  const ratio = Math.min(w / img.width, h / img.height);
  const drawW = img.width * ratio;
  const drawH = img.height * ratio;
  const offsetX = (w - drawW) / 2;
  const offsetY = (h - drawH) / 2;
  ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
}

function standardDeviation(data) {
  const mean = data.reduce((acc, v) => acc + v, 0) / data.length;
  const variance = data.reduce((acc, v) => acc + (v - mean) ** 2, 0) / data.length;
  return Math.sqrt(variance);
}

function percentile(data, p) {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx];
}

function roundedRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
