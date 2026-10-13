const startBtn = document.getElementById("startBtn");
const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const exportBtn = document.getElementById("exportBtn");
const clearMeasureBtn = document.getElementById("clearMeasureBtn");
const mmPerPixelInput = document.getElementById("mmPerPixel");
const diagnosisList = document.getElementById("diagnosisList");
const workLengthOutput = document.getElementById("workLengthOutput");

const sourceCanvas = document.getElementById("sourceCanvas");
const resultCanvas = document.getElementById("resultCanvas");
const sourceCtx = sourceCanvas.getContext("2d");
const resultCtx = resultCanvas.getContext("2d");

let imageReady = false;
let measurePoints = [];
let latestFindings = [];

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
    clearMeasurePoints();
    setDiagnosis(["تم تحميل الصورة بنجاح. اضغط زر المعالجة والتشخيص للمتابعة."], "ok");
  };
  image.src = URL.createObjectURL(file);
}

function runAnalysis() {
  if (!imageReady) return;

  const imageData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const enhanced = enhanceRadiograph(imageData);
  const findings = findDiagnosticCues(enhanced.gray, enhanced.width, enhanced.height);
  latestFindings = findings;

  drawEnhancedWithOverlays(enhanced.gray, enhanced.width, enhanced.height, findings);
  renderFindings(findings);
  exportBtn.disabled = false;
  clearMeasurePoints();
}

function enhanceRadiograph(imageData) {
  const { data, width, height } = imageData;
  const gray = new Uint8ClampedArray(width * height);

  let min = 255;
  let max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const g = Math.round(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    const idx = i / 4;
    gray[idx] = g;
    min = Math.min(min, g);
    max = Math.max(max, g);
  }

  const span = Math.max(1, max - min);
  for (let i = 0; i < gray.length; i++) {
    const normalized = ((gray[i] - min) / span) * 255;
    gray[i] = Math.max(0, Math.min(255, Math.round((normalized - 128) * 1.35 + 128)));
  }

  return { gray, width, height };
}

function findDiagnosticCues(gray, width, height) {
  const threshold = percentile(gray, 18);
  const mask = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    mask[i] = gray[i] < threshold ? 1 : 0;
  }

  const components = connectedComponents(mask, gray, width, height, 45, 6000);
  const findings = [];

  for (const comp of components) {
    const yRatio = comp.cy / height;
    const aspect = comp.w / Math.max(1, comp.h);
    const density = comp.area / (comp.w * comp.h);
    const darknessScore = Math.max(0, (threshold - comp.meanGray) / 60);
    const confidence = Math.min(
      0.95,
      0.45 + darknessScore * 0.25 + Math.min(0.2, density * 0.25) + Math.min(0.15, comp.area / 3000)
    );

    if (yRatio < 0.52 && aspect > 0.4 && aspect < 2.4 && comp.area < width * height * 0.08) {
      findings.push({
        type: "caries",
        label: "اشتباه منطقة نخر/انخفاض كثافة في التاج أو منطقة قريبة",
        severity: "warn",
        confidence,
        box: comp
      });
    } else if (yRatio >= 0.57 && comp.area < width * height * 0.16) {
      findings.push({
        type: "apical",
        label: "اشتباه آفة ذروية أو تبدل شعاعي حول الذروة",
        severity: "alert",
        confidence,
        box: comp
      });
    }
  }

  findings.sort((a, b) => b.confidence - a.confidence);
  return findings.slice(0, 6);
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

let latestGray = null;

function drawEnhancedWithOverlays(gray, width, height, findings) {
  latestGray = gray;
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
    resultCtx.lineWidth = 2;
    resultCtx.strokeStyle = item.severity === "alert" ? "#d53232" : "#d98007";
    resultCtx.fillStyle = "rgba(213,50,50,0.12)";
    resultCtx.strokeRect(item.box.x, item.box.y, item.box.w, item.box.h);
    resultCtx.font = "13px Arial";
    resultCtx.fillStyle = item.severity === "alert" ? "#8a1111" : "#8f5102";
    resultCtx.fillText(`${item.label} (${Math.round(item.confidence * 100)}%)`, item.box.x + 2, Math.max(14, item.box.y - 4));
  }
}

function renderFindings(findings) {
  diagnosisList.innerHTML = "";
  if (findings.length === 0) {
    const li = document.createElement("li");
    li.className = "ok";
    li.textContent = "لم يتم رصد مؤشرات شعاعية واضحة في هذه القراءة الآلية الأولية.";
    diagnosisList.appendChild(li);
    return;
  }

  for (const item of findings) {
    const li = document.createElement("li");
    li.className = item.severity;
    li.textContent = `${item.label} — الثقة التقريبية: ${Math.round(item.confidence * 100)}%`;
    diagnosisList.appendChild(li);
  }
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

function percentile(data, p) {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx];
}
