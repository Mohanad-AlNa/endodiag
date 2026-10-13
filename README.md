# EndoDiag

منصة أولية لتشخيص الصور الشعاعية الذروية في المداواة اللبية، تتضمن:
1) واجهة عربية للتجربة السريعة.
2) Backend للتنبؤ بنموذج ذكاء اصطناعي مدرّب.
3) سكربتات تدريب/تقييم/استدلال لبناء النموذج الطبي.

## الميزات
- صفحة رئيسية توضّح فكرة المشروع مع الإسناد إلى د. مهند النشواتي.
- رفع صورة شعاعية ذروية ومعالجتها داخل المتصفح.
- نمط **AI API** لقراءة الصورة عبر نموذج مدرّب مع إظهار الحدود الفعلية للآفات (Polygon/Mask عند توفرها).
- نمط **تجريبي محلي** (heuristic) كخيار مؤقت منخفض الدقة عند عدم توفر نموذج مدرّب.
- تحديد مناطق مشتبهة مثل: `class_ii_caries`، `apical_lesion`، `pdl_widening`، `resorption`.
- توليد تقرير شعاعي مفصل يتضمن الموقع، الأبعاد، المساحة، سبب الاشتباه، ومقترح المتابعة لكل منطقة.
- أداة تقدير الطول العامل التقريبي بالاعتماد على معايرة (مم/بكسل).
- تصدير صورة النتائج مع التحديدات.

## تشغيل الواجهة فقط
- افتح `index.html` مباشرة أو عبر خادم ملفات ثابتة.

### خيار ملف واحد (الأسهل)
إذا أردت التجربة بملف واحد فقط، افتح الملف:

`endodiag-standalone.html`

هذا الملف يحتوي HTML + CSS + JavaScript مضمّنة داخله، ويعمل مباشرة عند فتحه بالمتصفح.

## تشغيل Backend للذكاء الاصطناعي
> يتطلب Python 3.10+ وبيئة افتراضية.

```bash
cd /home/runner/work/endodiag/endodiag/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ثم شغّل الخادم:

```bash
cd /home/runner/work/endodiag/endodiag/backend
ENDODIAG_MODEL_PATH=/absolute/path/to/best.pt uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- فحص الحالة: `GET /health`
- التنبؤ: `POST /predict` (حقل `file` صورة)

## تدريب النموذج (YOLO Segmentation)
يوجد سكربتات تحت `/ml`:
- `train.py`: تدريب النموذج.
- `evaluate.py`: تقييم mAP للصناديق والـ masks.
- `infer.py`: استدلال على صورة واحدة وتصدير النتائج.

مثال إعداد البيانات:
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: class_ii_caries
  1: apical_lesion
  2: pdl_widening
  3: resorption
```
انظر: `ml/dataset.example.yaml`

### مثال تدريب
```bash
cd /home/runner/work/endodiag/endodiag
python ml/train.py --data /absolute/path/to/dataset.yaml --model yolov8n-seg.pt --epochs 120 --imgsz 1024
```

### مثال تقييم
```bash
python ml/evaluate.py --model /absolute/path/to/best.pt --data /absolute/path/to/dataset.yaml
```

### مثال استدلال منفرد
```bash
python ml/infer.py --model /absolute/path/to/best.pt --image /absolute/path/to/image.png
```

## توصيات رفع الدقة (مختصرة)
- توحيد بروتوكول الوسم بين أخصائيين قبل التدريب.
- تقسيم البيانات على مستوى **المريض** وليس الصورة.
- تتبع Sensitivity/Specificity/F1/mAP/Dice وليس الدقة العامة فقط.
- اعتماد حلقة تحسين مستمر للحالات الفاشلة (Active Learning).