"""
FastAPI Server & Single-Page Demo Interface for Multimodal AI Disaster Triaging System.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from src.inference.predict import MultimodalPredictor

app = FastAPI(title="AI Disaster Triage System Demo")

# Warm up / lazy load MultimodalPredictor instance
predictor: MultimodalPredictor | None = None


def get_predictor() -> MultimodalPredictor:
    global predictor
    if predictor is None:
        predictor = MultimodalPredictor(image_model_type="ensemble")
    return predictor


def get_recommendation(disaster_type: str, priority_level: str, score: float) -> str:
    """Generate actionable dispatch recommendation for emergency responders."""
    if priority_level == "Critical" or score >= 80.0:
        return f"⚡ CRITICAL DISPATCH: Immediate deployment of rescue boats, heavy equipment, and trauma care team required for {disaster_type}."
    elif priority_level == "High" or score >= 60.0:
        return f"⚠️ HIGH PRIORITY: Dispatch emergency medical responders and field evacuation units for {disaster_type} within 30 minutes."
    elif priority_level == "Medium" or score >= 40.0:
        return f"📋 MEDIUM PRIORITY: Schedule field assessment team and distribute relief supplies (water/food) for {disaster_type} sector."
    else:
        return f"ℹ️ LOW PRIORITY: Log report for spatial awareness. No immediate dispatch needed unless status escalates."


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Disaster Triage AI Server Operational"}


@app.post("/predict")
async def predict_triage(
    image: UploadFile | None = File(None),
    text: str | None = Form(None),
) -> JSONResponse:
    """
    Unified Triage Prediction Endpoint.
    Accepts optional image file and/or optional citizen text report.
    """
    pred_engine = get_predictor()

    pil_img: Image.Image | None = None
    if image is not None and image.filename:
        content = await image.read()
        if content:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    text_clean: str | None = text.strip() if text and text.strip() else None

    if pil_img is None and text_clean is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Please provide an image photo, a text report, or both."},
        )

    result = pred_engine.predict(image_input=pil_img, text_report=text_clean)
    res_dict = result.to_dict()

    # Add response recommendation
    res_dict["recommendation"] = get_recommendation(
        result.disaster_type, result.priority_level, result.priority_score
    )

    return JSONResponse(content=res_dict)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Disaster Triage System | Live Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen pb-12">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-red-600 flex items-center justify-center font-black text-xl text-white shadow-lg shadow-red-900/40">🚨</div>
                <div>
                    <h1 class="text-xl font-extrabold tracking-tight text-white">AI Disaster Triage System</h1>
                    <p class="text-xs text-slate-400 font-medium">Multimodal Emergency Prioritization & Code-Mixed NLP</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span> Production Ready
                </span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-6xl mx-auto px-4 mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        <!-- Input Form Section (7 cols) -->
        <section class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 class="text-lg font-bold text-slate-100 mb-1 flex items-center">
                <span class="mr-2">📥</span> Submit Emergency Report
            </h2>
            <p class="text-sm text-slate-400 mb-6">Upload a photo, enter a citizen report (English, Hindi, or Hinglish), or submit both.</p>

            <form id="triageForm" class="space-y-6">
                <!-- Image Upload Box -->
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">1. Upload Photo (Optional)</label>
                    <div class="relative border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-xl p-4 text-center cursor-pointer transition bg-slate-950/50 group" id="dropZone">
                        <input type="file" id="imageInput" accept="image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" onchange="previewImage(this)">
                        <div id="uploadPrompt" class="space-y-2 py-4">
                            <div class="w-12 h-12 rounded-full bg-slate-800 group-hover:bg-slate-700 text-slate-300 flex items-center justify-center mx-auto text-2xl transition">📷</div>
                            <p class="text-sm font-medium text-slate-300">Click or drag disaster image here</p>
                            <p class="text-xs text-slate-500">Supports JPG, PNG, WEBP</p>
                        </div>
                        <div id="imagePreviewContainer" class="hidden relative">
                            <img id="imagePreview" class="max-h-48 rounded-lg mx-auto object-cover border border-slate-700 shadow-md">
                            <button type="button" onclick="removeImage(event)" class="absolute top-2 right-2 bg-slate-900/90 hover:bg-red-600 text-white rounded-full p-1.5 transition">✕</button>
                        </div>
                    </div>
                </div>

                <!-- Text Input & Sample Presets -->
                <div>
                    <div class="flex items-center justify-between mb-2">
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-300">2. Citizen Report Text (Optional)</label>
                        <span class="text-xs text-slate-500">Supports Code-Mixed Hinglish</span>
                    </div>
                    <textarea id="textInput" rows="3" class="w-full bg-slate-950 border border-slate-700 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-red-500 transition" placeholder="Describe emergency situation... e.g. Paani ghar ke andar aa gaya hai, four people trapped on roof"></textarea>

                    <!-- Quick Preset Buttons for Judges -->
                    <div class="mt-3 flex flex-wrap gap-2">
                        <span class="text-xs text-slate-500 self-center font-medium mr-1">Quick Presets:</span>
                        <button type="button" onclick="setPreset('hinglish')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 transition">💬 Hinglish Flood</button>
                        <button type="button" onclick="setPreset('hindi')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 transition">🇮🇳 Hindi Wildfire</button>
                        <button type="button" onclick="setPreset('english')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 transition">🇬🇧 English Rescue</button>
                    </div>
                </div>

                <!-- Submit Button -->
                <button type="submit" id="submitBtn" class="w-full bg-red-600 hover:bg-red-500 text-white font-bold py-3.5 px-6 rounded-xl shadow-lg shadow-red-900/30 flex items-center justify-center space-x-2 transition text-sm">
                    <span>Analyze Rescue Priority</span>
                    <span id="btnSpinner" class="hidden">
                        <svg class="animate-spin h-5 w-5 text-white ml-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </span>
                </button>
            </form>
        </section>

        <!-- Prediction Result Output (5 cols) -->
        <section class="lg:col-span-5">
            <!-- Empty Placeholder State -->
            <div id="emptyState" class="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center h-full flex flex-col items-center justify-center">
                <div class="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center text-3xl mb-4 text-slate-500">📊</div>
                <h3 class="text-base font-bold text-slate-300 mb-1">Awaiting Prediction</h3>
                <p class="text-xs text-slate-500 max-w-xs">Submit an emergency report or upload a disaster photo to calculate priority score.</p>
            </div>

            <!-- Prediction Result Card -->
            <div id="resultCard" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
                <!-- Mode & Language Badges -->
                <div class="flex items-center justify-between border-b border-slate-800 pb-4">
                    <span id="modeBadge" class="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">Multimodal</span>
                    <span id="langBadge" class="px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Hinglish</span>
                </div>

                <!-- Main Priority Score Display -->
                <div>
                    <div class="flex items-baseline justify-between mb-2">
                        <span class="text-xs font-semibold uppercase text-slate-400">Priority Severity Score</span>
                        <span id="scoreText" class="text-3xl font-extrabold text-white">92.4 <span class="text-sm font-normal text-slate-400">/ 100</span></span>
                    </div>
                    <!-- Progress Bar -->
                    <div class="w-full bg-slate-950 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
                        <div id="progressBar" class="h-full rounded-full transition-all duration-500 bg-red-500" style="width: 92.4%"></div>
                    </div>
                    <div class="flex justify-between items-center mt-2">
                        <span class="text-xs text-slate-500">Severity Band</span>
                        <span id="levelBadge" class="px-3 py-0.5 rounded-full text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30">CRITICAL</span>
                    </div>
                </div>

                <!-- Disaster Classification Details -->
                <div class="bg-slate-950/80 rounded-xl p-4 border border-slate-800 space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-slate-400 font-medium">Predicted Disaster Type</span>
                        <span id="disasterType" class="text-sm font-bold text-white">Flood</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-slate-400 font-medium">Vision Model Confidence</span>
                        <span id="confidenceText" class="text-xs font-semibold text-slate-300">94.12%</span>
                    </div>
                </div>

                <!-- Actionable Response Recommendation -->
                <div class="bg-red-950/30 border border-red-900/40 rounded-xl p-4 space-y-1">
                    <span class="text-xs font-bold uppercase tracking-wider text-red-400">Rescue Dispatch Recommendation</span>
                    <p id="recommendationText" class="text-xs text-slate-300 leading-relaxed">Immediate deployment of rescue boats, heavy equipment, and trauma care team required.</p>
                </div>
            </div>
        </section>
    </main>

    <script>
        function previewImage(input) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('uploadPrompt').classList.add('hidden');
                    document.getElementById('imagePreviewContainer').classList.remove('hidden');
                }
                reader.readAsDataURL(input.files[0]);
            }
        }

        function removeImage(e) {
            e.stopPropagation();
            document.getElementById('imageInput').value = '';
            document.getElementById('imagePreviewContainer').classList.add('hidden');
            document.getElementById('uploadPrompt').classList.remove('hidden');
        }

        function setPreset(type) {
            const txt = document.getElementById('textInput');
            if (type === 'hinglish') {
                txt.value = "Paani ghar ke andar aa gaya hai, four people trapped on roof rescue team bhejo";
            } else if (type === 'hindi') {
                txt.value = "जंगल की आग का धुआँ घना है और 4 लोग फँसे हुए हैं, तुरंत मदद भेजो";
            } else if (type === 'english') {
                txt.value = "Flash flood damaged access road, three residents trapped upstairs without food";
            }
        }

        document.getElementById('triageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            const spinner = document.getElementById('btnSpinner');
            const fileInput = document.getElementById('imageInput');
            const textInput = document.getElementById('textInput');

            if (!fileInput.files[0] && !textInput.value.trim()) {
                alert("Please upload a photo, enter report text, or both.");
                return;
            }

            submitBtn.disabled = true;
            spinner.classList.remove('hidden');

            const formData = new FormData();
            if (fileInput.files[0]) {
                formData.append('image', fileInput.files[0]);
            }
            if (textInput.value.trim()) {
                formData.append('text', textInput.value.trim());
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    renderResult(data);
                } else {
                    alert(data.error || "Prediction failed.");
                }
            } catch (err) {
                alert("Server error connecting to FastAPI demo backend.");
            } finally {
                submitBtn.disabled = false;
                spinner.classList.add('hidden');
            }
        });

        function renderResult(data) {
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('resultCard').classList.remove('hidden');

            document.getElementById('modeBadge').innerText = data.input_mode;
            document.getElementById('langBadge').innerText = data.language_detected;
            document.getElementById('scoreText').innerHTML = `${data.priority_score.toFixed(1)} <span class="text-sm font-normal text-slate-400">/ 100</span>`;
            
            const pBar = document.getElementById('progressBar');
            pBar.style.width = `${Math.min(data.priority_score, 100)}%`;

            const lBadge = document.getElementById('levelBadge');
            lBadge.innerText = data.priority_level.toUpperCase();

            if (data.priority_score >= 80) {
                pBar.className = 'h-full rounded-full transition-all duration-500 bg-red-500';
                lBadge.className = 'px-3 py-0.5 rounded-full text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30';
            } else if (data.priority_score >= 60) {
                pBar.className = 'h-full rounded-full transition-all duration-500 bg-orange-500';
                lBadge.className = 'px-3 py-0.5 rounded-full text-xs font-bold bg-orange-500/20 text-orange-400 border border-orange-500/30';
            } else if (data.priority_score >= 40) {
                pBar.className = 'h-full rounded-full transition-all duration-500 bg-amber-500';
                lBadge.className = 'px-3 py-0.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30';
            } else {
                pBar.className = 'h-full rounded-full transition-all duration-500 bg-emerald-500';
                lBadge.className = 'px-3 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
            }

            document.getElementById('disasterType').innerText = data.disaster_type;
            document.getElementById('confidenceText').innerText = `${(data.confidence * 100).toFixed(2)}%`;
            document.getElementById('recommendationText').innerText = data.recommendation;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """Serve single-page emergency triage demo interface."""
    return HTML_CONTENT
