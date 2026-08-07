import React, { useState, useEffect } from 'react';
import { createIncidentApi } from '../api/incidents';
import { IncidentDetail } from '../types';
import { ProbabilityBars } from '../components/ProbabilityBars';
import { RecommendationCard } from '../components/RecommendationCard';
import { LeafletMap } from '../components/LeafletMap';
import {
  UploadCloud,
  MapPin,
  Send,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Sparkles,
  RefreshCw,
  Globe2,
  Cpu,
  Layers,
  Check,
  Navigation,
} from 'lucide-react';

export const CitizenPortal: React.FC = () => {
  const [description, setDescription] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [latitude, setLatitude] = useState<number>(28.6139); // Default New Delhi
  const [longitude, setLongitude] = useState<number>(77.2090);
  const [addressText, setAddressText] = useState('');
  const [gpsStatus, setGpsStatus] = useState<string>('Detecting location...');
  const [showManualMap, setShowManualMap] = useState(false);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IncidentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Auto request browser GPS on mount
  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLatitude(pos.coords.latitude);
          setLongitude(pos.coords.longitude);
          setGpsStatus(`GPS Acquired: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`);
        },
        (err) => {
          setGpsStatus('GPS permission denied. Please select location manually on map.');
          setShowManualMap(true);
        },
        { timeout: 8000 }
      );
    } else {
      setGpsStatus('Geolocation not supported. Please select location manually.');
      setShowManualMap(true);
    }
  }, []);

  const handleImageChange = (file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageChange(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim() && !imageFile) {
      setError('Please provide at least a text description or upload a disaster photo.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    if (description) formData.append('description', description);
    if (imageFile) formData.append('image', imageFile);
    formData.append('latitude', latitude.toString());
    formData.append('longitude', longitude.toString());
    if (addressText) formData.append('address_text', addressText);

    try {
      const res = await createIncidentApi(formData);
      setResult(res);
      // Smooth scroll to results
      setTimeout(() => {
        document.getElementById('triage-results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Incident submission failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const setSampleText = (text: string) => {
    setDescription(text);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8 space-y-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-slate-900 border border-slate-800 rounded-3xl p-8 lg:p-12 shadow-2xl">
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-950/80 border border-blue-800 text-blue-400 text-xs font-mono rounded-full font-semibold">
            <Sparkles className="w-3.5 h-3.5" /> Multimodal Emergency AI Triage System
          </div>

          <h1 className="text-3xl lg:text-5xl font-black text-slate-100 tracking-tight leading-tight">
            Rapid Disaster Reporting & <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-amber-400 to-blue-500">
              Automated Emergency Prioritization
            </span>
          </h1>

          <p className="text-sm lg:text-base text-slate-300 leading-relaxed font-sans">
            Submit citizen report photos and multilingual text descriptions. Our deep learning ensemble evaluates disaster severity in real time and automatically dispatches emergency rescue protocols.
          </p>

          {/* Quick Metrics Bar */}
          <div className="pt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 block text-[10px]">VISION ENSEMBLE</span>
              <span className="text-amber-400 font-bold text-sm">6 Disaster Classes</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 block text-[10px]">TEXT REGRESSOR</span>
              <span className="text-emerald-400 font-bold text-sm">Priority 0–100</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 block text-[10px]">LANGUAGES</span>
              <span className="text-blue-400 font-bold text-sm">EN / HI / Hinglish</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 block text-[10px]">LATENCY</span>
              <span className="text-purple-400 font-bold text-sm">&lt; 800ms Pipeline</span>
            </div>
          </div>
        </div>
      </section>

      {/* Incident Report Form */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-3xl p-6 lg:p-8 space-y-6 shadow-xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-slate-100">Report Emergency Incident</h2>
              <p className="text-xs text-slate-400">Upload disaster photo and describe the situation</p>
            </div>
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-full border border-emerald-800">
              Live Triage Form
            </span>
          </div>

          {error && (
            <div className="p-4 bg-red-950/80 border border-red-800 rounded-xl text-red-300 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Drag and Drop Image Upload */}
            <div className="space-y-2">
              <label className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold">
                1. Upload Disaster Photo (Optional)
              </label>
              
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                className="border-2 border-dashed border-slate-700 hover:border-blue-500 bg-slate-950/60 rounded-2xl p-6 text-center transition-colors cursor-pointer relative"
              >
                {imagePreview ? (
                  <div className="relative group max-h-48 mx-auto w-fit">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="max-h-48 rounded-xl object-contain border border-slate-700"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setImageFile(null);
                        setImagePreview(null);
                      }}
                      className="absolute top-2 right-2 bg-red-600 text-white p-1 rounded-full text-xs shadow-lg hover:bg-red-500"
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <label className="cursor-pointer space-y-3 block">
                    <UploadCloud className="w-10 h-10 text-blue-500 mx-auto animate-bounce" />
                    <div>
                      <p className="text-xs font-semibold text-slate-200">
                        Click or Drag & Drop photo here
                      </p>
                      <p className="text-[10px] text-slate-500 mt-1 font-mono">
                        Supports JPG, PNG, WEBP (Max 10MB)
                      </p>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => e.target.files?.[0] && handleImageChange(e.target.files[0])}
                    />
                  </label>
                )}
              </div>
            </div>

            {/* Multilingual Text Box */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold">
                  2. Citizen Text Report
                </label>
                <span className="text-[10px] font-mono text-purple-400">Supports English / Hindi / Hinglish</span>
              </div>

              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={4}
                placeholder="Describe the emergency... e.g. 'Massive flood waters rising fast. 4 people trapped on roof requiring immediate boat rescue!'"
                className="w-full p-4 bg-slate-950 border border-slate-800 rounded-2xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 leading-relaxed resize-none"
              />

              {/* Sample Prompt Chips */}
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-slate-500">Quick Test Prompts:</span>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  <button
                    type="button"
                    onClick={() => setSampleText("Massive flood waters rising rapidly. Four people trapped on roof needing immediate boat rescue!")}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-blue-300 rounded-lg transition-colors border border-slate-700"
                  >
                    🇬🇧 English Flood
                  </button>
                  <button
                    type="button"
                    onClick={() => setSampleText("बाढ़ का पानी घर में घुस गया है, 3 लोग फंसे हैं, तुरंत नाव भेजो!")}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-amber-300 rounded-lg transition-colors border border-slate-700"
                  >
                    🇮🇳 Hindi Flood
                  </button>
                  <button
                    type="button"
                    onClick={() => setSampleText("Aag bohot fail gayi hai building me, rescue team and ambulance jaldi bhejo!")}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-purple-300 rounded-lg transition-colors border border-slate-700"
                  >
                    🗣️ Hinglish Fire
                  </button>
                </div>
              </div>
            </div>

            {/* GPS Location & Map Toggle */}
            <div className="space-y-3 p-4 bg-slate-950 border border-slate-800 rounded-2xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <Navigation className="w-4 h-4 text-blue-400" />
                  <span className="font-mono text-[11px]">{gpsStatus}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setShowManualMap(!showManualMap)}
                  className="text-xs text-blue-400 hover:underline font-mono"
                >
                  {showManualMap ? 'Hide Map' : 'Select on Map'}
                </button>
              </div>

              {showManualMap && (
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-400 font-mono">
                    Click anywhere on the map to set exact incident coordinates:
                  </span>
                  <div className="h-56">
                    <LeafletMap
                      center={[latitude, longitude]}
                      zoom={12}
                      selectedLocation={{ latitude, longitude }}
                      onSelectLocation={(lat, lng) => {
                        setLatitude(lat);
                        setLongitude(lng);
                        setGpsStatus(`Manual GPS Selected: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
                      }}
                      interactiveSelect={true}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 bg-gradient-to-r from-red-600 via-amber-600 to-blue-600 hover:opacity-95 text-white font-black text-sm uppercase tracking-wider rounded-2xl shadow-xl shadow-red-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Running AI Inference & Triage...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Submit Incident Report</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Informational Panel / Feature Cards */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
            <h3 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-400" /> AI System Architecture
            </h3>

            <div className="space-y-3 text-xs text-slate-300">
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-amber-400 block">Weighted Vision Ensemble (85.26% Accuracy)</span>
                <p className="text-slate-400 text-[11px]">
                  Fuses prediction probabilities from Swin Transformer Tiny, ConvNeXt Tiny, Vision Transformer (ViT-B/16), and EfficientNetV2-S.
                </p>
              </div>

              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-emerald-400 block">Multilingual Text Priority Regressor</span>
                <p className="text-slate-400 text-[11px]">
                  SentenceTransformer (all-MiniLM-L6-v2) embedding + CatBoost regressor delivering continuous 0–100 priority scores across English, Hindi, and Hinglish.
                </p>
              </div>

              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-purple-400 block">Automated Dispatch Protocol</span>
                <p className="text-slate-400 text-[11px]">
                  Maps disaster classifications directly to action directives (rescue boats, fire brigades, heavy excavators) for instantaneous emergency responder dispatch.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Post-Submission Triage Results Section */}
      {result && (
        <section id="triage-results" className="bg-slate-900 border-2 border-emerald-500/50 rounded-3xl p-6 lg:p-8 space-y-6 shadow-2xl shadow-emerald-500/10 animate-fade-in">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-slate-800 pb-4 gap-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-emerald-950 border border-emerald-700 flex items-center justify-center text-emerald-400">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-100">Incident Triage Complete</h3>
                <p className="text-xs font-mono text-slate-400">REQUEST ID #{result.id} • STATUS: {result.status}</p>
              </div>
            </div>
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-3 py-1 rounded-full border border-emerald-800">
              Estimated Processing Time: &lt; 0.8s
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 font-mono">
              <span className="text-xs text-slate-400 block">Disaster Type</span>
              <span className="text-xl font-bold text-amber-400">{result.disaster_type}</span>
            </div>

            <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 font-mono">
              <span className="text-xs text-slate-400 block">Priority Score</span>
              <span className="text-xl font-bold text-emerald-400">{result.priority_score.toFixed(1)} / 100</span>
            </div>

            <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 font-mono">
              <span className="text-xs text-slate-400 block">Priority Level</span>
              <span className="text-xl font-bold text-red-400 uppercase">{result.priority_level}</span>
            </div>

            <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 font-mono">
              <span className="text-xs text-slate-400 block">Detected Language</span>
              <span className="text-xl font-bold text-purple-400">{result.language_detected}</span>
            </div>
          </div>

          <RecommendationCard
            recommendations={result.recommendations}
            priorityLevel={result.priority_level}
            disasterType={result.disaster_type}
          />

          <div className="space-y-3">
            <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider">
              Disaster Probability Breakdown (All 6 Classes)
            </h4>
            <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800">
              <ProbabilityBars
                probabilities={result.class_probabilities}
                predictedClass={result.disaster_type}
              />
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
