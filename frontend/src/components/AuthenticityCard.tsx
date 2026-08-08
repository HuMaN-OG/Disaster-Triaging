import React, { useState } from 'react';
import { predictAuthenticityApi } from '../api/predict';
import { AuthenticityResult } from '../types';
import { ShieldCheck, ShieldAlert, Upload, Cpu, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

export const AuthenticityCard: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AuthenticityResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (selectedFile: File) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleVerify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image', file);
      const res = await predictAuthenticityApi(formData);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authenticity verification failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/80 rounded-2xl p-6 shadow-xl backdrop-blur-md">
      <div className="flex items-center space-x-3 mb-4">
        <div className="p-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl text-blue-400">
          <ShieldCheck className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">AI Image Authenticity Detector</h3>
          <p className="text-xs text-slate-400">Verify image authenticity using ConvNeXt-Tiny Deepfake Inspection</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        {/* Upload Box */}
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-blue-500/50 rounded-xl p-4 transition-all bg-slate-900/40 min-h-[160px]">
          {preview ? (
            <div className="relative w-full flex flex-col items-center">
              <img src={preview} alt="Upload preview" className="max-h-36 rounded-lg object-cover mb-2 border border-slate-700" />
              <button
                type="button"
                onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                className="text-xs text-slate-400 hover:text-red-400 underline"
              >
                Change Image
              </button>
            </div>
          ) : (
            <label className="cursor-pointer flex flex-col items-center space-y-2 text-center w-full">
              <Upload className="w-8 h-8 text-blue-400 mb-1" />
              <span className="text-sm font-medium text-slate-300">Drop disaster image or click to upload</span>
              <span className="text-xs text-slate-500">Supports JPG, PNG, WEBP</span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
              />
            </label>
          )}
        </div>

        {/* Action & Result Box */}
        <div className="flex flex-col justify-center space-y-4">
          {!result && (
            <button
              onClick={handleVerify}
              disabled={!file || loading}
              className={`w-full py-3 px-4 rounded-xl font-medium flex items-center justify-center space-x-2 transition-all ${
                !file || loading
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/25'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Analyzing Image Authenticity...</span>
                </>
              ) : (
                <>
                  <Cpu className="w-5 h-5" />
                  <span>Run Authenticity Check</span>
                </>
              )}
            </button>
          )}

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-xl flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="p-4 bg-slate-900/60 border border-slate-700/60 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 font-medium">Classification:</span>
                <span className={`px-3 py-1 text-xs font-semibold rounded-full flex items-center space-x-1.5 ${
                  result.is_ai_generated
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                }`}>
                  {result.is_ai_generated ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
                  <span>{result.classification}</span>
                </span>
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-300 mb-1">
                  <span>Confidence Score</span>
                  <span className="font-semibold">{Math.round(result.confidence * 100)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.is_ai_generated ? 'bg-gradient-to-r from-amber-500 to-rose-500' : 'bg-gradient-to-r from-teal-500 to-emerald-400'
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="text-[11px] text-slate-400 flex justify-between pt-1 border-t border-slate-800">
                <span>Assessment: <strong className="text-slate-200">{result.assessment}</strong></span>
                <span>Model: <strong className="text-slate-200">{result.model_info?.architecture || 'ConvNeXt-Tiny'}</strong></span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
