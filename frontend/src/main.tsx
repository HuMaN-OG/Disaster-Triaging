import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import {
  Shield,
  FileText,
  Radio,
  BarChart2,
  LogIn,
  LogOut,
  Upload,
  Send,
  CheckCircle,
  AlertTriangle,
  Clock,
  RefreshCw,
  Search,
  Filter,
  X,
  ChevronRight,
  MapPin,
  Check,
  UserCheck,
  Building2,
  Flame,
  Globe,
  SlidersHorizontal,
  Paperclip,
  Clipboard,
} from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ResponseBriefModal } from './components/ResponseBriefModal';

// ── Leaflet Marker SVG Helper ──────────────────────────────────────────────
const createMarkerIcon = (color: string) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 32" width="20" height="28"><path fill="${color}" stroke="#111111" stroke-width="1.5" d="M12 0C5.37 0 0 5.37 0 12c0 9 12 20 12 20s12-11 12-20c0-6.63-5.37-12-12-12zm0 16c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z"/></svg>`;
  return L.divIcon({
    className: '',
    html: svg,
    iconSize: [20, 28],
    iconAnchor: [10, 28],
    popupAnchor: [0, -28],
  });
};

const MARKER_ICONS: Record<string, L.DivIcon> = {
  Critical: createMarkerIcon('#EF4444'),
  High: createMarkerIcon('#F59E0B'),
  Medium: createMarkerIcon('#EAB308'),
  Low: createMarkerIcon('#10B981'),
  VeryLow: createMarkerIcon('#10B981'),
};

const getMarkerIcon = (priorityLevel: string): L.DivIcon =>
  MARKER_ICONS[priorityLevel] || MARKER_ICONS.Low;

// ── Types ──────────────────────────────────────────────────────────────────
type Tab = 'citizen' | 'dashboard' | 'analytics';
type Role = 'Citizen' | 'Responder' | 'Admin';

interface AuthUser {
  email: string;
  role: Role;
  full_name: string;
}

interface Incident {
  id: number;
  title: string;
  description?: string;
  disaster_type: string;
  priority_score: number;
  priority_level: string;
  confidence: number;
  status: string;
  image_url?: string;
  language_detected: string;
  created_at: string;
  location?: { latitude: number; longitude: number; address_text?: string };
  class_probabilities?: Record<string, number>;
  recommendations?: string[];
  status_history?: Array<{ id: number; new_status: string; notes?: string; created_at: string }>;
}

interface DashStats {
  total_incidents: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  very_low: number;
  pending: number;
  assigned: number;
  en_route: number;
  in_progress: number;
  completed: number;
}

interface AnalyticsData {
  disaster_distribution: Record<string, number>;
  priority_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  language_distribution: Record<string, number>;
  average_confidence: number;
}

// ── API Helper ──────────────────────────────────────────────────────────────
const BASE = 'http://127.0.0.1:8000';

async function apiCall<T>(method: string, path: string, body?: any, isForm = false): Promise<T> {
  const token = localStorage.getItem('disaster_triage_token');
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!isForm && body) headers['Content-Type'] = 'application/json';

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    if (res.status === 401) {
      localStorage.removeItem('disaster_triage_token');
    }
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Status & Priority Badges ───────────────────────────────────────────────
const PriorityBadge = ({ level }: { level: string }) => {
  let style = 'bg-neutral-800 text-neutral-300 border-neutral-700';
  if (level === 'Critical') style = 'bg-red-950/60 text-red-400 border-red-900/60';
  else if (level === 'High') style = 'bg-amber-950/60 text-amber-400 border-amber-900/60';
  else if (level === 'Medium') style = 'bg-yellow-950/60 text-yellow-400 border-yellow-900/60';
  else if (level === 'Low' || level === 'Very Low') style = 'bg-emerald-950/60 text-emerald-400 border-emerald-900/60';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${style}`}>
      {level}
    </span>
  );
};

const StatusBadge = ({ status }: { status: string }) => {
  let style = 'bg-neutral-800 text-neutral-300 border-neutral-700';
  if (status === 'Completed') style = 'bg-emerald-950/50 text-emerald-400 border-emerald-900/50';
  else if (status === 'Pending') style = 'bg-neutral-800 text-neutral-400 border-neutral-700';
  else style = 'bg-amber-950/50 text-amber-400 border-amber-900/50';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${style}`}>
      {status}
    </span>
  );
};

// ── Map Sub-Components ─────────────────────────────────────────────────────
const MapUpdater: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

const LocationPicker: React.FC<{ onSelect: (lat: number, lng: number) => void }> = ({ onSelect }) => {
  useMapEvents({
    click(e) {
      onSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
};

// ── LOGIN MODAL ────────────────────────────────────────────────────────────
function LoginModal({
  onClose,
  onSuccess,
  inline = false,
}: {
  onClose?: () => void;
  onSuccess: (u: AuthUser) => void;
  inline?: boolean;
}) {
  const [email, setEmail] = useState('');
  const [pass, setPass] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (eEmail?: string, ePass?: string) => {
    const targetEmail = eEmail || email;
    const targetPass = ePass || pass;
    setLoading(true);
    setErr('');
    try {
      const tok = await apiCall<any>('POST', '/auth/login', { username: targetEmail, password: targetPass });
      localStorage.setItem('disaster_triage_token', tok.access_token);
      onSuccess({ email: targetEmail, role: tok.role, full_name: tok.full_name });
      if (onClose) onClose();
    } catch (ex: any) {
      setErr(ex.message);
    } finally {
      setLoading(false);
    }
  };

  const card = (
    <div className={`w-full ${inline ? 'h-full flex flex-col justify-center' : 'max-w-sm'} bg-[#181818] border border-[#2D2D2D] rounded-lg p-5 shadow-2xl relative`}>
      {!inline && onClose && (
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-neutral-400 hover:text-neutral-200 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}

      <div className="flex items-center gap-2 mb-1">
        <Shield className="w-4 h-4 text-[#D97706]" />
        <h2 className="text-sm font-semibold text-[#FAFAFA]">Responder Authentication</h2>
      </div>
      <p className="text-xs text-[#A3A3A3] mb-4">Sign in to manage incident status and response units.</p>

      {err && (
        <div className="mb-4 p-2.5 bg-red-950/50 border border-red-900/60 rounded text-xs text-red-300">
          {err}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleLogin();
        }}
        className="space-y-3"
      >
        <div>
          <label className="block text-[11px] font-medium text-[#A3A3A3] mb-1">Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="responder@disaster.org"
            className="w-full px-3 py-2 bg-[#111111] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
          />
        </div>

        <div>
          <label className="block text-[11px] font-medium text-[#A3A3A3] mb-1">Password</label>
          <input
            type="password"
            required
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            placeholder="••••••••"
            className="w-full px-3 py-2 bg-[#111111] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2 bg-[#D97706] hover:bg-[#b45309] text-white font-medium rounded text-xs transition-colors disabled:opacity-50"
        >
          {loading ? 'Authenticating...' : 'Sign In'}
        </button>
      </form>

      <div className="mt-4 pt-3 border-t border-[#2D2D2D]">
        <p className="text-[10px] text-[#A3A3A3] mb-2 font-mono uppercase">Quick Demo Access</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => handleLogin('responder@disaster.org', 'Responder@123')}
            className="py-1.5 px-2 bg-[#202020] hover:bg-[#2A2A2A] text-xs text-[#FAFAFA] border border-[#2D2D2D] rounded transition-colors"
          >
            Responder
          </button>
          <button
            onClick={() => handleLogin('admin@disaster.org', 'Admin@123')}
            className="py-1.5 px-2 bg-[#202020] hover:bg-[#2A2A2A] text-xs text-[#FAFAFA] border border-[#2D2D2D] rounded transition-colors"
          >
            Admin
          </button>
        </div>
      </div>
    </div>
  );

  if (inline) return card;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
      {card}
    </div>
  );
}

// ── CITIZEN PORTAL ─────────────────────────────────────────────────────────
function CitizenPortal() {
  const [desc, setDesc] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [lat, setLat] = useState(28.6139);
  const [lng, setLng] = useState(77.2090);
  const [gpsStatus, setGpsStatus] = useState('Detecting location...');
  const [showMap, setShowMap] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Incident | null>(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (!navigator.geolocation) {
      setGpsStatus('GPS unavailable. Select location on map.');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLat(pos.coords.latitude);
        setLng(pos.coords.longitude);
        setGpsStatus(`${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`);
      },
      () => setGpsStatus('GPS permission denied. Select on map.')
    );
  }, []);

  const handleFileChange = (f: File) => {
    setFile(f);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(f);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!desc.trim() && !file) {
      setErr('Please provide a description or photo.');
      return;
    }
    setLoading(true);
    setErr('');
    const fd = new FormData();
    if (desc) fd.append('description', desc);
    if (file) fd.append('image', file);
    fd.append('latitude', String(lat));
    fd.append('longitude', String(lng));

    try {
      const res = await apiCall<Incident>('POST', '/incidents', fd, true);
      setResult(res);
    } catch (ex: any) {
      setErr(ex.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="border-b border-[#2D2D2D] pb-4">
        <h1 className="text-lg font-semibold text-[#FAFAFA]">Report Emergency Incident</h1>
        <p className="text-xs text-[#A3A3A3]">
          Submit details for automated AI classification and priority routing.
        </p>
      </div>

      {err && (
        <div className="p-3 bg-red-950/50 border border-red-900/60 rounded text-xs text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
          <span>{err}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="bg-[#181818] border border-[#2D2D2D] rounded-lg p-6 space-y-5">
        {/* Photo Upload */}
        <div className="space-y-1.5">
          <label className="block text-xs font-medium text-[#FAFAFA]">Incident Image (Optional)</label>
          <div className="border border-dashed border-[#2D2D2D] hover:border-neutral-500 rounded bg-[#111111] p-4 text-center transition-colors">
            {preview ? (
              <div className="relative inline-block">
                <img src={preview} alt="Upload preview" className="max-h-40 rounded border border-[#2D2D2D]" />
                <button
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                  }}
                  className="absolute -top-2 -right-2 bg-neutral-800 text-neutral-300 p-1 rounded-full text-xs hover:bg-neutral-700"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <label className="cursor-pointer space-y-1 block">
                <Upload className="w-6 h-6 text-[#A3A3A3] mx-auto" />
                <p className="text-xs text-[#FAFAFA]">Click or drop photo here</p>
                <p className="text-[10px] text-[#A3A3A3]">JPG, PNG, WEBP</p>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                />
              </label>
            )}
          </div>
        </div>

        {/* Text Description */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <label className="block text-xs font-medium text-[#FAFAFA]">Description</label>
            <span className="text-[10px] text-[#A3A3A3] font-mono">English / Hindi</span>
          </div>
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            rows={3}
            placeholder="Describe the emergency, casualties, or required aid..."
            className="w-full p-2.5 bg-[#111111] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] placeholder-neutral-600 focus:outline-none focus:border-[#D97706] resize-none"
          />
          <div className="flex flex-wrap gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setDesc('Flood waters rising fast. Trapped on roof needing boat rescue!')}
              className="px-2 py-1 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] rounded text-[11px] text-[#A3A3A3]"
            >
              English Flood
            </button>
            <button
              type="button"
              onClick={() => setDesc('बाढ़ का पानी घर में घुस गया है, तुरंत नाव भेजो!')}
              className="px-2 py-1 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] rounded text-[11px] text-[#A3A3A3]"
            >
              Hindi Flood
            </button>
            <button
              type="button"
              onClick={() => setDesc('Aag bohot fail gayi hai building me, rescue team jaldi bhejo!')}
              className="px-2 py-1 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] rounded text-[11px] text-[#A3A3A3]"
            >
              Hinglish Fire
            </button>
          </div>
        </div>

        {/* Location Picker */}
        <div className="space-y-2 pt-2 border-t border-[#2D2D2D]">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5 text-[#A3A3A3]">
              <MapPin className="w-3.5 h-3.5 text-[#D97706]" />
              <span className="font-mono text-[11px]">{gpsStatus}</span>
            </div>
            <button
              type="button"
              onClick={() => setShowMap(!showMap)}
              className="text-[#D97706] hover:underline text-[11px]"
            >
              {showMap ? 'Hide Map' : 'Set Location on Map'}
            </button>
          </div>

          {showMap && (
            <div className="h-48 rounded overflow-hidden border border-[#2D2D2D]">
              <MapContainer center={[lat, lng]} zoom={12} style={{ width: '100%', height: '100%' }}>
                <MapUpdater center={[lat, lng]} zoom={12} />
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                />
                <LocationPicker
                  onSelect={(nLat, nLng) => {
                    setLat(nLat);
                    setLng(nLng);
                    setGpsStatus(`${nLat.toFixed(4)}, ${nLng.toFixed(4)}`);
                  }}
                />
                <Marker position={[lat, lng]} icon={getMarkerIcon('Critical')} />
              </MapContainer>
            </div>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-[#D97706] hover:bg-[#b45309] text-white font-medium rounded text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Analyzing Multimodal Input...</span>
            </>
          ) : (
            <>
              <Send className="w-3.5 h-3.5" />
              <span>Submit Report</span>
            </>
          )}
        </button>
      </form>

      {/* Result Card */}
      {result && (
        <div className="bg-[#181818] border border-[#2D2D2D] rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#2D2D2D] pb-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-500" />
              <h3 className="text-sm font-semibold text-[#FAFAFA]">Incident Report Submitted</h3>
            </div>
            <span className="text-[11px] font-mono text-[#A3A3A3]">ID #{result.id}</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-[#111111] p-2.5 rounded border border-[#2D2D2D]">
              <span className="text-[10px] text-[#A3A3A3] block font-mono">Disaster Type</span>
              <span className="font-medium text-[#FAFAFA]">{result.disaster_type}</span>
            </div>
            <div className="bg-[#111111] p-2.5 rounded border border-[#2D2D2D]">
              <span className="text-[10px] text-[#A3A3A3] block font-mono">Priority Score</span>
              <span className="font-mono font-semibold text-emerald-400">{result.priority_score.toFixed(1)} / 100</span>
            </div>
            <div className="bg-[#111111] p-2.5 rounded border border-[#2D2D2D]">
              <span className="text-[10px] text-[#A3A3A3] block font-mono">Priority Level</span>
              <PriorityBadge level={result.priority_level} />
            </div>
            <div className="bg-[#111111] p-2.5 rounded border border-[#2D2D2D]">
              <span className="text-[10px] text-[#A3A3A3] block font-mono">Confidence</span>
              <span className="font-mono text-[#FAFAFA]">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Class Probabilities */}
          {result.class_probabilities && (
            <div className="space-y-2 pt-2 border-t border-[#2D2D2D]">
              <span className="text-[11px] font-mono text-[#A3A3A3]">Classification Probabilities</span>
              <div className="space-y-1.5">
                {Object.entries(result.class_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div key={cls} className="space-y-0.5">
                      <div className="flex justify-between text-[11px]">
                        <span className={cls === result.disaster_type ? 'text-[#FAFAFA] font-medium' : 'text-[#A3A3A3]'}>
                          {cls}
                        </span>
                        <span className="font-mono text-[10px] text-[#A3A3A3]">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="bg-[#111111] h-1.5 rounded overflow-hidden">
                        <div
                          className={`h-full ${cls === result.disaster_type ? 'bg-[#D97706]' : 'bg-neutral-700'}`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Directives */}
          {result.recommendations && result.recommendations.length > 0 && (
            <div className="space-y-1.5 pt-2 border-t border-[#2D2D2D]">
              <span className="text-[11px] font-mono text-[#A3A3A3]">Recommended Action Protocol</span>
              <ul className="space-y-1 text-xs text-[#FAFAFA]">
                {result.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-[#D97706]">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── RESCUE DASHBOARD ───────────────────────────────────────────────────────
function RescueDashboard({
  userRole,
  onLoginSuccess,
}: {
  userRole: Role;
  onLoginSuccess?: (u: AuthUser) => void;
}) {
  const [stats, setStats] = useState<DashStats | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [search, setSearch] = useState('');
  const [priorityFilter, setPriorityFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState('');
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState<any | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showBrief, setShowBrief] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [uploading, setUploading] = useState(false);

  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const [locationFilter, setLocationFilter] = useState('');
  const [contextFilter, setContextFilter] = useState('ALL');

  // True when a responder/admin is authenticated (role is not the guest 'Citizen' default)
  const isAuth = userRole !== 'Citizen';

  const resetFilters = () => {
    setSearch('');
    setPriorityFilter('ALL');
    setStatusFilter('ALL');
    setCategoryFilter('ALL');
    setLocationFilter('');
    setContextFilter('ALL');
  };

  useEffect(() => {
    if (!selectedIncident) {
      setSelectedIncidentDetail(null);
      return;
    }
    setDetailLoading(true);
    apiCall<any>('GET', `/incidents/${selectedIncident.id}`)
      .then((data) => setSelectedIncidentDetail(data))
      .catch((err) => console.error("Failed to load details", err))
      .finally(() => setDetailLoading(false));
  }, [selectedIncident]);

  const loadData = async () => {
    setLoading(true);
    try {
      let url = '/incidents?limit=200';
      if (categoryFilter !== 'ALL') url += `&category=${encodeURIComponent(categoryFilter)}`;
      if (locationFilter) url += `&location=${encodeURIComponent(locationFilter)}`;
      if (contextFilter !== 'ALL') url += `&context=${encodeURIComponent(contextFilter)}`;

      const [sData, iData] = await Promise.all([
        apiCall<DashStats>('GET', '/dashboard/stats'),
        apiCall<Incident[]>('GET', url),
      ]);
      setStats(sData);
      setIncidents(iData);
    } catch {
      // Backend offline fallback
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [categoryFilter, locationFilter, contextFilter]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const filtered = incidents.filter((i) => {
    const q = search.toLowerCase();
    const matchesSearch =
      i.title.toLowerCase().includes(q) ||
      i.disaster_type.toLowerCase().includes(q) ||
      (i.description || '').toLowerCase().includes(q);
    const matchesPriority = priorityFilter === 'ALL' || i.priority_level === priorityFilter;
    const matchesStatus = statusFilter === 'ALL' || i.status === statusFilter;
    return matchesSearch && matchesPriority && matchesStatus;
  });

  const handleStatusUpdate = async (incId: number, newStatus: string) => {
    setActionMsg('Updating status...');
    try {
      await apiCall('PATCH', `/responder/${incId}/status`, {
        status: newStatus,
        notes: `Status changed to ${newStatus}`,
      });
      setActionMsg(`Updated to ${newStatus}`);
      await loadData();
      if (selectedIncident?.id === incId) {
        const updated = await apiCall<Incident>('GET', `/incidents/${incId}`);
        setSelectedIncident(updated);
      }
    } catch (ex: any) {
      setActionMsg(ex.message || 'Update failed');
    }
    setTimeout(() => setActionMsg(''), 3000);
  };

  const handleUpload = async () => {
    if (!selectedFiles || !selectedIncident) return;
    setUploading(true);
    setActionMsg('Uploading attachments...');
    try {
      const formData = new FormData();
      Array.from(selectedFiles).forEach((file) => {
        formData.append('files', file);
      });
      await apiCall('POST', `/incidents/${selectedIncident.id}/attachments`, formData, true);
      setSelectedFiles(null);
      setActionMsg('Attachments uploaded successfully');
      const updated = await apiCall<any>('GET', `/incidents/${selectedIncident.id}`);
      setSelectedIncidentDetail(updated);
      await loadData();
    } catch (ex: any) {
      setActionMsg(ex.message || 'Upload failed');
    }
    setTimeout(() => setActionMsg(''), 3000);
    setUploading(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-5">
      {/* Top Bar */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-[#2D2D2D] pb-3">
        <div>
          <h1 className="text-base font-semibold text-[#FAFAFA]">Rescue Operations Command Center</h1>
          <p className="text-xs text-[#A3A3A3]">Real-time incident dispatch & GIS operational monitoring</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={resetFilters}
            className="px-3 py-1.5 bg-[#202020] hover:bg-red-950/40 hover:text-red-400 border border-[#2D2D2D] hover:border-red-900/40 text-xs text-[#FAFAFA] rounded transition-colors flex items-center gap-1.5"
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>Reset Filters</span>
          </button>
          <button
            onClick={loadData}
            disabled={loading}
            className="px-3 py-1.5 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] text-xs text-[#FAFAFA] rounded transition-colors flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 text-xs">
          <div className="bg-[#181818] border border-[#2D2D2D] p-2.5 rounded">
            <span className="text-[10px] text-[#A3A3A3] block font-mono">TOTAL</span>
            <span className="text-sm font-semibold text-[#FAFAFA] font-mono">{stats.total_incidents}</span>
          </div>
          <div className="bg-[#181818] border border-red-900/40 p-2.5 rounded">
            <span className="text-[10px] text-red-400 block font-mono">CRITICAL</span>
            <span className="text-sm font-semibold text-red-400 font-mono">{stats.critical}</span>
          </div>
          <div className="bg-[#181818] border border-amber-900/40 p-2.5 rounded">
            <span className="text-[10px] text-amber-400 block font-mono">HIGH</span>
            <span className="text-sm font-semibold text-amber-400 font-mono">{stats.high}</span>
          </div>
          <div className="bg-[#181818] border border-yellow-900/40 p-2.5 rounded">
            <span className="text-[10px] text-yellow-400 block font-mono">MEDIUM</span>
            <span className="text-sm font-semibold text-yellow-400 font-mono">{stats.medium}</span>
          </div>
          <div className="bg-[#181818] border border-emerald-900/40 p-2.5 rounded">
            <span className="text-[10px] text-emerald-400 block font-mono">LOW</span>
            <span className="text-sm font-semibold text-emerald-400 font-mono">{(stats.low || 0) + (stats.very_low || 0)}</span>
          </div>
          <div className="bg-[#181818] border border-[#2D2D2D] p-2.5 rounded">
            <span className="text-[10px] text-[#A3A3A3] block font-mono">PENDING</span>
            <span className="text-sm font-semibold text-[#FAFAFA] font-mono">{stats.pending}</span>
          </div>
          <div className="bg-[#181818] border border-[#2D2D2D] p-2.5 rounded">
            <span className="text-[10px] text-amber-400 block font-mono">DISPATCHED</span>
            <span className="text-sm font-semibold text-amber-400 font-mono">
              {(stats.assigned || 0) + (stats.en_route || 0) + (stats.in_progress || 0)}
            </span>
          </div>
          <div className="bg-[#181818] border border-emerald-900/40 p-2.5 rounded">
            <span className="text-[10px] text-emerald-400 block font-mono">RESOLVED</span>
            <span className="text-sm font-semibold text-emerald-400 font-mono">{stats.completed}</span>
          </div>
        </div>
      )}

      {/* Main Layout Section */}
      {!isAuth && (
        /* Unauthenticated Side-by-Side View: Operational GIS Map (Left) + Responder Authentication (Right) */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-6 h-[480px] bg-[#181818] border border-[#2D2D2D] rounded flex flex-col p-2 space-y-2">
            <div className="flex justify-between items-center px-1 text-xs text-[#A3A3A3]">
              <span className="font-medium text-[#FAFAFA] flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-[#D97706]" /> Operational GIS Map
              </span>
              <span className="font-mono text-[10px]">{incidents.length} Markers</span>
            </div>
            <div className="flex-1 rounded overflow-hidden border border-[#2D2D2D]">
              <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ width: '100%', height: '100%' }}>
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                />
                {incidents.map((inc) =>
                  inc.location &&
                  typeof inc.location.latitude === 'number' &&
                  typeof inc.location.longitude === 'number' &&
                  !isNaN(inc.location.latitude) &&
                  !isNaN(inc.location.longitude) ? (
                    <Marker
                      key={inc.id}
                      position={[inc.location.latitude, inc.location.longitude]}
                      icon={getMarkerIcon(inc.priority_level)}
                      eventHandlers={{ click: () => setSelectedIncident(inc) }}
                    >
                      <Popup>
                        <div className="p-1 text-xs space-y-1">
                          <div className="font-semibold text-[#FAFAFA]">{inc.disaster_type}</div>
                          <div>Priority: {inc.priority_level}</div>
                          <div>Status: {inc.status}</div>
                        </div>
                      </Popup>
                    </Marker>
                  ) : null
                )}
              </MapContainer>
            </div>
          </div>

          <div className="lg:col-span-6 min-h-[480px]">
            <LoginModal inline={true} onSuccess={onLoginSuccess || (() => {})} />
          </div>
        </div>
      )}

      {/* Main Split Screen: GIS Map + Incident Data Table (when logged in) OR Data Table (when unauthenticated) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: GIS Map Panel (only when logged in) */}
        {isAuth && (
          <div className="lg:col-span-5 h-[420px] bg-[#181818] border border-[#2D2D2D] rounded flex flex-col p-2 space-y-2">
            <div className="flex justify-between items-center px-1 text-xs text-[#A3A3A3]">
              <span className="font-medium text-[#FAFAFA] flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-[#D97706]" /> Operational GIS Map
              </span>
              <span className="font-mono text-[10px]">{incidents.length} Markers</span>
            </div>
            <div className="flex-1 rounded overflow-hidden border border-[#2D2D2D]">
              <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ width: '100%', height: '100%' }}>
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                />
                {incidents.map((inc) =>
                  inc.location &&
                  typeof inc.location.latitude === 'number' &&
                  typeof inc.location.longitude === 'number' &&
                  !isNaN(inc.location.latitude) &&
                  !isNaN(inc.location.longitude) ? (
                    <Marker
                      key={inc.id}
                      position={[inc.location.latitude, inc.location.longitude]}
                      icon={getMarkerIcon(inc.priority_level)}
                      eventHandlers={{ click: () => setSelectedIncident(inc) }}
                    >
                      <Popup>
                        <div className="p-1 text-xs space-y-1">
                          <div className="font-semibold text-[#FAFAFA]">{inc.disaster_type}</div>
                          <div>Priority: {inc.priority_level}</div>
                          <div>Status: {inc.status}</div>
                        </div>
                      </Popup>
                    </Marker>
                  ) : null
                )}
              </MapContainer>
            </div>
          </div>
        )}

        {/* Right: Data Table */}
        <div className={isAuth ? "lg:col-span-7 space-y-3" : "lg:col-span-12 space-y-3"}>
          {/* Controls */}
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <Search className="w-3.5 h-3.5 text-[#A3A3A3] absolute left-2.5 top-2.5" />
              <input
                type="text"
                placeholder="Search by disaster type, title, text..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              />
            </div>
            <div className="flex gap-2">
              <select
                value={priorityFilter}
                onChange={(e) => setPriorityFilter(e.target.value)}
                className="px-2.5 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              >
                <option value="ALL">All Priorities</option>
                <option value="Critical">Critical</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-2.5 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              >
                <option value="ALL">All Statuses</option>
                <option value="Pending">Pending</option>
                <option value="Assigned">Assigned</option>
                <option value="En Route">En Route</option>
                <option value="Rescue In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>
            </div>
          </div>

          {/* Bounty Phase 1 Filters Row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 bg-[#151515] p-3.5 rounded border border-[#2D2D2D] text-xs">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-[#A3A3A3] uppercase tracking-wider">Disaster Category</label>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="w-full px-2.5 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              >
                <option value="ALL">All Categories</option>
                <option value="Flood">Flood</option>
                <option value="Earthquake">Earthquake</option>
                <option value="Urban Fire">Urban Fire</option>
                <option value="Wild Fire">Wild Fire</option>
                <option value="Landslide">Landslide</option>
                <option value="Drought">Drought</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-[#A3A3A3] uppercase tracking-wider">Location / Address</label>
              <input
                type="text"
                placeholder="Filter by location/address..."
                value={locationFilter}
                onChange={(e) => setLocationFilter(e.target.value)}
                className="w-full px-2.5 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-[#A3A3A3] uppercase tracking-wider">Application / Context</label>
              <select
                value={contextFilter}
                onChange={(e) => setContextFilter(e.target.value)}
                className="w-full px-2.5 py-1.5 bg-[#181818] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] focus:outline-none focus:border-[#D97706]"
              >
                <option value="ALL">All Contexts</option>
                <option value="Citizen Portal">Citizen Portal</option>
                <option value="Web dispatch">Web dispatch</option>
              </select>
            </div>
          </div>

          {/* Table Container */}
          <div className="bg-[#181818] border border-[#2D2D2D] rounded overflow-hidden">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-[#111111] border-b border-[#2D2D2D] text-[10px] font-mono uppercase text-[#A3A3A3]">
                  <th className="py-2 px-3">ID</th>
                  <th className="py-2 px-3">Disaster</th>
                  <th className="py-2 px-3">Priority</th>
                  <th className="py-2 px-3">Score</th>
                  <th className="py-2 px-3">Status</th>
                  <th className="py-2 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2D2D2D] text-[#FAFAFA]">
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-xs text-[#A3A3A3]">
                      No incidents match criteria.
                    </td>
                  </tr>
                ) : (
                  filtered.map((inc) => (
                    <tr
                      key={inc.id}
                      onClick={() => setSelectedIncident(inc)}
                      className={`hover:bg-[#202020] cursor-pointer transition-colors ${selectedIncident?.id === inc.id ? 'bg-[#202020]' : ''
                        }`}
                    >
                      <td className="py-2 px-3 font-mono text-[11px] text-[#A3A3A3]">#{inc.id}</td>
                      <td className="py-2 px-3 font-medium">{inc.disaster_type}</td>
                      <td className="py-2 px-3">
                        <PriorityBadge level={inc.priority_level} />
                      </td>
                      <td className="py-2 px-3 font-mono text-emerald-400">{inc.priority_score.toFixed(1)}</td>
                      <td className="py-2 px-3">
                        <StatusBadge status={inc.status} />
                      </td>
                      <td className="py-2 px-3 text-right">
                        <span className="text-xs text-blue-400 hover:text-blue-300 font-semibold font-mono">
                          Details &rarr;
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Incident Detail Drawer / Sidebar Panel */}
      {selectedIncident && (
        <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-[#181818] border-l border-[#2D2D2D] shadow-2xl flex flex-col min-h-0 overflow-hidden font-sans">
          {/* Header */}
          <div className="px-5 py-4 border-b border-[#2D2D2D] flex justify-between items-center bg-[#151515] flex-shrink-0">
            <div>
              <span className="text-[10px] font-mono text-[#A3A3A3]">INCIDENT #{selectedIncident.id}</span>
              <h3 className="text-sm font-semibold text-[#FAFAFA]">{selectedIncident.title}</h3>
            </div>
            <button
              onClick={() => setSelectedIncident(null)}
              className="text-[#A3A3A3] hover:text-[#FAFAFA] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Scrollable Content Container */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4 min-h-0">
            {actionMsg && (
              <div className="p-2 bg-emerald-950/60 border border-emerald-900/60 rounded text-xs text-emerald-300 font-mono">
                {actionMsg}
              </div>
            )}

            {detailLoading && (
              <div className="flex items-center justify-center py-6">
                <div className="w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
              </div>
            )}

            {/* Base Telemetry */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Disaster Type</span>
                <span className="font-medium text-[#FAFAFA]">{selectedIncident.disaster_type}</span>
              </div>
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Priority Score</span>
                <span className="font-mono text-emerald-400 font-semibold">{selectedIncident.priority_score.toFixed(1)} / 100</span>
              </div>
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Priority Level</span>
                <PriorityBadge level={selectedIncident.priority_level} />
              </div>
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Confidence</span>
                <span className="font-mono text-[#FAFAFA]">{(selectedIncident.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Language & Status Telemetry */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Detected Language</span>
                <span className="font-medium text-[#FAFAFA]">{selectedIncident.language_detected || 'Unknown'}</span>
              </div>
              <div className="bg-[#111111] p-2 rounded border border-[#2D2D2D]">
                <span className="text-[10px] text-[#A3A3A3] block font-mono">Status</span>
                <span className="font-medium text-cyan-400">{selectedIncident.status}</span>
              </div>
            </div>

            {/* Generate Response Brief Button */}
            <button
              onClick={() => setShowBrief(true)}
              className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-500 text-white rounded font-bold flex items-center justify-center gap-1.5 transition-colors text-xs"
            >
              <Clipboard className="w-4 h-4" />
              <span>Generate Response Brief</span>
            </button>

            {selectedIncident.description && (
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-[#A3A3A3]">Citizen Report Text</span>
                <div className="p-2.5 bg-[#111111] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA]">
                  "{selectedIncident.description}"
                </div>
              </div>
            )}

            {/* Attachments Section */}
            <div className="space-y-3 bg-[#111111] p-3.5 rounded border border-[#2D2D2D] text-xs">
              <h4 className="font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5 text-[10px]">
                <Paperclip className="w-3.5 h-3.5 text-blue-500" /> Attachments & Documents
              </h4>

              {selectedIncidentDetail?.attachments && selectedIncidentDetail.attachments.length > 0 ? (
                <div className="space-y-1.5">
                  {selectedIncidentDetail.attachments.map((att: any) => (
                    <div key={att.id} className="bg-[#181818] p-2 rounded border border-[#2D2D2D] flex items-center justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-slate-300 font-mono text-[10px] font-bold" title={att.file_name}>
                          {att.file_name}
                        </p>
                        <span className="text-[9px] text-slate-500 font-mono">
                          {(att.file_size / 1024).toFixed(1)} KB
                        </span>
                      </div>
                      <a
                        href={att.file_path.startsWith('/') ? `${BASE}${att.file_path}` : att.file_path}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2 py-0.5 bg-[#202020] hover:bg-blue-600 hover:text-white text-[10px] rounded transition-colors font-mono"
                      >
                        View
                      </a>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 font-mono text-[10px]">No additional files attached.</p>
              )}

              <div className="pt-2 border-t border-[#2D2D2D] flex flex-col gap-1.5">
                <input
                  type="file"
                  multiple
                  onChange={(e) => setSelectedFiles(e.target.files)}
                  className="w-full text-slate-400 bg-[#181818] border border-[#2D2D2D] rounded p-1 text-[10px] file:bg-[#2D2D2D] file:text-slate-200 file:border-0 file:rounded file:px-2 file:py-0.5 file:mr-2 file:hover:bg-[#3D3D3D] cursor-pointer"
                />
                <button
                  onClick={handleUpload}
                  disabled={uploading || !selectedFiles}
                  className="w-full py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-[#202020] text-white rounded font-bold flex items-center justify-center gap-1.5 transition-colors font-mono disabled:opacity-50 text-[10px]"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>{uploading ? 'Uploading...' : 'Upload'}</span>
                </button>
              </div>
            </div>

            {/* Probability Breakdown */}
            {selectedIncidentDetail?.class_probabilities && (
              <div className="space-y-1.5 pt-2 border-t border-[#2D2D2D]">
                <span className="text-[10px] font-mono text-[#A3A3A3]">Vision Model Class Breakdown</span>
                <div className="space-y-1 text-xs">
                  {Object.entries(selectedIncidentDetail.class_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => (
                      <div key={cls} className="space-y-0.5">
                        <div className="flex justify-between text-[11px]">
                          <span className={cls === selectedIncident.disaster_type ? 'text-[#FAFAFA] font-medium' : 'text-[#A3A3A3]'}>
                            {cls}
                          </span>
                          <span className="font-mono text-[10px] text-[#A3A3A3]">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bg-[#111111] h-1 rounded overflow-hidden">
                          <div
                            className={`h-full ${cls === selectedIncident.disaster_type ? 'bg-[#D97706]' : 'bg-neutral-700'}`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Dispatch Controls */}
            {(userRole === 'Responder' || userRole === 'Admin') && (
              <div className="space-y-2 pt-3 border-t border-[#2D2D2D]">
                <span className="text-[10px] font-mono text-[#D97706]">Responder Status Controls</span>
                <div className="grid grid-cols-2 gap-2">
                  {selectedIncident.status === 'Pending' && (
                    <button
                      onClick={() => handleStatusUpdate(selectedIncident.id, 'Assigned')}
                      className="col-span-2 py-1.5 bg-[#D97706] hover:bg-[#b45309] text-white text-xs font-medium rounded transition-colors"
                    >
                      Accept Incident
                    </button>
                  )}
                  <button
                    onClick={() => handleStatusUpdate(selectedIncident.id, 'En Route')}
                    disabled={selectedIncident.status === 'En Route'}
                    className="py-1.5 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] text-xs text-[#FAFAFA] rounded transition-colors disabled:opacity-40"
                  >
                    En Route
                  </button>
                  <button
                    onClick={() => handleStatusUpdate(selectedIncident.id, 'Rescue In Progress')}
                    disabled={selectedIncident.status === 'Rescue In Progress'}
                    className="py-1.5 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] text-xs text-[#FAFAFA] rounded transition-colors disabled:opacity-40"
                  >
                    In Progress
                  </button>
                  <button
                    onClick={() => handleStatusUpdate(selectedIncident.id, 'Completed')}
                    disabled={selectedIncident.status === 'Completed'}
                    className="col-span-2 py-1.5 bg-emerald-900/60 hover:bg-emerald-800/60 border border-emerald-800/60 text-emerald-300 text-xs font-medium rounded transition-colors disabled:opacity-40"
                  >
                    Mark Completed
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Response Brief Modal */}
      {showBrief && selectedIncident && (
        <ResponseBriefModal
          incidentId={selectedIncident.id}
          onClose={() => setShowBrief(false)}
        />
      )}
    </div>
  );
}

// ── ANALYTICS PAGE ─────────────────────────────────────────────────────────
function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [stats, setStats] = useState<DashStats | null>(null);
  const [loading, setLoading] = useState(true);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const [aRes, sRes] = await Promise.all([
        apiCall<AnalyticsData>('GET', '/dashboard/analytics'),
        apiCall<DashStats>('GET', '/dashboard/stats'),
      ]);
      setData(aRes);
      setStats(sRes);
    } catch {
      // Handle offline
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const renderBar = (label: string, val: number, max: number) => (
    <div key={label} className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-[#A3A3A3]">{label}</span>
        <span className="font-mono text-[#FAFAFA]">{val}</span>
      </div>
      <div className="bg-[#111111] h-1.5 rounded overflow-hidden border border-[#2D2D2D]">
        <div
          className="bg-[#D97706] h-full transition-all duration-300"
          style={{ width: `${max ? (val / max) * 100 : 0}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
      <div className="flex justify-between items-center border-b border-[#2D2D2D] pb-3">
        <div>
          <h1 className="text-base font-semibold text-[#FAFAFA]">Analytics & System Performance</h1>
          <p className="text-xs text-[#A3A3A3]">Telemetry distribution across disaster classes and responder units.</p>
        </div>
        <button
          onClick={loadAnalytics}
          className="px-3 py-1.5 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] text-xs text-[#FAFAFA] rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded">
          <span className="text-[10px] text-[#A3A3A3] font-mono block">TOTAL INCIDENTS</span>
          <span className="text-xl font-semibold text-[#FAFAFA] font-mono">{stats?.total_incidents || 0}</span>
        </div>
        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded">
          <span className="text-[10px] text-[#A3A3A3] font-mono block">AVG MODEL CONFIDENCE</span>
          <span className="text-xl font-semibold text-emerald-400 font-mono">
            {data ? `${(data.average_confidence * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded">
          <span className="text-[10px] text-[#A3A3A3] font-mono block">RESOLVED INCIDENTS</span>
          <span className="text-xl font-semibold text-[#FAFAFA] font-mono">{stats?.completed || 0}</span>
        </div>
      </div>

      {/* Grid Charts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded space-y-3">
          <h3 className="text-xs font-semibold text-[#FAFAFA]">Disaster Class Distribution</h3>
          <div className="space-y-2">
            {data &&
              Object.entries(data.disaster_distribution).map(([k, v]) =>
                renderBar(k, v, Math.max(...Object.values(data.disaster_distribution)))
              )}
          </div>
        </div>

        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded space-y-3">
          <h3 className="text-xs font-semibold text-[#FAFAFA]">Priority Level Distribution</h3>
          <div className="space-y-2">
            {data &&
              Object.entries(data.priority_distribution).map(([k, v]) =>
                renderBar(k, v, Math.max(...Object.values(data.priority_distribution)))
              )}
          </div>
        </div>

        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded space-y-3">
          <h3 className="text-xs font-semibold text-[#FAFAFA]">Rescue Status Breakdown</h3>
          <div className="space-y-2">
            {data &&
              Object.entries(data.status_distribution).map(([k, v]) =>
                renderBar(k, v, Math.max(...Object.values(data.status_distribution)))
              )}
          </div>
        </div>

        <div className="bg-[#181818] border border-[#2D2D2D] p-4 rounded space-y-3">
          <h3 className="text-xs font-semibold text-[#FAFAFA]">Language Distribution</h3>
          <div className="space-y-2">
            {data &&
              Object.entries(data.language_distribution).map(([k, v]) =>
                renderBar(k, v, Math.max(...Object.values(data.language_distribution)))
              )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── ROOT APP ───────────────────────────────────────────────────────────────
function App() {
  const [tab, setTab] = useState<Tab>('citizen');
  const [user, setUser] = useState<AuthUser | null>(null);
  const [showLogin, setShowLogin] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('disaster_triage_token');
    if (!token) return;
    apiCall<any>('GET', '/auth/me')
      .then((u) => setUser({ email: u.email, role: u.role, full_name: u.full_name }))
      .catch(() => localStorage.removeItem('disaster_triage_token'));
  }, []);

  return (
    <div className="min-h-screen bg-[#111111] text-[#FAFAFA] flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-[#2D2D2D] bg-[#181818] px-4 py-2.5 flex items-center justify-between">
        <div
          onClick={() => setTab('citizen')}
          className="flex items-center gap-2 cursor-pointer select-none"
        >
          <Shield className="w-4 h-4 text-[#D97706]" />
          <span className="text-xs font-bold tracking-tight text-[#FAFAFA]">DISASTER TRIAGE</span>
          <span className="text-[10px] font-mono text-[#A3A3A3] hidden sm:inline">v2.0</span>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-[#111111] p-1 rounded border border-[#2D2D2D]">
          <button
            onClick={() => setTab('citizen')}
            className={`px-3 py-1 text-xs rounded transition-colors flex items-center gap-1.5 ${tab === 'citizen' ? 'bg-[#202020] text-[#FAFAFA] font-medium' : 'text-[#A3A3A3] hover:text-[#FAFAFA]'
              }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Citizen Portal</span>
          </button>
          <button
            onClick={() => setTab('dashboard')}
            className={`px-3 py-1 text-xs rounded transition-colors flex items-center gap-1.5 ${tab === 'dashboard' ? 'bg-[#202020] text-[#FAFAFA] font-medium' : 'text-[#A3A3A3] hover:text-[#FAFAFA]'
              }`}
          >
            <Radio className="w-3.5 h-3.5" />
            <span>Rescue Command</span>
          </button>
          <button
            onClick={() => setTab('analytics')}
            className={`px-3 py-1 text-xs rounded transition-colors flex items-center gap-1.5 ${tab === 'analytics' ? 'bg-[#202020] text-[#FAFAFA] font-medium' : 'text-[#A3A3A3] hover:text-[#FAFAFA]'
              }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>Analytics</span>
          </button>
        </nav>

        {/* Auth status */}
        <div className="flex items-center gap-2">
          {user ? (
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[#A3A3A3] hidden sm:inline">{user.full_name}</span>
              <span className="px-1.5 py-0.5 bg-[#202020] border border-[#2D2D2D] rounded text-[10px] font-mono text-[#D97706]">
                {user.role}
              </span>
              <button
                onClick={() => {
                  localStorage.removeItem('disaster_triage_token');
                  setUser(null);
                }}
                className="p-1 text-[#A3A3A3] hover:text-red-400 transition-colors"
                title="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowLogin(true)}
              className="px-2.5 py-1 bg-[#202020] hover:bg-[#2A2A2A] border border-[#2D2D2D] rounded text-xs text-[#FAFAFA] transition-colors flex items-center gap-1.5"
            >
              <LogIn className="w-3.5 h-3.5 text-[#D97706]" />
              <span>Responder Access</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Page Body */}
      <main className="flex-1">
        {tab === 'citizen' && <CitizenPortal />}
        {tab === 'dashboard' && (
          <RescueDashboard
            userRole={user?.role || 'Citizen'}
            onLoginSuccess={(u) => {
              setUser(u);
              setShowLogin(false);
              setTab('dashboard');
            }}
          />
        )}
        {tab === 'analytics' && <AnalyticsPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#2D2D2D] bg-[#181818] px-4 py-2 text-[11px] font-mono text-[#A3A3A3] flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <span>System Operational</span>
        </div>
        <div>FastAPI · PyTorch Ensemble · CatBoost Regressor</div>
      </footer>

      {showLogin && tab !== 'dashboard' && (
        <LoginModal
          onClose={() => setShowLogin(false)}
          onSuccess={(u) => {
            setUser(u);
            setShowLogin(false);
            setTab('dashboard');
          }}
        />
      )}
    </div>
  );
}

// ── Mount ──────────────────────────────────────────────────────────────────
const root = document.getElementById('root')!;
createRoot(root).render(<App />);
