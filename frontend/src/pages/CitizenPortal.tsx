import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, ArrowRight, ArrowUpRight, Camera, Check, CheckCircle2, ChevronDown, Crosshair, FileImage, Globe2, ImagePlus, Info, Layers3, MapPin, Navigation, RadioTower, RotateCcw, Send, ShieldCheck, Sparkles, UploadCloud, X } from 'lucide-react';
import { createIncidentApi } from '../api/incidents';
import { readableApiError } from '../api/client';
import { LeafletMap } from '../components/LeafletMap';
import type { IncidentDetail } from '../types';

interface CitizenPortalProps {
  onViewOperations: () => void;
  apiAvailable?: boolean;
}

const samples = [
  { label: 'English · Flood', text: 'Flood water is rising rapidly around the neighbourhood. Several residents are trapped on rooftops and need assessment.' },
  { label: 'हिन्दी · भूकंप', text: 'भूकंप के बाद एक इमारत क्षतिग्रस्त है, अंदर कुछ लोग फंसे हो सकते हैं। सहायता की जरूरत है।' },
  { label: 'Hinglish · Fire', text: 'Building mein aag lag gayi hai, smoke bohot hai. Log bahar wait kar rahe hain, help chahiye.' },
];
const maxBytes = 10 * 1024 * 1024;

function tone(level: string) {
  if (level === 'Critical') return 'critical';
  if (level === 'High') return 'high';
  if (level === 'Medium') return 'medium';
  return 'low';
}
function validCoordinates(lat: string, lng: string) {
  if (!lat.trim() || !lng.trim()) return false;
  const a = Number(lat), b = Number(lng);
  return Number.isFinite(a) && Number.isFinite(b) && a >= -90 && a <= 90 && b >= -180 && b <= 180;
}

export const CitizenPortal: React.FC<CitizenPortalProps> = ({ onViewOperations, apiAvailable = true }) => {
  const [description, setDescription] = useState('');
  const [photo, setPhoto] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [address, setAddress] = useState('');
  const [showMap, setShowMap] = useState(false);
  const [locating, setLocating] = useState(false);
  const [locationSource, setLocationSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IncidentDetail | null>(null);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!photo) { setPreview(''); return; }
    const url = URL.createObjectURL(photo);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  const coordsOK = validCoordinates(latitude, longitude);
  const formOK = (description.trim().length >= 10 || photo !== null) && coordsOK;
  const choosePhoto = (file?: File) => {
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Choose a JPG, PNG or WEBP image.'); return;
    }
    if (file.size > maxBytes) { setError('Images must be 10 MB or smaller.'); return; }
    setPhoto(file); setError('');
  };
  const locate = () => {
    if (!navigator.geolocation) { setError('Geolocation is not supported. Enter coordinates manually or select on the map.'); return; }
    setLocating(true); setError('');
    navigator.geolocation.getCurrentPosition(
      position => {
        setLatitude(String(position.coords.latitude));
        setLongitude(String(position.coords.longitude));
        setLocationSource('GPS location selected'); setLocating(false);
      },
      () => { setError('Could not get your location. Choose a point on the map or enter coordinates manually.'); setLocating(false); },
      { timeout: 10000, maximumAge: 120000, enableHighAccuracy: false }
    );
  };
  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!formOK) {
      setError('Add a report (at least 10 characters) or photo, then confirm a valid location.'); return;
    }
    setLoading(true); setError(''); setResult(null);
    const form = new FormData();
    if (description.trim()) form.append('description', description.trim());
    if (photo) form.append('image', photo);
    form.append('latitude', String(Number(latitude)));
    form.append('longitude', String(Number(longitude)));
    if (address.trim()) form.append('address_text', address.trim());
    form.append('application_context', 'Citizen Portal');
    try {
      const response = await createIncidentApi(form);
      setResult(response);
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    } catch (err) {
      setError(readableApiError(err, 'The report could not be submitted. Please retry.'));
    } finally { setLoading(false); }
  };
  const reset = () => {
    setDescription(''); setPhoto(null); setResult(null); setError('');
    setLatitude(''); setLongitude(''); setAddress(''); setLocationSource(''); setShowMap(false);
    if (inputRef.current) inputRef.current.value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="page-stack">
      <section className="report-hero">
        <div className="report-hero-copy">
          <div className="overline on-dark"><span className="tiny-live-dot" /> MULTILINGUAL RESPONSE RESEARCH</div>
          <h1>Every report matters.<br /><em>Clarity saves time.</em></h1>
          <p>Share an incident description, optional photograph and a confirmed location. Our research models classify the report and estimate priority for human review.</p>
          <a className="hero-link" href="#report-form">Create a report <ArrowRight size={16} /></a>
          <div className="hero-tags"><span><Globe2 size={14} /> English · हिन्दी · Hinglish</span><span><Layers3 size={14} /> Text + image analysis</span></div>
        </div>
        <div className="signal-art" aria-hidden="true">
          <div className="signal-ring r1" /><div className="signal-ring r2" /><div className="signal-ring r3" />
          <div className="signal-center"><RadioTower size={42} /></div>
          <div className="floating-signal signal-one"><span>01</span><strong>Report</strong><small>Received</small></div>
          <div className="floating-signal signal-two"><span>02</span><strong>Analyze</strong><small>Model output</small></div>
          <div className="floating-signal signal-three"><span>03</span><strong>Review</strong><small>Human action</small></div>
        </div>
      </section>

      <div className="section-intro">
        <div><div className="overline">CITIZEN REPORTING WORKFLOW</div><h2>Tell us what happened.</h2><p>Three simple steps to prepare a report for review.</p></div>
        <span className="section-count">01 / REPORT</span>
      </div>

      <div className="report-grid" id="report-form">
        <form className="dt-card form-card" onSubmit={onSubmit}>
          <div className="form-section-head"><span className="step-number">01</span><div><h3>Describe the situation</h3><p>Your report can be written in English, Hindi or Hinglish.</p></div></div>
          <label className="field-label" htmlFor="incident-description">What is happening?</label>
          <textarea id="incident-description" rows={5} maxLength={5000} className="dt-textarea report-textarea"
            placeholder="Describe what you can see, who may be affected, and why the situation needs attention…"
            value={description} onChange={e => setDescription(e.target.value)} />
          <div className="input-meta"><span>Keep personal identifying details out of prototype reports.</span><span>{description.length} / 5000</span></div>
          <div className="sample-row"><span>TRY A SAMPLE</span>{samples.map(s => <button key={s.label} type="button" className="sample-chip" onClick={() => setDescription(s.text)}>{s.label}</button>)}</div>

          <div className="form-rule" />
          <div className="form-section-head"><span className="step-number">02</span><div><h3>Add visual context <span className="optional">Optional</span></h3><p>Photos can help the classifier understand the scene.</p></div></div>
          <div className={'upload-zone' + (photo ? ' has-image' : '')} onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); choosePhoto(e.dataTransfer.files?.[0]); }}>
            {photo && preview ? (
              <div className="preview-photo">
                <img src={preview} alt="Selected disaster report upload" />
                <div><strong>{photo.name}</strong><small>{(photo.size / 1024 / 1024).toFixed(2)} MB</small></div>
                <button type="button" className="icon-button" onClick={() => { setPhoto(null); if(inputRef.current)inputRef.current.value=''; }} aria-label="Remove image"><X size={17} /></button>
              </div>
            ) : (
              <label htmlFor="incident-photo" className="upload-label"><span className="upload-icon"><ImagePlus size={25} /></span><strong>Drop your image here, or browse files</strong><small>JPG, PNG or WEBP · Maximum 10 MB</small></label>
            )}
            <input id="incident-photo" ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp"
              onChange={e => choosePhoto(e.target.files?.[0])} className="visually-hidden" />
          </div>

          <div className="form-rule" />
          <div className="form-section-head"><span className="step-number">03</span><div><h3>Confirm incident location</h3><p>We never assume that your device location is where the incident happened.</p></div></div>
          <div className="location-actions">
            <button type="button" className="btn btn-outline" disabled={locating} onClick={locate}><Crosshair size={16} />{locating ? 'Locating…' : 'Use my current location'}</button>
            <button type="button" className="btn btn-quiet" onClick={() => setShowMap(!showMap)}><MapPin size={16} /> {showMap ? 'Hide map' : 'Choose on map'} <ChevronDown size={14} /></button>
          </div>
          {showMap && <div className="map-picker"><LeafletMap center={coordsOK ? [Number(latitude), Number(longitude)] : [20.5937, 78.9629]}
            zoom={coordsOK ? 12 : 5} interactiveSelect selectedLocation={coordsOK ? { latitude: Number(latitude), longitude: Number(longitude) } : undefined}
            onSelectLocation={(lat, lng) => { setLatitude(String(lat)); setLongitude(String(lng)); setLocationSource('Map location selected'); }} /></div>}
          <div className="two-fields">
            <div><label className="field-label" htmlFor="latitude">Latitude</label><input className="dt-input" id="latitude" type="number" step="any" min="-90" max="90" value={latitude} placeholder="e.g. 28.6139" onChange={e => {setLatitude(e.target.value);setLocationSource('Manual coordinates');}} /></div>
            <div><label className="field-label" htmlFor="longitude">Longitude</label><input className="dt-input" id="longitude" type="number" step="any" min="-180" max="180" value={longitude} placeholder="e.g. 77.2090" onChange={e => {setLongitude(e.target.value);setLocationSource('Manual coordinates');}} /></div>
          </div>
          <label className="field-label" htmlFor="address">Area or landmark <span className="optional">Optional</span></label>
          <input id="address" className="dt-input" maxLength={250} value={address} onChange={e => setAddress(e.target.value)} placeholder="e.g. Near the main bridge, Zone 3" />
          <div className={'location-confirm' + (coordsOK ? ' confirmed' : '')}><Navigation size={15} /> {coordsOK ? (locationSource || 'Coordinates entered') + ' · ' + Number(latitude).toFixed(4) + ', ' + Number(longitude).toFixed(4) : 'Select a location to enable submission'}</div>

          {error && <div className="alert-banner error" role="alert"><AlertCircle size={18} />{error}</div>}
          {!apiAvailable && <div className="alert-banner warning" role="status">Backend appears offline. Start the FastAPI service before submitting.</div>}
          <button type="submit" className="btn btn-coral btn-submit" disabled={loading || !formOK || !apiAvailable}>
            {loading ? <><span className="button-spinner" /> Analyzing report…</> : <><Send size={18} /> Submit for triage <ArrowUpRight size={17} /></>}
          </button>
          <div className="under-submit"><ShieldCheck size={15} /> Creates a research database incident, not an emergency dispatch.</div>
        </form>

        <aside className="report-aside">
          <div className="dt-card process-card">
            <div className="card-overline">HOW IT WORKS</div><h3>A thoughtful signal,<br />not an automatic decision.</h3>
            <div className="process-line"><div className="process-icon peach"><FileImage size={19} /></div><div><strong>01 / Collect</strong><p>Combine a written report, optional photo and coordinates.</p></div></div>
            <div className="process-line"><div className="process-icon blue"><Sparkles size={19} /></div><div><strong>02 / Analyze</strong><p>Configured models estimate category, confidence and priority.</p></div></div>
            <div className="process-line"><div className="process-icon mint"><CheckCircle2 size={19} /></div><div><strong>03 / Human review</strong><p>Authorized users can inspect and update cases in the command view.</p></div></div>
          </div>
          <div className="dt-card architecture-card"><div className="card-overline">UNDER THE HOOD</div><h3>Multimodal by design</h3>
            <div className="architecture-item"><Globe2 size={18} /><div><strong>Multilingual NLP</strong><span>English, Hindi and Hinglish scoring</span></div></div>
            <div className="architecture-item"><Camera size={18} /><div><strong>Vision models</strong><span>Image classification when a photograph is attached</span></div></div>
            <div className="architecture-item"><ShieldCheck size={18} /><div><strong>Role-aware workflow</strong><span>Citizen reporting, responders and admin review</span></div></div>
          </div>
          <div className="aside-info"><Info size={18} /><p>Model outputs are research estimates and can be wrong. They are not medical or emergency instructions.</p></div>
        </aside>
      </div>

      {result && (
        <div ref={resultRef} id="triage-results" className="result-wrap">
          <div className="result-header"><div className="result-check"><Check size={24} /></div><div><div className="overline">REPORT RECORDED</div><h2>Triage analysis ready</h2><p>Reference #{result.id} · Status: {result.status}. No responders have been automatically dispatched.</p></div></div>
          <div className="result-metrics">
            <div><span>Predicted category</span><strong>{result.disaster_type || 'Unclassified'}</strong></div>
            <div><span>Priority score</span><strong>{Number(result.priority_score).toFixed(1)}<small> / 100</small></strong></div>
            <div><span>Priority level</span><strong className={'priority-text ' + tone(result.priority_level)}>{result.priority_level}</strong></div>
            <div><span>Language</span><strong>{result.language_detected || 'Not determined'}</strong></div>
          </div>
          {result.recommendations?.length > 0 && <div className="result-panel"><h3>Suggested review actions <span>AI / rule-derived</span></h3><ul>{result.recommendations.map((item, index) => <li key={index}><Check size={15} /> {item}</li>)}</ul></div>}
          {Object.keys(result.class_probabilities || {}).length > 0 && <div className="result-panel"><h3>Model class probabilities</h3><div className="prob-grid">{Object.entries(result.class_probabilities).sort((a,b)=>b[1]-a[1]).map(([key,val]) =>
            <div className="prob-row" key={key}><span>{key}</span><div className="prob-track"><i style={{ width: Math.max(0,Math.min(100,Number(val)*100)) + '%' }} /></div><strong>{(Number(val)*100).toFixed(1)}%</strong></div>)}</div></div>}
          <div className="result-actions"><button className="btn btn-dark" type="button" onClick={onViewOperations}>Open command view <ArrowRight size={16} /></button><button className="btn btn-outline" type="button" onClick={reset}><RotateCcw size={16} /> Start a new report</button></div>
        </div>
      )}
    </div>
  );
};
