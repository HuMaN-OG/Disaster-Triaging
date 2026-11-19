import React, { useEffect, useState } from 'react';
import { AlertCircle, ArrowRight, CalendarClock, Check, CheckCircle2, ClipboardList, Clock3, Download, FileText, Image as ImageIcon, MapPin, Paperclip, Send, ShieldAlert, ShieldCheck, UploadCloud, X } from 'lucide-react';
import { getIncidentByIdApi, uploadAttachmentsApi } from '../api/incidents';
import { acceptIncidentApi, updateResponderStatusApi } from '../api/responder';
import { getApiAssetUrl, readableApiError } from '../api/client';
import { ResponseBriefModal } from './ResponseBriefModal';
import type { IncidentDetail, UserRole } from '../types';

interface IncidentDrawerProps {
  incidentId: number | null;
  userRole?: UserRole;
  onClose: () => void;
  onIncidentUpdated?: () => void;
}

const nextStage: Record<string, {label: string; status: string}> = {
  'Assigned': {label:'Mark team en route',status:'En Route'},
  'En Route': {label:'Start rescue work',status:'Rescue In Progress'},
  'Rescue In Progress': {label:'Mark completed',status:'Completed'},
};

function dateTime(value?: string) {
  if (!value) return '—';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString(undefined, {dateStyle:'medium',timeStyle:'short'});
}
function priorityClass(level: string) {
  return level === 'Critical' ? 'critical' : level === 'High' ? 'high' : level === 'Medium' ? 'medium' : 'low';
}

export const IncidentDrawer: React.FC<IncidentDrawerProps> = ({
  incidentId, userRole = 'Citizen', onClose, onIncidentUpdated,
}) => {
  const [incident, setIncident] = useState<IncidentDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notes, setNotes] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [briefOpen, setBriefOpen] = useState(false);
  const canManage = userRole === 'Responder' || userRole === 'Admin';

  useEffect(() => {
    if (incidentId == null) { setIncident(null); setError(''); return; }
    let mounted = true;
    setLoading(true); setError(''); setIncident(null); setNotes(''); setFiles([]);
    getIncidentByIdApi(incidentId).then(value => {if(mounted)setIncident(value);})
      .catch(e => {if(mounted)setError(readableApiError(e,'Failed to load incident.'));})
      .finally(() => {if(mounted)setLoading(false);});
    return () => {mounted=false;};
  }, [incidentId]);

  useEffect(() => {
    if (incidentId == null) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const key = (e: KeyboardEvent) => {if(e.key === 'Escape' && !briefOpen)onClose();};
    document.addEventListener('keydown', key);
    return () => {document.body.style.overflow = previous;document.removeEventListener('keydown',key);};
  }, [incidentId, onClose, briefOpen]);

  if (incidentId == null) return null;

  const updateWorkflow = async (status?: string) => {
    if (!incident || busy) return;
    setBusy(true); setError('');
    try {
      const result = status ? await updateResponderStatusApi(incident.id,status,notes.trim() || undefined)
        : await acceptIncidentApi(incident.id,notes.trim() || undefined);
      setIncident(result); setNotes(''); onIncidentUpdated?.();
    } catch(e) { setError(readableApiError(e,'Could not change incident status.')); }
    finally {setBusy(false);}
  };
  const upload = async () => {
    if (!incident || !files.length || busy) return;
    if (files.some(x=>x.size>10*1024*1024)){setError('Each attachment must be 10 MB or smaller.');return;}
    setBusy(true);setError('');
    try {
      await uploadAttachmentsApi(incident.id,files);
      setIncident(await getIncidentByIdApi(incident.id));
      setFiles([]);onIncidentUpdated?.();
    } catch(e) {setError(readableApiError(e,'Could not upload attachments.'));}
    finally {setBusy(false);}
  };
  const next = incident ? nextStage[incident.status] : undefined;
  const image = incident ? getApiAssetUrl(incident.image_url) : '';

  return (
    <>
      <div className="drawer-overlay" onMouseDown={event=>{if(event.target===event.currentTarget)onClose();}}>
        <aside className="incident-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
          <header className="drawer-header">
            <div><div className="overline">INCIDENT INTELLIGENCE / #{incidentId}</div>
              <h2 id="drawer-title">{incident?.title || (loading?'Loading incident…':'Incident details')}</h2>
            </div>
            <button className="icon-button" type="button" aria-label="Close incident details" onClick={onClose}><X size={21}/></button>
          </header>

          <div className="drawer-body">
            {loading && <div className="loading-panel"><span className="button-spinner dark-spinner"/><strong>Loading case details…</strong></div>}
            {error && <div className="alert-banner error" role="alert"><AlertCircle size={18}/>{error}</div>}
            {incident && !loading && (
              <>
                <div className="drawer-pills"><span className={'priority-pill '+priorityClass(incident.priority_level)}>{incident.priority_level} priority</span><span className="status-pill">{incident.status}</span><span className="drawer-time"><Clock3 size={14}/>{dateTime(incident.created_at)}</span></div>
                <div className="drawer-summary-grid">
                  <div><small>Predicted disaster</small><strong>{incident.disaster_type}</strong></div>
                  <div><small>Priority score</small><strong>{incident.priority_score.toFixed(1)} <em>/ 100</em></strong></div>
                  <div><small>Model confidence</small><strong>{(incident.confidence*100).toFixed(1)}%</strong></div>
                  <div><small>Detected language</small><strong>{incident.language_detected||'—'}</strong></div>
                </div>
                {image && <div className="drawer-media"><img src={image} alt="Photograph submitted with this report"/><span><ImageIcon size={14}/> Report image</span></div>}
                <section className="drawer-section"><h3><FileText size={18}/> Citizen report</h3><p className="report-quote">{incident.description || 'Image-only submission; no text description supplied.'}</p></section>
                {incident.location && <section className="drawer-section"><h3><MapPin size={18}/> Location</h3><div className="location-summary"><strong>{incident.location.address_text || 'Coordinates provided'}</strong><small>{incident.location.latitude.toFixed(5)}, {incident.location.longitude.toFixed(5)}</small></div></section>}
                <section className="drawer-section"><div className="drawer-section-title"><h3><ClipboardList size={18}/> Response brief</h3><button className="btn btn-outline btn-small" type="button" onClick={()=>setBriefOpen(true)}>Open brief <ArrowRight size={15}/></button></div>
                  <p className="drawer-helper">Confirmed incident details, derived risk assessment, suggested resources and timeline—generated from the backend.</p></section>

                {incident.recommendations?.length>0 && <section className="drawer-section"><h3><ShieldCheck size={18}/> Suggested review actions</h3><ul className="recommendation-list">{incident.recommendations.map((r,i)=><li key={i}><Check size={15}/>{r}</li>)}</ul><p className="drawer-helper">AI- and rule-derived suggestions; require human verification.</p></section>}
                {Object.keys(incident.class_probabilities||{}).length>0 && <section className="drawer-section"><h3>Model probabilities</h3><div className="drawer-probs">{Object.entries(incident.class_probabilities).sort((a,b)=>b[1]-a[1]).map(([label,value])=><div key={label}><span>{label}</span><div><i style={{width:Math.max(0,Math.min(100,value*100))+'%'}}/></div><b>{(value*100).toFixed(1)}%</b></div>)}</div></section>}

                <section className="drawer-section"><h3><Paperclip size={18}/> Attachments</h3>
                  {incident.attachments?.length ? <div className="attachment-list">{incident.attachments.map(a=><div className="attachment-row" key={a.id}><Paperclip size={16}/><span><strong>{a.file_name}</strong><small>{(a.file_size/1024).toFixed(1)} KB</small></span>
                    {getApiAssetUrl(a.file_path) && <a href={getApiAssetUrl(a.file_path)} target="_blank" rel="noopener noreferrer">View <ArrowRight size={14}/></a>}</div>)}</div>
                  :<p className="drawer-helper">No additional attachments yet.</p>}
                  {canManage && <div className="attachment-uploader"><label className="file-input-label" htmlFor="attachment-input"><UploadCloud size={18}/> Add evidence files</label><input id="attachment-input" type="file" multiple onChange={e=>setFiles(Array.from(e.target.files||[]))}/>{files.length>0 && <button className="btn btn-outline btn-small" type="button" disabled={busy} onClick={()=>void upload()}>Upload {files.length} file{files.length>1?'s':''}</button>}</div>}
                </section>

                <section className="drawer-section"><h3><CalendarClock size={18}/> Status history</h3>
                  {incident.status_history?.length ? <ol className="timeline">{incident.status_history.map(h=><li key={h.id}><span className="timeline-dot"/><div><strong>{h.new_status}</strong><small>{dateTime(h.created_at)}</small>{h.notes && <p>{h.notes}</p>}</div></li>)}</ol>
                  :<p className="drawer-helper">No status changes recorded yet.</p>}
                </section>

                <section className="drawer-actions-section"><div className="overline">HUMAN-AUTHORIZED ACTIONS</div>
                  <h3>Manage this case</h3>
                  {canManage ? <><label className="field-label" htmlFor="responder-notes">Responder notes <span className="optional">Optional</span></label><textarea id="responder-notes" className="dt-textarea" value={notes} onChange={e=>setNotes(e.target.value)} rows={3} placeholder="Add relevant action notes before updating…"/>
                    {incident.status==='Pending' && <button type="button" className="btn btn-coral btn-submit" disabled={busy} onClick={()=>void updateWorkflow()}><ShieldCheck size={17}/> {busy?'Working…':'Accept incident'}</button>}
                    {next && <button type="button" className="btn btn-coral btn-submit" disabled={busy} onClick={()=>void updateWorkflow(next.status)}><Send size={17}/> {busy?'Updating…':next.label}</button>}
                    {incident.status==='Completed' && <div className="completion-note"><CheckCircle2 size={20}/> Workflow marked completed.</div>}
                    <p className="drawer-helper">The status change is saved in the backend and reflected in the incident timeline.</p>
                  </>:<div className="access-restricted"><ShieldAlert size={19}/><span>Read-only view. Authorized Responder or Admin sign-in is required to change status.</span></div>}
                </section>
              </>
            )}
          </div>
          <footer className="drawer-footer"><span>Research prototype · no real dispatch</span><button type="button" className="btn btn-outline btn-small" onClick={onClose}>Close</button></footer>
        </aside>
      </div>
      {briefOpen && incidentId!=null && <ResponseBriefModal incidentId={incidentId} onClose={()=>setBriefOpen(false)}/>}
    </>
  );
};
