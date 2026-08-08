import React, { useState, useEffect } from 'react';
import { IncidentDetail, UserRole } from '../types';
import { getIncidentByIdApi, updateIncidentApi } from '../api/incidents';
import { acceptIncidentApi, updateResponderStatusApi } from '../api/responder';
import { ProbabilityBars } from './ProbabilityBars';
import { RecommendationCard } from './RecommendationCard';
import { ResponseBriefModal } from './ResponseBriefModal';
import { X, Clock, MapPin, UserCheck, Shield, CheckCircle, Flame, AlertCircle, Clipboard } from 'lucide-react';

interface IncidentDrawerProps {
  incidentId: number | null;
  userRole?: UserRole;
  onClose: () => void;
  onIncidentUpdated?: () => void;
}

export const IncidentDrawer: React.FC<IncidentDrawerProps> = ({
  incidentId,
  userRole = 'Citizen',
  onClose,
  onIncidentUpdated,
}) => {
  const [incident, setIncident] = useState<IncidentDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [showBrief, setShowBrief] = useState(false);

  useEffect(() => {
    if (!incidentId) return;
    setLoading(true);
    setError(null);
    getIncidentByIdApi(incidentId)
      .then((data) => setIncident(data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load incident details.'))
      .finally(() => setLoading(false));
  }, [incidentId]);

  if (!incidentId) return null;

  const handleAccept = async () => {
    if (!incident) return;
    setActionLoading(true);
    try {
      const updated = await acceptIncidentApi(incident.id, notes || 'Accepted by responder');
      setIncident(updated);
      setNotes('');
      if (onIncidentUpdated) onIncidentUpdated();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to accept incident.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleStatusUpdate = async (newStatus: string) => {
    if (!incident) return;
    setActionLoading(true);
    try {
      const updated = await updateResponderStatusApi(incident.id, newStatus, notes || `Status updated to ${newStatus}`);
      setIncident(updated);
      setNotes('');
      if (onIncidentUpdated) onIncidentUpdated();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update status.');
    } finally {
      setActionLoading(false);
    }
  };

  const canManage = userRole === 'Responder' || userRole === 'Admin';

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/75 backdrop-blur-sm transition-opacity">
      <div className="w-full max-w-2xl bg-slate-900 border-l border-slate-800 h-full flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/50">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">INCIDENT #{incidentId}</span>
              {incident && (
                <span className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded ${
                  incident.priority_level === 'Critical' ? 'bg-red-950 text-red-400 border border-red-800' :
                  incident.priority_level === 'High' ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-slate-800 text-slate-300'
                }`}>
                  {incident.priority_level} Priority
                </span>
              )}
            </div>
            <h2 className="text-lg font-bold text-slate-100 truncate mt-0.5">
              {incident?.title || 'Loading Incident Telemetry...'}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading && (
            <div className="flex items-center justify-center py-20">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-950/60 border border-red-800 rounded-xl text-red-300 text-sm flex items-center gap-2">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {incident && !loading && (
            <>
              {/* Media & Key Telemetry Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {incident.image_url ? (
                  <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 relative h-48">
                    <img
                      src={incident.image_url}
                      alt="Disaster report photo"
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-slate-950/80 text-[10px] font-mono text-slate-300 rounded backdrop-blur">
                      Citizen Image Upload
                    </span>
                  </div>
                ) : (
                  <div className="rounded-xl border border-slate-800/80 bg-slate-950/50 h-48 flex flex-col items-center justify-center text-slate-500">
                    <Flame className="w-8 h-8 mb-2 opacity-50" />
                    <span className="text-xs">No Photo Uploaded</span>
                  </div>
                )}

                <div className="space-y-3 bg-slate-950/40 p-4 rounded-xl border border-slate-800/60 text-xs font-mono">
                  <div className="flex justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-400">Disaster Type</span>
                    <span className="text-amber-400 font-bold">{incident.disaster_type}</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-400">Priority Score</span>
                    <span className="text-emerald-400 font-bold">{incident.priority_score.toFixed(1)} / 100</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-400">Confidence</span>
                    <span className="text-blue-400 font-bold">{(incident.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-400">Detected Language</span>
                    <span className="text-purple-400 font-bold">{incident.language_detected}</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-1.5">
                    <span className="text-slate-400">Status</span>
                    <span className="text-cyan-400 font-bold">{incident.status}</span>
                  </div>
                  <button
                    onClick={() => setShowBrief(true)}
                    className="w-full mt-2 py-2 px-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold flex items-center justify-center gap-1.5 transition-colors font-sans text-xs"
                  >
                    <Clipboard className="w-4 h-4" />
                    <span>Generate Response Brief</span>
                  </button>
                </div>
              </div>

              {/* Citizen Text Report */}
              {incident.description && (
                <div className="space-y-1.5">
                  <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Citizen Text Report</label>
                  <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl text-sm text-slate-200 leading-relaxed font-sans">
                    "{incident.description}"
                  </div>
                </div>
              )}

              {/* Response Directives */}
              <RecommendationCard
                recommendations={incident.recommendations}
                priorityLevel={incident.priority_level}
                disasterType={incident.disaster_type}
              />

              {/* Probability Breakdown */}
              <div className="space-y-2.5">
                <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Model Probability Breakdown (All 6 Classes)
                </h4>
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl">
                  <ProbabilityBars
                    probabilities={incident.class_probabilities}
                    predictedClass={incident.disaster_type}
                  />
                </div>
              </div>

              {/* Status History Timeline */}
              {incident.status_history && incident.status_history.length > 0 && (
                <div className="space-y-2.5">
                  <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider">Incident Status Audit Log</h4>
                  <div className="space-y-2 border-l-2 border-slate-800 pl-4">
                    {incident.status_history.map((hist) => (
                      <div key={hist.id} className="relative text-xs space-y-0.5">
                        <div className="absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full bg-blue-500 border-2 border-slate-900" />
                        <div className="flex justify-between font-mono text-[11px] text-slate-400">
                          <span>{hist.new_status}</span>
                          <span>{new Date(hist.created_at).toLocaleTimeString()}</span>
                        </div>
                        {hist.notes && <p className="text-slate-300 text-[11px]">{hist.notes}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Responder Action Controls */}
              {canManage && (
                <div className="pt-4 border-t border-slate-800 space-y-3 bg-slate-950/60 -mx-6 -mb-6 p-6">
                  <h4 className="text-xs font-mono text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                    <UserCheck className="w-4 h-4" /> Responder Dispatch Controls
                  </h4>

                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Enter dispatch notes / status update details..."
                    className="w-full p-2.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-100 focus:outline-none focus:border-blue-500 h-16 resize-none"
                  />

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {incident.status === 'Pending' && (
                      <button
                        onClick={handleAccept}
                        disabled={actionLoading}
                        className="py-2 px-3 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-bold transition-colors disabled:opacity-50 col-span-2 md:col-span-4"
                      >
                        {actionLoading ? 'Processing...' : 'Accept Incident'}
                      </button>
                    )}

                    <button
                      onClick={() => handleStatusUpdate('En Route')}
                      disabled={actionLoading || incident.status === 'En Route'}
                      className="py-2 px-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 text-white rounded-lg text-xs font-semibold transition-colors"
                    >
                      En Route
                    </button>

                    <button
                      onClick={() => handleStatusUpdate('Rescue In Progress')}
                      disabled={actionLoading || incident.status === 'Rescue In Progress'}
                      className="py-2 px-2 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 text-white rounded-lg text-xs font-semibold transition-colors"
                    >
                      In Progress
                    </button>

                    <button
                      onClick={() => handleStatusUpdate('Completed')}
                      disabled={actionLoading || incident.status === 'Completed'}
                      className="py-2 px-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 text-white rounded-lg text-xs font-semibold transition-colors col-span-2"
                    >
                      Mark Completed
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      {showBrief && incident && (
        <ResponseBriefModal
          incidentId={incident.id}
          onClose={() => setShowBrief(false)}
        />
      )}
    </div>
  );
};
