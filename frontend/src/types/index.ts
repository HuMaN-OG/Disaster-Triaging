export type UserRole = 'Citizen' | 'Responder' | 'Admin';

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  role: UserRole;
  user_id: number;
  full_name: string;
}

export interface LocationData {
  latitude: number;
  longitude: number;
  address_text?: string;
}

export interface StatusHistoryItem {
  id: number;
  old_status?: string;
  new_status: string;
  changed_by_user_id?: number;
  notes?: string;
  created_at: string;
}

export interface PredictionResult {
  incident_id?: number;
  disaster_type: string;
  priority_score: number;
  priority_level: string; // 'Very Low' | 'Low' | 'Medium' | 'High' | 'Critical'
  confidence: number;
  language_detected: string;
  class_probabilities: Record<string, number>;
  created_at: string;
}

export interface Incident {
  id: number;
  title: string;
  description?: string;
  disaster_type: string;
  priority_score: number;
  priority_level: string;
  confidence: number;
  status: 'Pending' | 'Assigned' | 'En Route' | 'Rescue In Progress' | 'Completed';
  image_url?: string;
  language_detected: string;
  created_by_id?: number;
  assigned_responder_id?: number;
  assigned_team_id?: number;
  created_at: string;
  updated_at: string;
  location?: LocationData;
}

export interface IncidentDetail extends Incident {
  class_probabilities: Record<string, number>;
  recommendations: string[];
  status_history: StatusHistoryItem[];
}

export interface DashboardStats {
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

export interface RecentIncident {
  id: number;
  title: string;
  disaster_type: string;
  priority_score: number;
  priority_level: string;
  status: string;
  created_at: string;
  location_summary?: string;
}

export interface MapLocation {
  id: number;
  title: string;
  latitude: number;
  longitude: number;
  priority_score: number;
  priority_level: string;
  status: string;
  disaster_type: string;
}

export interface AnalyticsResponse {
  disaster_distribution: Record<string, number>;
  priority_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  language_distribution: Record<string, number>;
  daily_incidents: Record<string, number>;
  average_confidence: number;
}
