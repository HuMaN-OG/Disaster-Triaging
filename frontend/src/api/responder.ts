import { apiClient } from './client';
import { Incident, IncidentDetail } from '../types';

export const getPendingResponderIncidentsApi = async (): Promise<Incident[]> => {
  const response = await apiClient.get<Incident[]>('/responder/incidents');
  return response.data;
};

export const acceptIncidentApi = async (id: number, notes?: string): Promise<IncidentDetail> => {
  const response = await apiClient.post<IncidentDetail>(`/responder/${id}/accept`, { notes });
  return response.data;
};

export const updateResponderStatusApi = async (
  id: number,
  status: string,
  notes?: string
): Promise<IncidentDetail> => {
  const response = await apiClient.patch<IncidentDetail>(`/responder/${id}/status`, { status, notes });
  return response.data;
};
