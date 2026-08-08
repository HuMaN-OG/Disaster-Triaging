import { apiClient } from './client';
import type { Incident, IncidentDetail } from '../types';

export const createIncidentApi = async (formData: FormData): Promise<IncidentDetail> => {
  const response = await apiClient.post<IncidentDetail>('/incidents', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getIncidentsApi = async (params?: {
  skip?: number;
  limit?: number;
  disaster_type?: string;
  priority_level?: string;
  status?: string;
  location?: string;
  application_context?: string;
  sort_by?: string;
  sort_order?: string;
}): Promise<Incident[]> => {
  const response = await apiClient.get<Incident[]>('/incidents', { params });
  return response.data;
};

export const getIncidentByIdApi = async (id: number): Promise<IncidentDetail> => {
  const response = await apiClient.get<IncidentDetail>(`/incidents/${id}`);
  return response.data;
};

export const getIncidentBriefApi = async (id: number): Promise<any> => {
  const response = await apiClient.get<any>(`/incidents/${id}/brief`);
  return response.data;
};

export const updateIncidentApi = async (
  id: number,
  data: { status?: string; assigned_responder_id?: number; assigned_team_id?: number; notes?: string }
): Promise<IncidentDetail> => {
  const response = await apiClient.patch<IncidentDetail>(`/incidents/${id}`, data);
  return response.data;
};

export const deleteIncidentApi = async (id: number): Promise<void> => {
  await apiClient.delete(`/incidents/${id}`);
};

export const uploadAttachmentsApi = async (id: number, files: File[]): Promise<any[]> => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  const response = await apiClient.post<any[]>(`/incidents/${id}/attachments`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
