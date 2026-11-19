import { apiClient } from './client';
import type { DashboardStats, RecentIncident, MapLocation, AnalyticsResponse } from '../types';

export const getDashboardStatsApi = async (): Promise<DashboardStats> => {
  const response = await apiClient.get<DashboardStats>('/dashboard/stats');
  return response.data;
};

export const getRecentIncidentsApi = async (): Promise<RecentIncident[]> => {
  const response = await apiClient.get<RecentIncident[]>('/dashboard/recent');
  return response.data;
};

export const getMapLocationsApi = async (): Promise<MapLocation[]> => {
  const response = await apiClient.get<MapLocation[]>('/dashboard/map');
  return response.data;
};

export const getAnalyticsApi = async (): Promise<AnalyticsResponse> => {
  const response = await apiClient.get<AnalyticsResponse>('/dashboard/analytics');
  return response.data;
};
