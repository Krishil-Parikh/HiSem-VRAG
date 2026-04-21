/**
 * API Configuration
 * Centralized configuration for backend API communication
 */

export const API_BASE = 'http://localhost:8000';

export const API_ENDPOINTS = {
  // Auth
  login: `${API_BASE}/api/auth/login`,
  signup: `${API_BASE}/api/auth/signup`,
  me: `${API_BASE}/api/auth/me`,
  
  // Video
  videoUpload: `${API_BASE}/api/video/upload`,
  videoList: `${API_BASE}/api/video/list`,
  videoStatus: (id: string) => `${API_BASE}/api/video/status/${id}`,
  videoMemory: (id: string) => `${API_BASE}/api/video/memory/${id}`,
  videoTopics: (id: string) => `${API_BASE}/api/video/topics/${id}`,
  videoFrame: (id: string, topicId: number, filename: string) => `${API_BASE}/api/video/frame/${id}/${topicId}/${filename}`,
  videoDelete: (id: string) => `${API_BASE}/api/video/${id}`,
  query: `${API_BASE}/api/video/query`,
  
  // Health
  health: `${API_BASE}/health`,
};

/**
 * Helper to make authenticated API requests
 */
export async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;
  
  const headers = {
    ...options.headers,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  
  return fetch(url, { ...options, headers });
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  if (typeof window === 'undefined') return false;
  return !!localStorage.getItem('authToken');
}

/**
 * Get current user from localStorage
 */
export function getCurrentUser(): { name: string; email: string } | null {
  if (typeof window === 'undefined') return null;
  const userStr = localStorage.getItem('user');
  if (!userStr) return null;
  try {
    return JSON.parse(userStr);
  } catch {
    return null;
  }
}

/**
 * Clear auth data and redirect to login
 */
export function logout() {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('authToken');
  localStorage.removeItem('user');
  window.location.href = '/';
}
