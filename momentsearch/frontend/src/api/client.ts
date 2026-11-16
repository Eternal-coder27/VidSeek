import type { SearchResponse, VideoMetadata } from '../types';

export const API_BASE = 'http://localhost:8000';

export async function fetchVideos(): Promise<VideoMetadata[]> {
  const res = await fetch(`${API_BASE}/api/videos`);
  if (!res.ok) throw new Error('Failed to load video library.');
  return res.json();
}

export async function uploadVideo(file: File): Promise<VideoMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/videos/upload`, {
    method: 'POST',
    body: formData,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Video upload failed.');
  return data;
}

export async function reprocessVideo(videoId: string): Promise<VideoMetadata> {
  const res = await fetch(`${API_BASE}/api/videos/${videoId}/process`, {
    method: 'POST',
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Failed to restart processing.');
  return data;
}

export async function searchVideo(
  videoId: string,
  query: string,
  topK: number
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/api/videos/${videoId}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      top_k: topK,
      merge_gap_sec: 2.0,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Search failed.');
  return data;
}

export function getVideoStreamUrl(videoId: string): string {
  return `${API_BASE}/api/videos/${videoId}/stream`;
}

export function getThumbnailUrl(relativeUrl: string): string {
  return `${API_BASE}${relativeUrl}`;
}