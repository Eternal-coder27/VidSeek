export interface VideoMetadata {
  video_id: string;
  filename: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  duration_sec: number;
  fps: number;
  frame_count: number;
  error_message?: string | null;
  created_at: string;
}

export interface SearchMatch {
  rank: number;
  start_time: number;
  end_time: number;
  peak_timestamp: number;
  similarity_score: number;
  thumbnail_url: string;
}

export interface SearchResponse {
  video_id: string;
  query: string;
  disclaimer: string;
  results: SearchMatch[];
}