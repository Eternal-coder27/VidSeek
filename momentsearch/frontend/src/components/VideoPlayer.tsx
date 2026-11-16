import { useEffect, useRef, useState } from 'react';
import { Clock, Film } from 'lucide-react';
import { getVideoStreamUrl } from '../api/client';
import type { SearchMatch, VideoMetadata } from '../types';

export function formatTimestamp(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
}

interface VideoPlayerProps {
  video: VideoMetadata | null;
  seekRequest: { time: number; nonce: number } | null;
  matches: SearchMatch[];
  activeRank: number | null;
  onSelectMatch: (match: SearchMatch) => void;
}

export default function VideoPlayer({
  video,
  seekRequest,
  matches,
  activeRank,
  onSelectMatch,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    setCurrentTime(0);
  }, [video?.video_id]);

  useEffect(() => {
    if (videoRef.current && seekRequest !== null) {
      videoRef.current.currentTime = seekRequest.time;
      videoRef.current.play().catch(() => {});
    }
  }, [seekRequest]);

  if (!video) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-2 py-20">
        <Film className="w-10 h-10 text-slate-700" />
        <p className="text-sm">Select or upload a video from the sidebar to begin</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-white truncate">{video.filename}</h2>
          <p className="text-xs text-slate-400">
            ID: <span className="font-mono">{video.video_id.slice(0, 8)}</span> • {video.fps} FPS
          </p>
        </div>
        <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 font-mono text-xs text-indigo-300 flex items-center gap-1.5 shrink-0">
          <Clock className="w-3.5 h-3.5" />
          {formatTimestamp(currentTime)} / {formatTimestamp(video.duration_sec)}
        </div>
      </div>

      <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
        <video
          ref={videoRef}
          key={video.video_id}
          controls
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          className="w-full h-full object-contain"
          src={getVideoStreamUrl(video.video_id)}
        />
      </div>

      {video.duration_sec > 0 && matches.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
          <div className="flex justify-between text-[11px] text-slate-400 mb-1.5">
            <span>Matched Event Intervals</span>
            <span>Click a marker to seek</span>
          </div>
          <div className="relative h-3.5 bg-slate-800 rounded-md overflow-hidden">
            {matches.map((match) => {
              const leftPct = (match.start_time / video.duration_sec) * 100;
              const widthPct = Math.max(
                1.5,
                ((match.end_time - match.start_time) / video.duration_sec) * 100
              );
              return (
                <button
                  key={match.rank}
                  type="button"
                  onClick={() => onSelectMatch(match)}
                  title={`#${match.rank}: ${formatTimestamp(match.start_time)} – ${formatTimestamp(
                    match.end_time
                  )} (Sim: ${match.similarity_score})`}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  className={`absolute top-0 bottom-0 transition cursor-pointer ${
                    activeRank === match.rank
                      ? 'bg-emerald-400'
                      : 'bg-indigo-500/85 hover:bg-indigo-400'
                  }`}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}