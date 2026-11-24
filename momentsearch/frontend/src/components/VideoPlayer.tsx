import React, { useEffect, useRef } from "react";
import { getVideoStreamUrl } from "../api/client";
import type { VideoMetadata } from "../types";

export interface VideoPlayerProps {
  video: VideoMetadata;
  seekToSeconds?: number | null;
}

export function VideoPlayer({ video, seekToSeconds }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (seekToSeconds == null || Number.isNaN(seekToSeconds)) return;
    const el = videoRef.current;
    if (!el) return;

    const applySeek = () => {
      el.currentTime = Math.max(0, seekToSeconds);
      el.play().catch(() => {});
    };

    if (el.readyState >= 1) {
      applySeek();
    } else {
      el.addEventListener("loadedmetadata", applySeek, { once: true });
      return () => el.removeEventListener("loadedmetadata", applySeek);
    }
  }, [seekToSeconds, video.video_id]);

  return (
    <div className="p-4 border rounded-lg bg-white space-y-2">
      <div className="flex items-center justify-between text-xs text-gray-600">
        <span className="font-semibold text-sm text-gray-900">{video.filename}</span>
        <span>
          {video.duration_sec}s • {video.fps} fps • {video.pipeline?.toUpperCase() || "V1"}
        </span>
      </div>

      <video
        ref={videoRef}
        key={video.video_id}
        src={getVideoStreamUrl(video.video_id)}
        controls
        className="w-full rounded bg-black max-h-[420px]"
      />
    </div>
  );
}

export default VideoPlayer;