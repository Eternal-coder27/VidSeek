import React, { useRef, useState } from 'react';
import { Loader2, Upload } from 'lucide-react';
import { uploadVideo } from '../api/client';
import type { VideoMetadata } from '../types';

interface VideoUploadProps {
  onUploaded: (video: VideoMetadata) => void;
}

export default function VideoUpload({ onUploaded }: VideoUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const created = await uploadVideo(file);
      onUploaded(created);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.mov,.avi,.mkv"
        className="hidden"
        onChange={handleFileChange}
      />
      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        className="w-full py-3.5 px-4 rounded-xl border border-dashed border-slate-700 hover:border-indigo-500 bg-slate-900/90 hover:bg-slate-900 transition flex flex-col items-center justify-center gap-1.5 text-sm font-medium text-slate-200 disabled:opacity-50 cursor-pointer"
      >
        {uploading ? (
          <>
            <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
            <span>Uploading footage...</span>
          </>
        ) : (
          <>
            <Upload className="w-5 h-5 text-indigo-400" />
            <span>Upload CCTV Footage</span>
            <span className="text-[11px] text-slate-500">
              MP4, MOV, AVI, MKV (max 500 MB)
            </span>
          </>
        )}
      </button>

      {error && (
        <p className="mt-2 text-xs text-rose-300 bg-rose-950/60 border border-rose-900 rounded-lg p-2">
          {error}
        </p>
      )}
    </div>
  );
}