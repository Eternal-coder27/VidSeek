import { Play, Search, ShieldAlert } from 'lucide-react';
import { getThumbnailUrl } from '../api/client';
import type { SearchMatch } from '../types';
import { formatTimestamp } from './VideoPlayer';

interface SearchPanelProps {
  matches: SearchMatch[];
  disclaimer: string | null;
  activeRank: number | null;
  onSelectMatch: (match: SearchMatch) => void;
}

export default function SearchPanel({
  matches,
  disclaimer,
  activeRank,
  onSelectMatch,
}: SearchPanelProps) {
  return (
    <div className="flex flex-col gap-3 h-full">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        Retrieved Moments ({matches.length})
      </h2>

      {disclaimer && (
        <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/50 flex items-start gap-2.5 text-xs text-amber-200/90">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <p>{disclaimer}</p>
        </div>
      )}

      {matches.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-500 px-6 py-12">
          <Search className="w-8 h-8 text-slate-700 mb-2" />
          <p className="text-sm text-slate-400">No retrieved moments yet</p>
          <p className="text-xs text-slate-500 mt-1">
            Enter a natural-language event description below the video player to search.
          </p>
        </div>
      ) : (
        <div className="space-y-3 overflow-y-auto pr-1">
          {matches.map((match) => {
            const isActive = activeRank === match.rank;
            return (
              <div
                key={match.rank}
                onClick={() => onSelectMatch(match)}
                className={`group rounded-xl border p-3 transition cursor-pointer flex gap-3 ${
                  isActive
                    ? 'bg-indigo-950/40 border-indigo-500'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="relative w-32 h-20 rounded-lg overflow-hidden bg-black shrink-0 border border-slate-800">
                  <img
                    src={getThumbnailUrl(match.thumbnail_url)}
                    alt={`Match #${match.rank}`}
                    className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
                  />
                  <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white">
                    #{match.rank}
                  </span>
                </div>

                <div className="flex-1 min-w-0 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs font-semibold text-indigo-300">
                        {formatTimestamp(match.start_time)} – {formatTimestamp(match.end_time)}
                      </span>
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        Sim: {match.similarity_score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-1">
                      Peak frame at {formatTimestamp(match.peak_timestamp)}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectMatch(match);
                    }}
                    className="mt-2 w-full py-1.5 px-3 rounded-lg bg-slate-800 group-hover:bg-indigo-600 text-xs font-medium text-slate-200 group-hover:text-white flex items-center justify-center gap-1.5 transition"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" /> Play from{' '}
                    {formatTimestamp(match.start_time)}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}