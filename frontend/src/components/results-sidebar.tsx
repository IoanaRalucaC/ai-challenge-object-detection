"use client";

import { BarChart3, Tag, ChevronRight, Inbox, TrendingUp } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn, getLabelColor } from "@/lib/utils";
import { useDetectionStore } from "@/lib/store";
import type { Detection } from "@/types/detection";

// ── Single detection card ──────────────────────────────────────────────────
function DetectionCard({
  detection,
  index,
}: {
  detection: Detection;
  index: number;
}) {
  const colorInfo = getLabelColor(detection.label);
  const pct = Math.round(detection.confidence * 100);

  return (
    <div className="group flex flex-col gap-2.5 p-3.5 rounded-lg bg-zinc-800/50 border border-zinc-700/40 hover:border-zinc-600/60 hover:bg-zinc-800/80 transition-all duration-150">
      {/* Top row: index, badge, confidence */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-zinc-600 w-5 shrink-0 tabular-nums">
          {String(index + 1).padStart(2, "0")}
        </span>

        <Badge
          className={cn("flex-shrink-0 gap-1 border", colorInfo.badge)}
          variant={null as never}
        >
          <Tag className="w-3 h-3" />
          {detection.label}
        </Badge>

        <span
          className={cn(
            "ml-auto text-sm font-bold tabular-nums shrink-0",
            pct >= 80
              ? "text-green-400"
              : pct >= 60
                ? "text-yellow-400"
                : "text-red-400",
          )}
        >
          {pct}%
        </span>
      </div>

      {/* Confidence progress bar */}
      <div className="h-1.5 w-full rounded-full bg-zinc-700 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, backgroundColor: colorInfo.hex }}
        />
      </div>

      {/* Bbox coordinates */}
      <p className="text-[11px] text-zinc-600 font-mono leading-none">
        bbox [{detection.bbox.join(", ")}]
      </p>
    </div>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────
export function ResultsSidebar() {
  const detections = useDetectionStore((s) => s.detections);
  const isProcessed = useDetectionStore((s) => s.isProcessed);

  // Aggregate counts per label
  const grouped = detections.reduce<Record<string, number>>((acc, d) => {
    acc[d.label] = (acc[d.label] ?? 0) + 1;
    return acc;
  }, {});

  // Average confidence overall
  const avgConf =
    detections.length > 0
      ? Math.round(
          (detections.reduce((sum, d) => sum + d.confidence, 0) /
            detections.length) *
            100,
        )
      : null;

  return (
    <aside className="flex flex-col h-full w-80 shrink-0 bg-zinc-900/80 border-l border-zinc-800">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-4 shrink-0">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-600/20">
          <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
        </div>
        <h2 className="text-sm font-semibold text-zinc-200">Results</h2>
        {detections.length > 0 && (
          <span className="ml-auto text-xs bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded-full font-medium tabular-nums">
            {detections.length}
          </span>
        )}
      </div>

      <Separator />

      {/* Stats row (visible after detection) */}
      {isProcessed && detections.length > 0 && (
        <>
          <div className="flex items-center gap-3 px-4 py-3 shrink-0">
            <div className="flex-1 flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-widest text-zinc-600 font-medium">
                Detected
              </span>
              <span className="text-lg font-bold text-zinc-100 tabular-nums leading-none">
                {detections.length}
              </span>
            </div>
            <div className="w-px h-8 bg-zinc-800" />
            <div className="flex-1 flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-widest text-zinc-600 font-medium">
                Avg Conf
              </span>
              <div className="flex items-baseline gap-1">
                <TrendingUp className="w-3 h-3 text-green-400 mb-0.5" />
                <span className="text-lg font-bold text-green-400 tabular-nums leading-none">
                  {avgConf}%
                </span>
              </div>
            </div>
          </div>

          {/* Category pills */}
          <div className="flex flex-wrap gap-1.5 px-4 pb-3 shrink-0">
            {Object.entries(grouped).map(([label, count]) => {
              const colorInfo = getLabelColor(label);
              return (
                <span
                  key={label}
                  className={cn(
                    "flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border",
                    colorInfo.badge,
                  )}
                >
                  {label}
                  <ChevronRight className="w-3 h-3 opacity-40" />
                  {count}
                </span>
              );
            })}
          </div>

          <Separator />
        </>
      )}

      {/* Detection list */}
      <ScrollArea className="flex-1">
        <div className="px-4 py-3">
          {!isProcessed ? (
            <EmptyState message="Upload an image and run detection to see results here." />
          ) : detections.length === 0 ? (
            <EmptyState message="No objects were detected in this image." />
          ) : (
            <div className="flex flex-col gap-2">
              {detections.map((det, i) => (
                <DetectionCard
                  key={`${det.label}-${i}-${det.bbox.join("")}`}
                  detection={det}
                  index={i}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="flex items-center justify-center w-12 h-12 rounded-2xl bg-zinc-800">
        <Inbox className="w-6 h-6 text-zinc-600" />
      </div>
      <p className="text-sm text-zinc-500 max-w-[200px] leading-relaxed">
        {message}
      </p>
    </div>
  );
}
