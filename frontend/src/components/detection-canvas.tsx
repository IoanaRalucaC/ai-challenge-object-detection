"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { Scan, RotateCcw, Loader2, AlertCircle, X } from "lucide-react";
import { cn, getLabelColor } from "@/lib/utils";
import { useDetectionStore } from "@/lib/store";
import { detectObjects } from "@/lib/api";

export function DetectionCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Fine-grained Zustand selectors to avoid unnecessary re-renders
  const imageUrl = useDetectionStore((s) => s.imageUrl);
  const imageFile = useDetectionStore((s) => s.imageFile);
  const detections = useDetectionStore((s) => s.detections);
  const isProcessed = useDetectionStore((s) => s.isProcessed);
  const setDetections = useDetectionStore((s) => s.setDetections);
  const reset = useDetectionStore((s) => s.reset);

  const {
    mutate,
    isPending,
    error,
    reset: resetMutation,
  } = useMutation({
    mutationFn: detectObjects,
    onSuccess: (data) => setDetections(data.detections),
  });

  // ── Draw image + bounding boxes ────────────────────────────────────────────
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || canvas.width === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Full-res draw; CSS scales the canvas element to fit the viewport
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const color = getLabelColor(det.label).hex;
      const bw = x2 - x1;
      const bh = y2 - y1;

      // Bounding box
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 600));
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.restore();

      // Label pill
      const label = `${det.label}  ${(det.confidence * 100).toFixed(0)}%`;
      const fontSize = Math.max(13, Math.round(canvas.width / 80));
      ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
      const textW = ctx.measureText(label).width;
      const padX = 8;
      const padY = 5;
      const pillH = fontSize + padY * 2;
      const pillY = y1 >= pillH + 4 ? y1 - pillH - 4 : y1 + 4;

      // Pill background
      ctx.save();
      ctx.fillStyle = color + "dd"; // ~87% opacity
      ctx.beginPath();
      ctx.roundRect(x1, pillY, textW + padX * 2, pillH, 5);
      ctx.fill();
      ctx.restore();

      // Pill text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + padX, pillY + padY + fontSize - 2);
    }
  }, [detections]);

  // ── Load image into canvas on URL change ──────────────────────────────────
  useEffect(() => {
    if (!imageUrl) {
      setImageLoaded(false);
      imageRef.current = null;
      return;
    }

    setImageLoaded(false);
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      setImageLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // ── Redraw whenever image is ready or detections arrive ──────────────────
  useEffect(() => {
    if (imageLoaded) drawCanvas();
  }, [imageLoaded, drawCanvas]);

  const handleClear = () => {
    resetMutation();
    reset();
  };

  // Yield to UploadZone when no image is selected
  if (!imageUrl) return null;

  return (
    <div className="flex-1 flex flex-col items-center gap-4 min-h-0">
      {/* Canvas container */}
      <div className="relative flex-1 flex items-center justify-center w-full min-h-0 overflow-hidden rounded-xl">
        {/* Canvas — drawn at native resolution, CSS scales it to fit */}
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full rounded-xl shadow-2xl shadow-black/60 object-contain"
          style={{ display: imageLoaded ? "block" : "none" }}
        />

        {/* Image loading spinner */}
        {!imageLoaded && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-zinc-600 animate-spin" />
          </div>
        )}

        {/* Detection in-flight overlay */}
        {isPending && (
          <div className="absolute inset-0 flex flex-col items-center justify-center rounded-xl bg-black/65 backdrop-blur-sm gap-4">
            <div className="relative flex items-center justify-center">
              <div className="absolute w-16 h-16 rounded-full border-2 border-indigo-500/30 animate-ping" />
              <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
            </div>
            <p className="text-sm font-medium text-zinc-300 tracking-wide">
              Running YOLO + SAHI detection…
            </p>
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-start gap-2.5 px-4 py-3 w-full rounded-lg bg-red-500/10 border border-red-500/25 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span className="flex-1 min-w-0">{error.message}</span>
        </div>
      )}

      {/* Action bar */}
      <div className="flex items-center gap-3 w-full">
        {/* Primary action */}
        {!isProcessed ? (
          <button
            onClick={() => imageFile && mutate(imageFile)}
            disabled={isPending || !imageFile}
            className={cn(
              "flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all duration-150",
              "bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white",
              "shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40",
              "disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none",
            )}
          >
            {isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Scan className="w-4 h-4" />
            )}
            {isPending ? "Detecting…" : "Run Detection"}
          </button>
        ) : (
          <button
            onClick={() => imageFile && mutate(imageFile)}
            disabled={isPending}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-all disabled:opacity-50"
          >
            <RotateCcw className="w-4 h-4" />
            Re-run
          </button>
        )}

        {/* Secondary: clear */}
        <button
          onClick={handleClear}
          disabled={isPending}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-all disabled:opacity-50"
        >
          <X className="w-4 h-4" />
          Clear
        </button>

        {/* Count badge */}
        {isProcessed && !isPending && (
          <span className="ml-auto text-sm text-zinc-500">
            {detections.length} {detections.length === 1 ? "object" : "objects"}{" "}
            detected
          </span>
        )}
      </div>
    </div>
  );
}
