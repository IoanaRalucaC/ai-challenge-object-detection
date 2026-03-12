"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDetectionStore } from "@/lib/store";

export function UploadZone() {
  const imageUrl = useDetectionStore((s) => s.imageUrl);
  const setImage = useDetectionStore((s) => s.setImage);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      setImage(file, url);
    },
    [setImage],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/webp": [".webp"],
      "image/gif": [".gif"],
    },
    maxFiles: 1,
    multiple: false,
  });

  // Hand off to DetectionCanvas once an image is loaded
  if (imageUrl) return null;

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex-1 flex flex-col items-center justify-center w-full",
        "rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200",
        "bg-zinc-900/40 hover:bg-zinc-800/40",
        isDragActive
          ? "border-indigo-500 bg-indigo-500/5 scale-[1.01]"
          : "border-zinc-700 hover:border-zinc-600",
      )}
    >
      <input {...getInputProps()} />

      <div className="flex flex-col items-center gap-5 px-8 py-16 text-center pointer-events-none select-none">
        {/* Icon */}
        <div
          className={cn(
            "flex items-center justify-center w-20 h-20 rounded-2xl transition-all duration-200",
            isDragActive
              ? "bg-indigo-500/20 text-indigo-400 scale-110"
              : "bg-zinc-800 text-zinc-500",
          )}
        >
          {isDragActive ? (
            <ImageIcon className="w-9 h-9" strokeWidth={1.5} />
          ) : (
            <Upload className="w-9 h-9" strokeWidth={1.5} />
          )}
        </div>

        {/* Copy */}
        <div>
          <p className="text-xl font-semibold text-zinc-200">
            {isDragActive ? "Release to upload" : "Drag & drop an image"}
          </p>
          <p className="mt-2 text-sm text-zinc-500">
            or{" "}
            <span className="text-indigo-400 hover:text-indigo-300 transition-colors">
              click to browse
            </span>{" "}
            your files
          </p>
        </div>

        <p className="text-xs text-zinc-600 bg-zinc-800/60 px-3 py-1.5 rounded-full">
          PNG · JPG · WEBP · GIF
        </p>
      </div>
    </div>
  );
}
