import { ScanEye, Cpu, ExternalLink } from "lucide-react";
import { UploadZone } from "@/components/upload-zone";
import { DetectionCanvas } from "@/components/detection-canvas";
import { ResultsSidebar } from "@/components/results-sidebar";

export default function HomePage() {
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-zinc-950">
      {/* ── Top navigation bar ── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm z-10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-600/20">
            <ScanEye className="w-4 h-4 text-indigo-400" />
          </div>
          <span className="font-semibold text-zinc-100 tracking-tight">
            ObjectDetect AI
          </span>
          <span className="flex items-center gap-1 text-xs bg-zinc-800 text-zinc-400 border border-zinc-700 px-2 py-0.5 rounded-full">
            <Cpu className="w-3 h-3" />
            YOLOv11 + SAHI
          </span>
        </div>

        <a
          href={`${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          API Docs
        </a>
      </header>

      {/* ── Main layout ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Canvas / upload area */}
        <main className="flex-1 flex flex-col gap-0 p-5 overflow-hidden min-w-0">
          <UploadZone />
          <DetectionCanvas />
        </main>

        {/* Results sidebar */}
        <ResultsSidebar />
      </div>
    </div>
  );
}
