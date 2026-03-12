import { create } from "zustand";
import type { Detection } from "@/types/detection";

interface DetectionStore {
  imageFile: File | null;
  imageUrl: string | null;
  detections: Detection[];
  isProcessed: boolean;
  setImage: (file: File, url: string) => void;
  setDetections: (detections: Detection[]) => void;
  reset: () => void;
}

export const useDetectionStore = create<DetectionStore>((set) => ({
  imageFile: null,
  imageUrl: null,
  detections: [],
  isProcessed: false,

  setImage: (file, url) =>
    set((state) => {
      // Revoke the previous object URL to prevent memory leaks
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      return {
        imageFile: file,
        imageUrl: url,
        detections: [],
        isProcessed: false,
      };
    }),

  setDetections: (detections) => set({ detections, isProcessed: true }),

  reset: () =>
    set((state) => {
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      return {
        imageFile: null,
        imageUrl: null,
        detections: [],
        isProcessed: false,
      };
    }),
}));
