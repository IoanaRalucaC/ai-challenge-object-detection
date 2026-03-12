import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Per-label visual identity: canvas hex + badge Tailwind classes
export const LABEL_COLORS: Record<string, { hex: string; badge: string }> = {
  Person: {
    hex: "#22c55e",
    badge: "bg-green-500/15 text-green-400 border-green-500/30",
  },
  Dog: {
    hex: "#3b82f6",
    badge: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  },
  Cat: {
    hex: "#f97316",
    badge: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  },
};

export function getLabelColor(label: string) {
  return (
    LABEL_COLORS[label] ?? {
      hex: "#a855f7",
      badge: "bg-purple-500/15 text-purple-400 border-purple-500/30",
    }
  );
}
