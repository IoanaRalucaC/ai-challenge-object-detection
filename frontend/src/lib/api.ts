import type { DetectionResponse } from "@/types/detection";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function detectObjects(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`Detection failed (${response.status}): ${errorText}`);
  }

  return response.json() as Promise<DetectionResponse>;
}
