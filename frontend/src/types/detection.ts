export interface Detection {
  label: string;
  confidence: number;
  /** [x1, y1, x2, y2] in original image pixel coordinates */
  bbox: [number, number, number, number];
}

export interface DetectionResponse {
  count: number;
  detections: Detection[];
}
