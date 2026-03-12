# The Learning Journal: My 48-Hour Journey

## 1. The Starting Point: What was your prior experience with Computer Vision or Python before Friday?

My background in Computer Vision was mostly theoretical, based on a course I took during college. As for Python, I have not worked as a dedicated Python developer, but I have used it in various university laboratories.

I am comfortable navigating the environment - I know how to install dependencies, manage libraries, and follow technical logic - but I am not at the stage where I can write complex scripts entirely from memory. I rely on a collaborative workflow, using documentation, StackOverflow, and AI tools like Copilot to bridge the gap between my logical understanding and the specific Python syntax required.

## 2. The Research Path: List resources you used to learn.

I took a structured approach to catch up with the current state of the field:

I started by searching for "State of the art computer vision models in 2026" to ensure I was not using outdated methods. I performed a comparative analysis of different model versions (like YOLOv8 vs. the latest iterations) to understand the trade-offs between inference speed and mAP (mean Average Precision).

I refreshed my knowledge with Microsoft's documentation:

- Introduction to Computer Vision: https://learn.microsoft.com/en-us/training/modules/introduction-computer-vision/
- Introduction to Machine Learning: https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/

I checked the Ultralytics/YOLO documentation to understand how to move from a pre-trained model to a functional script:

- https://docs.ultralytics.com/tasks/detect/

I worked with Copilot to draft the application script, specifically focusing on how to configure parameters to optimize results without needing a massive GPU for retraining.

## 3. The "Pivot" Moments: Describe one specific technical hurdle you hit and exactly how you found the fix.

Since the app was relatively simple, I did not have many bugs, but I hit a major wall with model accuracy. I was getting too many false positives, and the model was missing smaller objects (false negatives).

Initially, I tried a fix by adjusting the global confidence threshold, but it did not solve the core issue. I searched for "how to improve YOLO accuracy without retraining" and "YOLO low recall on small objects". This led me to a few sophisticated techniques:

- SAHI (Slicing Aided Hyper Inference): I discovered that by slicing input images into smaller patches, the model could see details it previously ignored.
- TTA (Test Time Augmentation): This allowed the model to look at multiple versions of the same image (flipped, slightly scaled) to confirm its findings.
- Class-Specific Thresholds: I realized that different entities need different sensitivity levels. Instead of setting confidence in the call, it filters the results post-inference.
- NMS (Non-Maximum Suppression) Tuning: To filter out redundant detections like double detections (one dog getting two boxes) or if objects are overlapping (a person holding a cat) and one disappears.

Implementing SAHI and NMS (Non-Maximum Suppression) tuning turned the project from a basic script into a much more robust application.

## 4. Model Choice: Why did you choose the specific model you used?

I chose a discriminative/classification model (specifically from the YOLO family) because it is the industry standard for real-time identification. Our requirement was to identify and localize a specific entity, which is exactly what these models are optimized for.

I purposely avoided generative or diffusion models because they are computationally expensive and designed for creating content. For an efficient application, I needed a model optimized for low inference latency - getting a fast, accurate result using minimal resources.

## 5. If I Had More Time: If you had more than 48 hours, what would you do next?

- Custom Training and Data Augmentation: I would build a dedicated dataset and use a data pipeline to simulate different lighting and weather conditions, making the model more experienced.
- Hybrid Pipelines: I would experiment with a human-in-the-loop feedback system, where the app flags uncertain results for review, which could then be used to retrain and improve the model over time.
