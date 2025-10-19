import yolov5
import sys
import pathlib
from pathlib import Path
import torch
import cv2
import numpy as np
import gradio as gr
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import glob
import torch.serialization
from models.yolo import DetectionModel
from models.common import Conv, C3, BottleneckCSP, SPPF
from utils.general import yaml_load
from utils.activations import Hardswish, SiLU
import torch.nn as nn
import collections

from yolov5.utils import downloads

# Patch attempt_download to just return the local path
downloads.attempt_download = lambda x, *a, **kw: x
downloads.attempt_download_from_hub = lambda x, *a, **kw: x

# Windows patch for local testing
if sys.platform.startswith("win"):
    pathlib.PosixPath = pathlib.WindowsPath

torch.serialization.add_safe_globals([
    DetectionModel,
    Conv,
    C3,
    BottleneckCSP,
    SPPF,
    Hardswish,
    SiLU,
    yaml_load,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.upsampling.Upsample,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.LeakyReLU,
    torch.nn.modules.conv.Conv2d,
    collections.OrderedDict,
])

# Device
device = select_device("cpu")

# Load YOLOv5
model = attempt_load("weights/best.pt", device=device)
model.eval()

# Global variable to store detection results
detection_history = []

def detect_image(image: np.ndarray):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    orig_h, orig_w = image.shape[:2]
    img_resized = letterbox(image, new_shape=640)[0]
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

    boat_count = 0
    if pred is not None and len(pred):
        boat_count = len(pred)
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], (orig_h, orig_w)).round()
        for *xyxy, conf, cls in pred:
            xyxy = [int(x) for x in xyxy]
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(image, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, boat_count

def detect_folder_images(folder_path="static"):
    global detection_history
    images_paths = sorted(glob.glob(f"{folder_path}/*.*"))
    
    print(f"Found {len(images_paths)} images in {folder_path}")
    
    if not images_paths:
        yield None, [], "Total number of boats detected: 0"
        return
    
    gallery_results = []
    processed_img = None
    total_boats = 0

    for i, img_path in enumerate(images_paths):
        print(f"Processing {i+1}/{len(images_paths)}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img, boat_count = detect_image(img_rgb)
        total_boats += boat_count
        gallery_results.append(processed_img)
        
        # Store detection info
        detection_history.append({
            'image_path': img_path,
            'boat_count': boat_count
        })
        
        yield processed_img, None, f"Processing... ({i+1}/{len(images_paths)})"

    # Final yield with complete gallery and total count
    final_message = f"✅ Total number of boats detected: {total_boats}"
    yield processed_img if gallery_results else None, gallery_results, final_message
def get_analytics():
    """Generate analytics from detection history"""
    if not detection_history:
        return "No detection data available yet. Run batch detection first.", ""
    
    total_images = len(detection_history)
    total_boats = sum(d['boat_count'] for d in detection_history)
    avg_boats = total_boats / total_images if total_images > 0 else 0
    images_with_boats = sum(1 for d in detection_history if d['boat_count'] > 0)
    
    # Create summary text
    summary = f"""
    ## Detection Summary
    
    - **Total Images Processed:** {total_images}
    - **Total Boats Detected:** {total_boats}
    - **Average Boats per Image:** {avg_boats:.2f}
    - **Images with Boats:** {images_with_boats} ({images_with_boats/total_images*100:.1f}%)
    - **Images without Boats:** {total_images - images_with_boats}
    """
    
    # Create detailed breakdown
    breakdown = "### Detailed Breakdown\n\n"
    for i, detection in enumerate(detection_history, 1):
        img_name = Path(detection['image_path']).name
        breakdown += f"{i}. **{img_name}**: {detection['boat_count']} boat(s)\n"
    
    return summary, breakdown

with gr.Blocks(title="YOLOv5 Boat Detector") as demo:
    with gr.Row():
        gr.Markdown("## YOLOv5 Boat Detector")
        analytics_btn = gr.Button("📊 View Analytics", size="sm", variant="primary", scale=0, min_width=180)
    
    with gr.Tab("Detection") as detection_tab:
        with gr.Row():
            inp_image = gr.Image(type="numpy", label="Upload Image")
            out_image = gr.Image(type="numpy", label="Detection Result")
        
        batch_btn = gr.Button("Start Detection", variant="primary")
        status_text = gr.Markdown("Total number of boats detected: 0")
        out_gallery = gr.Gallery(label="Detection Results", show_label=True, columns=4, height="auto")

        inp_image.change(
            lambda img: (detect_image(img)[0], ""), 
            inputs=inp_image, 
            outputs=[out_image, status_text]
        )
        batch_btn.click(
            fn=detect_folder_images, 
            inputs=None, 
            outputs=[out_image, out_gallery, status_text]
        )
    
    with gr.Tabs() as main_tabs:

        with gr.Tab("Graphs", visible=False, id="graphs") as graphs_tab:
            gr.Markdown("# 📈 Detection Analytics - Graphs")
            # gr.Markdown("##  Visualization Graphs")
            with gr.Row():
                graph1 = gr.Image(value="analytics\Hourly_Activity_Patterns.webp", label="Graph 1", show_label=True)
                graph4 = gr.Image(value="analytics\weekly_boat_activity.webp", label="Graph 4", show_label=True)

            with gr.Row():
                graph3 = gr.Image(value="analytics\Weekly_Boat_Activity_by_Hour.webp", label="Graph 3", show_label=True)
            with gr.Row():
                graph2 = gr.Image(value="analytics\monthly_boaat_activity.webp", label="Graph 2", show_label=True)
            
            
            
            back_btn_graphs = gr.Button("← Back to Detection")

        with gr.Tab("Results", visible=False, id="results") as results_tab:
            gr.Markdown("# 📊 Detection Analytics - Results")
            refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
            
            with gr.Row():
                analytics_summary = gr.Markdown()
            
            with gr.Row():
                analytics_breakdown = gr.Markdown()
            
            back_btn_results = gr.Button("← Back to Detection")
            
            refresh_analytics_btn.click(
                fn=get_analytics,
                inputs=None,
                outputs=[analytics_summary, analytics_breakdown]
            )

    # Navigation logic
    def show_analytics():
        summary, breakdown = get_analytics()
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(selected="graphs"),  # Select Results tab
            summary,
            breakdown
        )

    def show_detection():
        return (
            gr.update(visible=True),   # Show detection tab
            gr.update(visible=False),  # Hide results tab
            gr.update(visible=False)   # Hide graphs tab
        )

    analytics_btn.click(
        fn=show_analytics,
        inputs=None,
        outputs=[detection_tab, results_tab, graphs_tab, main_tabs, analytics_summary, analytics_breakdown]
    )

    back_btn_results.click(
        fn=show_detection,
        inputs=None,
        outputs=[detection_tab, results_tab, graphs_tab]
    )

    back_btn_graphs.click(
        fn=show_detection,
        inputs=None,
        outputs=[detection_tab, results_tab, graphs_tab]
    )

demo.launch()