<h1> Banner </h1>
<h4 align="center"> A mobile rock climbing hold detection app for colour blind rock climbers, powered by AI</h4>

---

<h1 align="center">AscendAI Backend</h1>
<h4 align="center"> <a href=https://github.com/KurtSchwimmbacher/AscendAI.git> Frontend Repo </a> • Backend Repo</h4>

## 🚀 Deployment

### Memory Requirements

This backend uses YOLOv8 for computer vision, which requires careful memory management:

- **Free Tier (512MB RAM)**: Use `yolo_imgsz: 640` ✅ (default)
- **Starter Tier (1GB RAM)**: Can increase to `yolo_imgsz: 1280`
- **Standard+ Tier (2GB+ RAM)**: Can use `yolo_imgsz: 2560` for maximum accuracy

### Environment Variables

Set these in your deployment platform (Render, Heroku, etc.):

- `PORT`: Server port (auto-set by Render)
- `HF_HOME`: Cache directory for Hugging Face models (auto-set to `/tmp/.cache/huggingface`)
- `YOLO_IMGSZ`: Optional - Override inference image size (default: 640)

### First Request Behavior

⏱️ The first API request after deployment will take **30-60 seconds** as it downloads the YOLO model (~6MB) from Hugging Face. Subsequent requests are fast (1-3 seconds).

<details>
<summary>📑 <strong>Table of Contents</strong> (Click to expand)</summary>
</details>

---

## Contributing & Licenses

> This project was developed as part of a university course requirement and is currently private and non-commercial.  
> No external contributions are being accepted at this time.

## Authors & Contact Info

Built with ❤️ by:

- **Kurt Schwimmbacher**

## Acknowledgements

Special thanks to:

- **HuggingFace user John LaRocque(jwlarocque)** for fine tuning the YoloV8 Climbing Hold detections model (https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2)
- **FastAPI Team** for backend scalability support
- **Open Window Lecturer Armand Pretorius** for providing feedback and insight
