# 🚗 CARLA Vision Navigation

Natural language navigation system for autonomous vehicles using Qwen2.5-VL vision-language model.

---

## 🚀 Quick Start

### 1️⃣ Setup Virtual Environment

**Mac/Linux:**
```bash
cd your-project-folder
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
cd your-project-folder
python -m venv venv
venv\Scripts\activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- PyTorch (deep learning)
- Transformers (Hugging Face)
- Qwen2.5-VL utilities
- OpenCV (image processing)
- Other required packages

⏱️ Takes ~5-10 minutes (downloads ~3GB)

---

### 3️⃣ Run Navigation

```bash
python vision_navigation.py your_image.jpg "Navigate to the traffic light"
```

**Example:**
```bash
python vision_navigation.py carla_screenshot.jpg "Navigate to the traffic light"
python vision_navigation.py scene.png "Go to the building"
python vision_navigation.py image.jpg "Navigate to the stop sign"
```

---

