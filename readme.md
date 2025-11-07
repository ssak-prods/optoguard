# 🎥 AI-Powered Object Identifier with Voice Feedback

A real-time desktop tool that uses your webcam to detect objects using AI and speaks out what it sees — designed to make machines visually aware of their environment. Built to run offline, on modest hardware, and expandable via UI layers.

---

## 🎯 Project Objective

To build a lightweight, real-time AI assistant that uses your **laptop camera** to detect objects in its view and **speak** them aloud.  
This is not just a fun project — it's a **tool that can assist visually impaired users**, serve as a **context-aware assistant**, or even act as a **voice-enabled smart monitor** for warehouse / work environments.

---

## 🔧 Tech Stack Overview

| Component              | Tool / Library                |
|------------------------|-------------------------------|
| Language               | Python 3.9–3.11                |
| Camera Input           | OpenCV                         |
| Object Detection       | YOLOv5n (via PyTorch) OR cvlib + MobileNet SSD |
| Text-to-Speech         | pyttsx3 (offline)              |
| Optional GUI           | Tkinter                        |
| OS Support             | Windows 10+ / Linux            |

---

## 🚀 Phase-Wise Build Plan

---

### ✅ Phase 1: Core Functionality (Terminal-Based MVP)

#### 🎯 Goals
- Capture video from webcam
- Run real-time object detection on frames
- Announce detected object names via TTS
- Prevent repetitive announcements using cooldown logic
- Keep system lightweight and responsive

#### 📁 Modules to Implement
- `main.py` – Entry point
- `detector.py` – Handles object detection
- `speaker.py` – Handles voice output
- `utils.py` – Manages cooldowns and de-duplication
- `requirements.txt` – Install instructions

#### 🧪 Expected Behavior
1. App starts and opens webcam
2. Objects like "bottle", "phone", "laptop" are detected
3. Assistant says:
   > "I see a bottle"  
   > "There's a laptop ahead"
4. Same object won’t be repeated unless it disappears and reappears

#### 🛠 Notes
- Use 640x480 frame size for speed
- Run every 2nd frame if lag occurs
- GPU acceleration optional, CPU fallback required

---

## 🖥️ Runtime Performance Expectation

- Target FPS: 10–15 on 640x480 resolution
- Expected memory usage: <1GB
- GPU acceleration recommended, but not required

---

## 🧠 Dataset & Model Info

- Use pretrained **COCO** dataset (80-class support)
- Recommended models:
  - YOLOv5n (`ultralytics/yolov5`)
  - OR `cvlib` with `MobileNet SSD`

---

## ⚙️ Folder Structure (Suggested)

object_talker_ai/
│
├── main.py # Starts webcam + detection + speaker
├── detector.py # Handles object detection logic
├── speaker.py # Text-to-speech logic (pyttsx3)
├── utils.py # Cooldown logic
├── gui.py # Tkinter-based UI (optional)
├── requirements.txt # Dependencies
└── README.md # Project overview and instructions


---

## 🗣️ Sample Output

 🖼️ Detected: bottle, phone  
 🔊 Speaking: "I see a bottle"  
 🖼️ Detected: phone, person  
 🔊 Speaking: "There's a phone"

---

## ✅ Final Thoughts

This assistant turns your laptop into a **real-time visual observer** that **understands and narrates** its surroundings — useful in accessibility, ambient monitoring, or just fun AI experimentation.  
Built for **clarity, performance, and expandability**, perfect for expo demos and real-world applications alike.

learning commits lol