# Keyword Spotting (KWS) Project using ESP32 and TensorFlow Lite Micro

## Project Overview
This project demonstrates a **Keyword Spotting (KWS) system** that uses **TinyML** to recognize specific spoken commands. The system runs on an **ESP32** microcontroller and uses the **SPH0645 MEMS microphone** to capture audio input. It leverages **TensorFlow Lite for Microcontrollers** to perform real-time inference on five key words:
- **Go**
- **Stop**
- **Unknown**
- **Silence**

The goal of this project is to showcase how powerful machine learning models can be deployed on resource-constrained devices for edge AI applications.

---

## Features
- **Five-word detection**: The system can detect the following words: go, stop, left, right, and silence.
- **Real-time inference**: Uses TensorFlow Lite Micro to run machine learning models in real-time on the ESP32.
- **Efficient edge AI**: Optimized for low-power and low-latency applications with minimal hardware.
  
---

## Hardware Requirements
- **ESP32 microcontroller**
- **SPH0645 MEMS microphone**
- USB cable for programming and power supply

---

## Software Requirements
- **Esp-idf** for ESP32 programming
- **TensorFlow Lite for Microcontrollers** library
- **Python 3.x** for training and converting the model (if required)

---

## Demo
I have recorded a video demonstration of the system detecting spoken commands in real-time. Watch the demo to see the KWS system in action!
The video is in video folder

---

## Future Enhancements
- Add support for additional commands.
- Improve noise filtering for better accuracy in noisy environments.
- Explore other use cases for voice-controlled applications.

---

## Acknowledgements
- **HarvardX** for the excellent TinyML courses on edX.
- **TensorFlow Lite** for providing tools to enable ML on microcontrollers.
- The **TinyML community** for continued support and knowledge sharing.

---

