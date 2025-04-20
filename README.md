
# Keyword Spotting (KWS) Project using ESP32 and TensorFlow Lite Micro

## Project Overview
This project demonstrates a **Keyword Spotting (KWS) system** using **TinyML** to recognize spoken commands. The system runs on an **ESP32** microcontroller and uses the **SPH0645 MEMS microphone** to capture audio input. It leverages **TensorFlow Lite for Microcontrollers** to perform real-time inference on four classes:
- **Go**
- **Stop**
- **Silence**
- **Unknown**

The goal is to showcase how machine learning models can run efficiently on resource-constrained embedded systems for edge AI applications.

---

## Repository Structure
```
KWS_model/
├── prepare_dataset.ipynb   : Prepares the dataset
├── kaggle_make_model.ipynb : Builds the model and converts it to a C++ array (intended for use on Kaggle)

KWS_with_ESP32_SPH0645/
├── C/C++ code for ESP32 using ESP-IDF
├── Integration with TensorFlow Lite Micro and SPH0645
```

---

## Features
- **Command Detection**: Recognizes "go" and "stop" commands, along with handling silence and unknown words.
- **Real-time inference**: Runs the ML model directly on ESP32 using TFLite Micro.
- **Edge optimized**: Low-power, low-latency inference ideal for embedded AI applications.

---

## Hardware Requirements
- **ESP32 microcontroller**
- **SPH0645 MEMS microphone**

---

## Software Requirements
- **ESP-IDF** (ESP32 development framework)
- **TensorFlow **
- **TensorFlow Lite for Microcontrollers**
---

## Demo
A demonstration video showing the system detecting voice commands in real-time is included in the `video/` folder.


---

## Acknowledgements
- **HarvardX** – for the excellent TinyML courses on edX.
- **TensorFlow Lite** – for enabling ML on microcontrollers.
- The **TinyML community** – for continuous support and shared knowledge.

---

