NeuroBright: A Neuroadaptive Learning Prototype

A Nallas Hackathon Submission

Live Demo: Add your Vercel URL

GitHub Repository: Add your repo URL

📌 1. Introduction

The global EdTech ecosystem — now a USD 20B industry — still relies mostly on static content delivery. The Nallas Hackathon challenges developers to combine:

Neuroscience

Brain–Computer Interfaces

Real-time cognitive sensing

Generative AI

AR/VR micro-interventions

…to create the next wave of brain-aware learning systems.

NeuroBright is our open-source neuroadaptive learning prototype that reacts to a learner’s cognitive state (Focused, Neutral, Drowsy) and dynamically adapts the content and experience.

🧠 2. Key Capabilities
✔ Real-Time BCI State Detection (Simulated + MATLAB Pipeline)

Supports real EEG device integration through MATLAB processing

Classifies: Focused, Neutral, Drowsy

✔ AI-Generated Learning Content (Gemini API)

Quizzes

Comics

Detailed explanations

Course outline

Adaptive analogies

✔ Adaptive Micro-Interventions

Triggered automatically during a Drowsy state:

VR Mini-Game (WebXR)

Break Timer

Leaderboard Competition

✔ Immersive VR Mini-Game (WebXR + Three.js)

Designed as a subconscious coaching tool to restore attention.

✔ Pre-Packaged Optics Module

Includes:

Static content

Images

AI comics

3D visuals

📊 3. Nallas Hackathon Problem Statement (Included)

We directly incorporate the official challenge text from your PDF.

🗂 Problem Statement (PDF Preview)

You can access the provided challenge document here:
📄 Neuroadaptive EdTech Hackathon Document.pdf)

Note: GitHub cannot render PDFs inline, but this link works in the repo.

Extract (From PDF):

Create an open-source neuroadaptive learning prototype that harnesses real-time brainwave data from consumer BCI devices to trigger creative, context-aware micro-interventions and dynamically generate personalized learning artifacts—such as bespoke analogies, AR experiences, or adaptive quizzes—based on each learner’s cognitive and emotional state.
Push beyond conventional UX—blend waking and dreaming states, prioritize user comfort, and build with transparency around consent and data privacy.

🧬 4. Real EEG Device Support (MATLAB → NeuroBright Pipeline)

Although the demo uses simulated BCI, the system also supports full integration with MATLAB EEG input files.

Pipeline Overview
Real EEG Device → MATLAB Script → Preprocessing → Feature Extraction → Classification  
→ Output JSON → NeuroBright Frontend → Intervention Engine

MATLAB Processing Steps

Import raw EEG signals (OpenBCI, Muse, Emotiv).

Preprocess

Bandpass 1–50 Hz

Notch filter (50/60 Hz)

Artifact removal (blink, muscle noise)

Extract features

Alpha (relaxed)

Beta (focused)

Theta (drowsy)

Classify state

SVM / KNN / Brain.js model

Write output file

Example state.json:

{
  "timestamp": 1732123456,
  "predicted_state": "Drowsy",
  "alpha": 0.21,
  "beta": 0.47,
  "theta": 0.36
}

NeuroBright Frontend Polls This File

Once new values arrive, the UI adapts dynamically.

🔧 5. Technology Stack
Frontend

React + TypeScript

Vite

Tailwind CSS

3D / VR

Three.js

WebXR

AI

Google Gemini API

EEG & Brain State Detection

MATLAB

Brain.js

OpenBCI SDK compatible

📦 6. Project Structure
/
├── eeg-processing/        # MATLAB scripts for EEG preprocessing & classification
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   └── features/
│       ├── course-chat/
│       │   ├── CourseChat.tsx
│       │   └── optics-data.ts
│       └── webxr/
│           └── WebXRDemo.tsx
├── vercel.json
└── README.md

🔁 7. System Flow Diagram
graph TD
    A[EEG Device] --> B[MATLAB Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Classification (Focused / Neutral / Drowsy)]
    D --> E[Output state.json]
    E --> F[NeuroBright Frontend]
    F --> G{State Engine}
    G -- Focused/Neutral --> H[Continue Learning]
    G -- Drowsy --> I[Trigger Interventions]
    I --> J[VR Game / Break Timer / Leaderboard]
    J --> F

🎮 8. VR Mini-Game (WebXR + Three.js)

Fast-paced

Stationary (to prevent motion sickness)

Designed to re-engage attention

Activated only during "Drowsy" episodes

🔒 9. Ethics & Safety

Zero EEG data leaves the device

User consent required before VR

Interventions are optional

Short VR sessions to avoid discomfort

🚀 10. Running the Project
Local Setup
npm install
npm run dev

To Use With Real EEG (MATLAB)

Connect EEG device

Run MATLAB script

Ensure JSON file updates

NeuroBright auto-updates based on state

Deployment

GitHub → Vercel

🛣 11. Roadmap

Native OpenBCI streaming

Deep-learning EEG state classifier

Personalized learning path memory

AR-based cognitive hints

🎉 12. Credits

Developed for Nallas Hackathon – Neuroadaptive EdTech Challenge.
