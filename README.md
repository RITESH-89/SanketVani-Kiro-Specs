# SanketVani-Kiro-Specs
An AI-powered assistive system enabling real-time two-way communication between sign language users and the hearing world.

🌐 SanketVani – Inclusive Communication Platform

Giving Voice to Hands | Turning Silence into Connection

🧩 Overview

SanketVani is an AI-powered assistive communication platform designed to bridge the communication gap between deaf / speech-impaired individuals and the hearing community.

Using computer vision, machine learning, and speech technologies, SanketVani enables real-time, two-way communication between sign language, text, and voice — all without any special hardware.

Accessibility is not charity — it is equality.

❗ Problem Statement

Millions of deaf and speech-impaired individuals face daily communication barriers in:

Education

Healthcare

Public services

Workplaces

Emergency situations

Key Challenges

Heavy dependence on human interpreters

Existing tools are costly, slow, or one-directional

Typing-based communication is unnatural and limiting

No unified, real-time solution available

💡 Solution – SanketVani

SanketVani provides a single, integrated platform that supports:

✋ Sign → Text

✋ Sign → Speech

🎤 Speech → Sign

⌨️ Text → Sign

📄 PDF / DOC → Sign

🎥 Live Video Call Integration (Sign–Speech overlay)

All processing is done using a camera + AI software, ensuring low cost, portability, and scalability.

⚙️ Technology Stack
Layer	Technologies
Programming	Python
Computer Vision	OpenCV, MediaPipe
Machine Learning	TensorFlow (CNN)
Speech Processing	SpeechRecognition, pyttsx3 / SAPI
UI / Integration	Tkinter, Virtual Camera
Hardware	Webcam / Laptop / Mobile Camera
🧠 Algorithms Used

MediaPipe Hand Landmark Detection
→ Tracks 21 key hand points in real time

Convolutional Neural Network (CNN)
→ Classifies gestures with ~92% accuracy

Speech-to-Text (STT)
→ Converts spoken language into text

Text-to-Speech (TTS)
→ Converts recognized text into natural voice

🔄 System Workflow

Camera captures hand gesture or voice input

AI model processes the input in real time

Gesture is classified using CNN

Output generated as text, speech, or sign animation

Reverse flow enables speech/text to sign conversion

Optional overlay during live video calls

📊 Results & Performance

✅ Accuracy: ~92% (live prototype testing)

⚡ Response Time: ~550–800 milliseconds

📷 Hardware: Standard webcam

🌐 Mode: Offline + Online support

🧪 Validation: Field testing at NGOs (Navjeevan, SOS)

🌍 Impact & Use Cases

🎓 Inclusive education for deaf students

🏥 Improved doctor–patient communication

🏢 Accessibility in government offices

🚨 Emergency communication support

👥 Independent daily conversations

SanketVani restores dignity, independence, and equality.

📈 Market Potential

India: ~6+ million deaf & speech-impaired individuals

Global: ~60+ million potential users

Target Users

Individuals

NGOs & special schools

Government institutions

Healthcare centers

🚀 Future Scope

Regional & multilingual sign support

Mobile application (Android / iOS)

AR/VR sign avatars

Government kiosk deployment

Offline-first optimization

📁 Project Structure
SanketVani/
├── requirements.md        # Kiro-generated requirements
├── design.md              # Kiro-generated system design
├── README.md              # Project documentation
├── texttosign.py
├── voicetosign.py
├── sign_recognition.py
├── videocall_virtual_cam.py
├── asl_signs/
└── assets/

📜 Documentation

This project follows Kiro’s Spec → Design workflow to generate:

requirements.md

design.md

These files define the functional requirements and system architecture in a professional, industry-standard format.

👨‍💻 Author

Ritesh Pravin Paithankar
Developer & Researcher
📍 India

❤️ Final Note

SanketVani is not just a technology — it is a movement.

It turns hands into voice and silence into connection.
