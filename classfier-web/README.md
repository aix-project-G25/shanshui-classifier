# Chinese-Japanese Landscape Painting Classifier

A single-page web application that classifies landscape paintings as either Chinese or Japanese using a ResNet18 deep learning model.

## Features

- Upload images via drag-and-drop or file selection
- Real-time classification with confidence scores
- Classification history with local storage
- Responsive design for desktop and mobile devices

## Project Structure

\`\`\`
.
├── app/                  # Next.js frontend
│   ├── page.tsx          # Main single-page application
│   └── layout.tsx        # Root layout
├── backend/              # Python FastAPI backend
│   ├── main.py           # API endpoints
│   ├── requirements.txt  # Python dependencies
└── README.md             # This file
\`\`\`

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   \`\`\`
   cd backend
   \`\`\`

2. Create a virtual environment:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Make sure your trained model file (`best_model.pth`) is in the backend directory.

5. Start the backend server:
   \`\`\`
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   \`\`\`

### Frontend Setup

1. Navigate to the project root directory.

2. Install dependencies:
   \`\`\`
   npm install
   \`\`\`

3. Start the development server:
   \`\`\`
   npm run dev
   \`\`\`

4. Open your browser and go to http://localhost:3000
