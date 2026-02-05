# Soccer Session OCR Web App

A simple web interface for processing soccer training session screenshots using Claude Vision API.

## Features

- 🖼️ Upload multiple images via click or drag-and-drop
- 🤖 Uses your own Anthropic API key (Claude Vision)
- ⚽ Supports three session types:
  - Ball Work
  - Speed & Agility
  - Match
- 📥 Download results as JSON
- 📋 Copy to clipboard
- 🎨 Beautiful, modern interface

## Setup

### 1. Install Dependencies

```bash
cd ocr_web_app
pip3 install -r requirements.txt
```

### 2. Get Your Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key (starts with `sk-ant-api03-...`)

### 3. Start the Server

```bash
python3 server.py
```

The server will start on http://localhost:5000

### 4. Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

## Usage

1. **Enter API Key**: Paste your Anthropic API key in the first field
2. **Select Session Type**: Choose Ball Work, Speed & Agility, or Match
3. **Upload Images**:
   - Click the upload area or drag-and-drop your screenshots
   - You can upload multiple images at once
   - Supported formats: PNG, JPG, JPEG
4. **Process**: Click "Process Images" button
5. **Get Results**:
   - View the extracted JSON data
   - Download as JSON file
   - Copy to clipboard

## Session Types & Images

### Ball Work
Upload 4 images showing:
- Main overview (session name, duration, training type, intensity, highlights)
- Two-Footed performance (touches, releases, kicking power)
- Speed performance (top speed, sprints)
- Agility performance (turns, entry/exit speeds)

### Speed & Agility
Upload 2 images showing:
- Main overview (session name, duration, highlights, speed metrics)
- Agility performance (turns, entry/exit speeds)

### Match
Upload 11 images showing:
- Main overview and various performance metrics
- Match-specific data (goals, assists, scores, positions)

## Output Format

The app generates JSON in the format expected by your cityplaysensorapp:

```json
{
  "session": {
    "session_name": "February 4, 2026 Afternoon - 15/14 White Training",
    "date": "2026-02-04",
    "duration_minutes": 81,
    "training_type": "Technical",
    "intensity": "Moderate"
  },
  "highlights": { ... },
  "two_footed": { ... },
  "speed": { ... },
  "agility": { ... }
}
```

## API Key Security

- Your API key is sent directly to the backend server
- The key is NOT stored or logged
- Each request uses your key to call Anthropic's API
- Keep your API key private and never share it

## Troubleshooting

### Server won't start
- Make sure you installed all dependencies: `pip3 install -r requirements.txt`
- Check if port 5000 is already in use
- Try a different port: `python3 server.py` and edit the port in server.py

### "Failed to connect to server"
- Make sure the server is running (you should see "Running on http://localhost:5000")
- Check your firewall settings
- Try refreshing the browser

### "Invalid API key"
- Verify your API key is correct
- Make sure it starts with `sk-ant-api03-`
- Check that your account has API access enabled

### Poor extraction results
- Ensure images are clear and readable
- Upload all required images for the session type
- Check that text in images is not too small or blurry

## Cost Estimation

Using Claude Vision API costs approximately:
- $3 per 1000 images (Sonnet 3.5)
- Each session processing costs ~$0.01-$0.03 depending on image count

## Development

### File Structure
```
ocr_web_app/
├── server.py         # Flask backend server
├── index.html        # Frontend interface
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

### Customization

To modify extraction logic, edit the extraction functions in `server.py`:
- `extract_ball_work_data()`
- `extract_speed_agility_data()`
- `extract_match_data()`

## Deployment to Render.com

### Quick Deploy (Server-Managed API Key)

1. **Push to GitHub:**
   ```bash
   cd web_app
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/weljim73-spec/soccertrainerocr.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to https://render.com and sign up
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Configure:
     - Name: soccer-ocr-app
     - Build: `pip install -r requirements.txt`
     - Start: `gunicorn server:app`
     - Plan: Free

3. **Set Environment Variables:**
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY
   API_KEY_MODE=server
   FLASK_ENV=production
   ```

4. **Update CORS (after first deploy):**
   - Edit server.py line 18 with your Render URL
   - Commit and push to redeploy

### Switch to User-Managed Keys

On Render dashboard:
1. Environment tab
2. Change `API_KEY_MODE` from `server` to `user`
3. Service auto-restarts
4. Users now provide their own API keys

## Support

For issues or questions about:
- The web app: Check the code in server.py and index.html
- Anthropic API: Visit https://docs.anthropic.com/
- cityplaysensorapp integration: Check your app's import functionality
