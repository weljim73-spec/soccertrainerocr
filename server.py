#!/usr/bin/env python3
"""
Soccer Session OCR Web Server
Simple Flask server for processing training session images with Claude Vision API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import base64
import re
from datetime import datetime
import os

# Configuration from environment
API_KEY_MODE = os.environ.get('API_KEY_MODE', 'user').lower()
SERVER_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_FILES = 20

# Validate server mode configuration
if API_KEY_MODE == 'server' and not SERVER_API_KEY:
    print("[ERROR] API_KEY_MODE is 'server' but ANTHROPIC_API_KEY not set!")
    print("[ERROR] Set ANTHROPIC_API_KEY or change API_KEY_MODE to 'user'")

app = Flask(__name__)

# Production: restrict origins, Development: allow all
if FLASK_ENV == 'production':
    CORS(app, origins=["https://soccertrainerocr.onrender.com"])
else:
    CORS(app)

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/config', methods=['GET'])
def get_config():
    """Return client configuration"""
    return jsonify({
        'api_key_mode': API_KEY_MODE,
        'max_file_size': MAX_FILE_SIZE,
        'max_files': MAX_FILES
    })

@app.route('/process', methods=['POST'])
def process_images():
    """Process uploaded images and extract session data"""
    try:
        session_type = request.form.get('session_type')
        files = request.files.getlist('images')

        print(f"[INFO] [{API_KEY_MODE.upper()} MODE] Processing request - Session: {session_type}, Files: {len(files)}")

        if not files:
            return jsonify({'error': 'No images uploaded'}), 400

        # Validate file count
        if len(files) > MAX_FILES:
            return jsonify({'error': f'Too many files. Maximum {MAX_FILES} allowed'}), 400

        # Validate file sizes
        for file in files:
            file.seek(0, 2)
            size = file.tell()
            file.seek(0)
            if size > MAX_FILE_SIZE:
                return jsonify({'error': f'File {file.filename} exceeds {MAX_FILE_SIZE/1024/1024}MB limit'}), 400

        # Get API key based on mode
        if API_KEY_MODE == 'server':
            api_key = SERVER_API_KEY
            if not api_key:
                print("[ERROR] Server mode but no API key configured")
                return jsonify({'error': 'Server configuration error'}), 500
        else:
            # User mode - get from request
            api_key = request.form.get('api_key')
            if not api_key:
                return jsonify({'error': 'API key is required'}), 400

        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Create detailed prompt based on session type
        if session_type == "match":
            extraction_prompt = """Extract all data from this soccer training/match session screenshot. Include ALL fields you can find:

Session: date, time of day, duration, opponent team name
Overview: position played, goals, assists, team scores
Skills: two-footed score, dribbling score, first touch score, agility score, speed score, power score
Highlights: work rate (yd/min), ball possessions, total distance (mi), sprint distance (yd), top speed (mph), kicking power (mph)
Two-Footed: left foot touches (#, %), right foot touches (#, %), left foot releases (#, %), right foot releases (#, %), left foot receives (#, %), right foot receives (#, %), left kicking power (mph), right kicking power (mph)
Dribbling: distance with ball (yd), top speed with ball (mph), intense turns with ball
First Touch: one-touch possessions, multiple-touch possessions, total duration (sec), ball release footzone (laces, inside, other)
Agility: left turns, back turns, right turns, intense turns, average turn entry speed (mph), average turn exit speed (mph)
Speed: top speed (mph), number of sprints
Power: first step accelerations, intense accelerations

Extract ALL text, labels, numbers, and values exactly as shown."""
        elif session_type == "ball_work":
            extraction_prompt = """Extract all data from this soccer ball work session screenshot. Include ALL fields:

Session: date, time, duration, training type, intensity
Highlights: ball touches, total distance (mi), sprint distance (yd), accl/decl, kicking power (mph)
Two-Footed: left/right touches (#, %), left/right releases (#, %), left/right kicking power (mph)
Speed: top speed (mph), sprints
Agility: left/back/right/intense turns, avg turn entry/exit speeds (mph)

Extract ALL text exactly as shown."""
        else:  # speed_agility
            extraction_prompt = """Extract all data from this speed & agility session screenshot. Include ALL fields:

Session: date, time, duration, training type, intensity
Highlights: total distance (mi), sprint distance (yd), accl/decl
Speed: top speed (mph), number of sprints
Agility: left/back/right/intense turns, avg turn entry/exit speeds (mph)

Extract ALL text exactly as shown."""

        # Process each image and extract text
        ocr_texts = []
        for i, file in enumerate(files):
            print(f"[INFO] Processing image {i+1}/{len(files)}: {file.filename}")
            image_data = base64.standard_b64encode(file.read()).decode('utf-8')

            # Determine media type
            media_type = 'image/jpeg'
            if file.filename.lower().endswith('.png'):
                media_type = 'image/png'

            # Call Claude Vision API
            print(f"[INFO] Calling Claude API for image {i+1}...")
            message = client.messages.create(
                model="claude-3-haiku-20240307",  # Claude 3 Haiku - fast and available!
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": extraction_prompt
                        }
                    ]
                }]
            )

            ocr_text = message.content[0].text
            print(f"[INFO] OCR extracted {len(ocr_text)} characters")
            ocr_texts.append(ocr_text)

        # Combine all OCR text
        combined_text = "\n\n".join(ocr_texts)
        print(f"[INFO] Combined OCR text length: {len(combined_text)}")

        # Extract structured data based on session type
        print(f"[INFO] Extracting structured data for {session_type}...")
        if session_type == 'ball_work':
            result = extract_ball_work_data(combined_text)
        elif session_type == 'speed_agility':
            result = extract_speed_agility_data(combined_text)
        elif session_type == 'match':
            result = extract_match_data(combined_text)
        else:
            return jsonify({'error': 'Invalid session type'}), 400

        print(f"[SUCCESS] Extraction complete!")
        return jsonify({
            'success': True,
            'data': result,
            'ocr_text': combined_text
        })

    except anthropic.AuthenticationError as e:
        print(f"[ERROR] Authentication error: {str(e)}")
        error_msg = 'Invalid API key' if API_KEY_MODE == 'user' else 'Server API key invalid'
        return jsonify({'error': error_msg}), 401
    except anthropic.RateLimitError as e:
        print(f"[ERROR] Rate limit: {str(e)}")
        return jsonify({'error': 'Rate limit exceeded. Try again later.'}), 429
    except Exception as e:
        print(f"[ERROR] Unexpected: {type(e).__name__}: {str(e)}")
        if FLASK_ENV == 'development':
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Processing error occurred'}), 500


def extract_ball_work_data(text):
    """Extract Ball Work session data from OCR text"""
    text = text.lower()

    # Session info
    session_name_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening)[^,]*)',
        text, re.IGNORECASE
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})',
        text, re.IGNORECASE
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), '%B %d, %Y')
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text, r'(\d+)\s*min')
    training_type = extract_value(text, r'(technical|physical|tactical)', default='Technical')
    intensity = extract_value(text, r'(low|moderate|high)', default='Moderate')

    # Highlights
    ball_touches = extract_number(text, r'ball\s*touches[:\s]*(\d+)')
    total_distance = extract_number(text, r'total\s*distance[:\s]*([\d.]+)')
    sprint_distance = extract_number(text, r'sprint\s*distance[:\s]*([\d.]+)')
    accl_decl = extract_number(text, r'accl\s*/\s*decl[:\s]*(\d+)')
    kicking_power = extract_number(text, r'kicking\s*power[:\s]*([\d.]+)')

    # Two-footed
    left_touches = extract_number(text, r'left\s*foot[^:]*touch[^:]*[:\s]*(\d+)')
    left_touches_pct = extract_number(text, r'left\s*foot[^:]*touch[^:]*\((\d+)%\)')
    right_touches = extract_number(text, r'right\s*foot[^:]*touch[^:]*[:\s]*(\d+)')
    right_touches_pct = extract_number(text, r'right\s*foot[^:]*touch[^:]*\((\d+)%\)')

    left_releases = extract_number(text, r'left\s*foot[^:]*release[^:]*[:\s]*(\d+)')
    left_releases_pct = extract_number(text, r'left\s*foot[^:]*release[^:]*\((\d+)%\)')
    right_releases = extract_number(text, r'right\s*foot[^:]*release[^:]*[:\s]*(\d+)')
    right_releases_pct = extract_number(text, r'right\s*foot[^:]*release[^:]*\((\d+)%\)')

    left_kicking = extract_number(text, r'left\s*foot\s*kicking\s*power[:\s]*([\d.]+)')
    right_kicking = extract_number(text, r'right\s*foot\s*kicking\s*power[:\s]*([\d.]+)')

    # Speed
    top_speed = extract_number(text, r'top\s*speed[:\s]*([\d.]+)')
    sprints = extract_number(text, r'sprints[:\s]*(\d+)')

    # Agility
    left_turns = extract_number(text, r'left\s*turns[:\s]*(\d+)')
    back_turns = extract_number(text, r'back\s*turns[:\s]*(\d+)')
    right_turns = extract_number(text, r'right\s*turns[:\s]*(\d+)')
    intense_turns = extract_number(text, r'intense\s*turns[:\s]*(\d+)')
    entry_speed = extract_number(text, r'(?:average\s*)?turn\s*entry\s*speed[:\s]*([\d.]+)')
    exit_speed = extract_number(text, r'(?:average\s*)?turn\s*exit\s*speed[:\s]*([\d.]+)')

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": training_type,
            "intensity": intensity
        },
        "highlights": {
            "ball_touches": ball_touches,
            "total_distance_miles": total_distance,
            "sprint_distance_yards": sprint_distance,
            "accl_decl": accl_decl,
            "kicking_power_mph": kicking_power
        },
        "two_footed": {
            "left_foot_touches": left_touches,
            "left_foot_touches_percentage": left_touches_pct,
            "right_foot_touches": right_touches,
            "right_foot_touches_percentage": right_touches_pct,
            "left_foot_releases": left_releases,
            "left_foot_releases_percentage": left_releases_pct,
            "right_foot_releases": right_releases,
            "right_foot_releases_percentage": right_releases_pct,
            "left_foot_kicking_power_mph": left_kicking,
            "right_foot_kicking_power_mph": right_kicking
        },
        "speed": {
            "top_speed_mph": top_speed,
            "sprints": sprints
        },
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "average_turn_entry_speed_mph": entry_speed,
            "average_turn_exit_speed_mph": exit_speed
        }
    }


def extract_speed_agility_data(text):
    """Extract Speed & Agility session data from OCR text"""
    text = text.lower()

    # Session info
    session_name_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening)[^,]*)',
        text, re.IGNORECASE
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})',
        text, re.IGNORECASE
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), '%B %d, %Y')
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text, r'(\d+)\s*min')
    training_type = extract_value(text, r'(technical|physical|tactical)', default='Physical')
    intensity = extract_value(text, r'(low|moderate|high)', default='Moderate')

    # Highlights
    total_distance = extract_number(text, r'total\s*distance[:\s]*([\d.]+)')
    sprint_distance = extract_number(text, r'sprint\s*distance[:\s]*([\d.]+)')
    accl_decl = extract_number(text, r'accl\s*/\s*decl[:\s]*(\d+)')

    # Speed
    top_speed = extract_number(text, r'top\s*speed[:\s]*([\d.]+)')
    sprints = extract_number(text, r'sprints[:\s]*(\d+)')

    # Agility
    left_turns = extract_number(text, r'left\s*turns[:\s]*(\d+)')
    back_turns = extract_number(text, r'back\s*turns[:\s]*(\d+)')
    right_turns = extract_number(text, r'right\s*turns[:\s]*(\d+)')
    intense_turns = extract_number(text, r'intense\s*turns[:\s]*(\d+)')
    entry_speed = extract_number(text, r'(?:average\s*)?turn\s*entry\s*speed[:\s]*([\d.]+)')
    exit_speed = extract_number(text, r'(?:average\s*)?turn\s*exit\s*speed[:\s]*([\d.]+)')

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": training_type,
            "intensity": intensity
        },
        "highlights": {
            "total_distance_miles": total_distance,
            "sprint_distance_yards": sprint_distance,
            "accl_decl": accl_decl
        },
        "speed": {
            "top_speed_mph": top_speed,
            "sprints": sprints
        },
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "average_turn_entry_speed_mph": entry_speed,
            "average_turn_exit_speed_mph": exit_speed
        }
    }


def extract_match_data(text):
    """Extract Match session data from OCR text"""
    text_lower = text.lower()

    # Session info
    session_name_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening))',
        text_lower, re.IGNORECASE
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})',
        text_lower, re.IGNORECASE
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), '%B %d, %Y')
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text_lower, r'(\d+)\s*min')

    # Match overview
    position = extract_value(text_lower, r'([a-z]{1,3})\s+position', default=None)
    goals = extract_number(text_lower, r'goals\s+(\d+)')
    assists = extract_number(text_lower, r'assists\s+(\d+)')

    # Team scores - looking for pattern like "cityplay fc 1 : 4 fc westlake"
    score_match = re.search(r'(\d+)\s*:\s*(\d+)', text_lower)
    if score_match:
        athlete_score = int(score_match.group(1))
        opposing_score = int(score_match.group(2))
    else:
        athlete_score = None
        opposing_score = None

    # Opponent name - text after the score pattern
    opponent_match = re.search(r'\d+\s*:\s*\d+\s+(.+?)(?:\n|$)', text_lower)
    opposing_team_name = opponent_match.group(1).strip() if opponent_match else None

    # Skills scores - simpler patterns matching "two-footed 55" format
    two_footed = extract_number(text_lower, r'two-?footed\s+(\d+)')
    dribbling = extract_number(text_lower, r'dribbling\s+(\d+)')
    first_touch = extract_number(text_lower, r'first\s+touch\s+(\d+)')
    agility_score = extract_number(text_lower, r'agility\s+(\d+)')
    speed_score = extract_number(text_lower, r'speed\s+(\d+)')
    power_score = extract_number(text_lower, r'power\s+(\d+)')

    # Highlights - simpler patterns matching actual format
    work_rate = extract_number(text_lower, r'work\s+rate\s+([\d.]+)')
    ball_possessions = extract_number(text_lower, r'ball\s+possessions\s+(\d+)')
    total_distance = extract_number(text_lower, r'total\s+distance\s+([\d.]+)')
    sprint_distance = extract_number(text_lower, r'sprint\s+distance\s+([\d.]+)')
    top_speed = extract_number(text_lower, r'top\s+speed\s+([\d.]+)')
    kicking_power = extract_number(text_lower, r'kicking\s+power\s+([\d.]+)')

    # Two-footed
    left_touches = extract_number(text_lower, r'left\s+foot[^:]*touch[^:]*[:\s]*(\d+)')
    left_touches_pct = extract_number(text_lower, r'left\s+foot[^:]*touch[^:]*\((\d+)%\)')
    right_touches = extract_number(text_lower, r'right\s+foot[^:]*touch[^:]*[:\s]*(\d+)')
    right_touches_pct = extract_number(text_lower, r'right\s+foot[^:]*touch[^:]*\((\d+)%\)')

    left_releases = extract_number(text_lower, r'left\s+foot[^:]*release[^:]*[:\s]*(\d+)')
    left_releases_pct = extract_number(text_lower, r'left\s+foot[^:]*release[^:]*\((\d+)%\)')
    right_releases = extract_number(text_lower, r'right\s+foot[^:]*release[^:]*[:\s]*(\d+)')
    right_releases_pct = extract_number(text_lower, r'right\s+foot[^:]*release[^:]*\((\d+)%\)')

    left_receives = extract_number(text_lower, r'left\s+foot[^:]*receive[^:]*[:\s]*(\d+)')
    left_receives_pct = extract_number(text_lower, r'left\s+foot[^:]*receive[^:]*\((\d+)%\)')
    right_receives = extract_number(text_lower, r'right\s+foot[^:]*receive[^:]*[:\s]*(\d+)')
    right_receives_pct = extract_number(text_lower, r'right\s+foot[^:]*receive[^:]*\((\d+)%\)')

    left_kicking = extract_number(text_lower, r'left\s+foot\s*kicking\s*power[:\s]*(\d+\.?\d*)')
    right_kicking = extract_number(text_lower, r'right\s+foot\s*kicking\s*power[:\s]*(\d+\.?\d*)')

    # Dribbling
    distance_with_ball = extract_number(text_lower, r'distance\s+with\s+ball[:\s]*(\d+\.?\d*)\s*yd')
    top_speed_with_ball = extract_number(text_lower, r'top\s+speed\s+with\s+ball[:\s]*(\d+\.?\d*)\s*mph')
    intense_turns_with_ball = extract_number(text_lower, r'intense\s+turns\s+with\s+ball[:\s]*(\d+)')

    # First touch - possessions
    one_touch_poss = extract_number(text_lower, r'one[- ]touch[:\s]*(\d+)')
    multiple_touch_poss = extract_number(text_lower, r'multiple[- ]touch[:\s]*(\d+)')
    total_duration_sec = extract_number(text_lower, r'total\s+duration[:\s]*(\d+\.?\d*)\s*s')

    # First touch - ball release footzone
    laces = extract_number(text_lower, r'laces[:\s]*(\d+)')
    inside = extract_number(text_lower, r'inside[:\s]*(\d+)')
    other_footzone = extract_number(text_lower, r'other[:\s]*(\d+)')

    # Agility
    left_turns = extract_number(text_lower, r'(\d+)[^\d]*\d+[^\d]*\d+[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?')
    back_turns = extract_number(text_lower, r'\d+[^\d]*(\d+)[^\d]*\d+[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?')
    right_turns = extract_number(text_lower, r'\d+[^\d]*\d+[^\d]*(\d+)[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?')
    intense_turns = extract_number(text_lower, r'intense\s+turns?(?!\s+with\s+ball)\s*[:\s#]*(\d+)')
    entry_speed = extract_number(text_lower, r'(?:average\s*)?(?:turn|tum)\s+entry\s+speed[:\s]*(\d+\.?\d*)')
    exit_speed = extract_number(text_lower, r'(?:average\s*)?(?:turn|tum)\s+exit\s+speed[:\s]*(\d+\.?\d*)')

    # Speed
    sprints = extract_number(text_lower, r'sprints?\s*[:\s#]*(\d+)')

    # Power
    first_step_accel = extract_number(text_lower, r'first[- ]step[:\s]*(\d+)')
    intense_accel = extract_number(text_lower, r'intense\s+(?:accel|acceleration)[:\s]*(\d+)')

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": "Match"
        },
        "overview": {
            "position": position,
            "goals": goals,
            "assists": assists,
            "athlete_team_score": athlete_score,
            "opposing_team_score": opposing_score,
            "opposing_team_name": opposing_team_name
        },
        "skills": {
            "two_footed_score": two_footed,
            "dribbling_score": dribbling,
            "first_touch_score": first_touch,
            "agility_score": agility_score,
            "speed_score": speed_score,
            "power_score": power_score
        },
        "highlights": {
            "work_rate_yd_per_min": work_rate,
            "ball_possessions": ball_possessions,
            "total_distance_mi": total_distance,
            "sprint_distance_yd": sprint_distance,
            "top_speed_mph": top_speed,
            "kicking_power_mph": kicking_power
        },
        "two_footed": {
            "left_foot_touches": left_touches,
            "left_foot_touches_pct": left_touches_pct,
            "right_foot_touches": right_touches,
            "right_foot_touches_pct": right_touches_pct,
            "left_foot_releases": left_releases,
            "left_foot_releases_pct": left_releases_pct,
            "right_foot_releases": right_releases,
            "right_foot_releases_pct": right_releases_pct,
            "left_foot_receives": left_receives,
            "left_foot_receives_pct": left_receives_pct,
            "right_foot_receives": right_receives,
            "right_foot_receives_pct": right_receives_pct,
            "left_foot_kicking_power_mph": left_kicking,
            "right_foot_kicking_power_mph": right_kicking
        },
        "dribbling": {
            "distance_with_ball_yd": distance_with_ball,
            "top_speed_with_ball_mph": top_speed_with_ball,
            "intense_turns_with_ball": intense_turns_with_ball
        },
        "first_touch": {
            "ball_possessions": {
                "total": ball_possessions,
                "one_touch": one_touch_poss,
                "multiple_touch": multiple_touch_poss,
                "total_duration_sec": total_duration_sec
            },
            "ball_release_footzone": {
                "laces": laces,
                "inside": inside,
                "other": other_footzone
            }
        },
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "avg_turn_entry_speed_mph": entry_speed,
            "avg_turn_exit_speed_mph": exit_speed
        },
        "speed": {
            "top_speed_mph": top_speed,
            "sprints": sprints
        },
        "power": {
            "first_step_accelerations": first_step_accel,
            "intense_accelerations": intense_accel
        }
    }


def extract_number(text, pattern):
    """Extract a number from text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        except ValueError:
            return None
    return None


def extract_value(text, pattern, default=None):
    """Extract a text value from text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return default


if __name__ == '__main__':
    print(f"Starting Soccer OCR Server...")
    print(f"Mode: {API_KEY_MODE.upper()}")
    print(f"Environment: {FLASK_ENV}")
    if FLASK_ENV == 'production':
        print("WARNING: Use gunicorn for production!")

    port = int(os.environ.get('PORT', 5000))
    debug = FLASK_ENV == 'development'
    print(f"Open http://localhost:{port} in your browser")
    app.run(host='0.0.0.0', port=port, debug=debug)
