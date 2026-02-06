#!/usr/bin/env python3
"""
Soccer Session OCR Web Server
Simple Flask server for processing training session images with Claude Vision API
"""

import base64
import os
import re
from datetime import datetime

import anthropic
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Configuration from environment
API_KEY_MODE = os.environ.get("API_KEY_MODE", "user").lower()
SERVER_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FLASK_ENV = os.environ.get("FLASK_ENV", "development")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_FILES = 20

# Validate server mode configuration
if API_KEY_MODE == "server" and not SERVER_API_KEY:
    print("[ERROR] API_KEY_MODE is 'server' but ANTHROPIC_API_KEY not set!")
    print("[ERROR] Set ANTHROPIC_API_KEY or change API_KEY_MODE to 'user'")

app = Flask(__name__)

# Production: restrict origins, Development: allow all
if FLASK_ENV == "production":
    CORS(app, origins=["https://soccertrainerocr.onrender.com"])
else:
    CORS(app)


# Serve the frontend
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/config", methods=["GET"])
def get_config():
    """Return client configuration"""
    return jsonify(
        {
            "api_key_mode": API_KEY_MODE,
            "max_file_size": MAX_FILE_SIZE,
            "max_files": MAX_FILES,
            "session_type_limits": {
                "speed_agility": {"max": 2, "description": "Speed & Agility: 1-2 images"},
                "ball_work": {"max": 4, "description": "Ball Work: 1-4 images"},
                "match": {"max": MAX_FILES, "description": "Match: Multiple images (typically 11)"},
            },
        }
    )


def validate_session_type(client, files, claimed_type):
    """Validate that uploaded images match the claimed session type.

    Args:
        client: Anthropic client instance
        files: List of uploaded file objects
        claimed_type: Session type claimed by user ("match", "ball_work", "speed_agility")

    Returns:
        tuple: (is_valid: bool, detected_type: str, confidence: str)
    """
    print(f"[INFO] Validating session type - claimed: {claimed_type}")

    # Sample 1-2 images for validation (first and middle if more than 3 images)
    sample_files = [files[0]]
    if len(files) > 3:
        sample_files.append(files[len(files) // 2])

    # Prepare images for validation
    validation_content = []
    for file in sample_files:
        file.seek(0)
        image_data = base64.standard_b64encode(file.read()).decode("utf-8")
        file.seek(0)  # Reset for later use

        media_type = "image/jpeg"
        if file.filename.lower().endswith(".png"):
            media_type = "image/png"

        validation_content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_data},
            }
        )

    # Add validation prompt
    validation_prompt = """Analyze these soccer training session screenshots and identify the session type.

CRITICAL: Match sessions include ALL the same metrics as Ball Work sessions PLUS game-specific information.

**Match Session (GAME):**
MUST HAVE at least ONE of these game-specific indicators:
- Opposing team name
- Goals and assists scored
- Team scores (e.g., "2-1" or "Team Score: 3, Opponent: 1")
- Position played (FWD, MID, DEF, GK)
- Skill scores shown as circular percentages (Two-Footed, Dribbling, First Touch)

Match sessions will ALSO have ball work metrics like touches, kicking power, releases - don't let these fool you into thinking it's Ball Work!

**Ball Work Session (TRAINING):**
- Has ball touches, kicking power, left/right metrics
- NO opposing team name
- NO game scores or opponent
- NO position played
- NO skill score circles

**Speed and Agility Session:**
- Focus on movement: distance, speed, sprints, turns
- Minimal or no ball metrics

Respond with ONLY one of these exact phrases:
- "Match" (if you see ANY game/opponent info)
- "Ball Work" (if ONLY training metrics, no game info)
- "Speed and Agility" (if movement-focused)

If uncertain, add "uncertain:" prefix (e.g., "uncertain: Match")"""

    validation_content.append({"type": "text", "text": validation_prompt})

    # Call Claude for validation (use Haiku for speed and cost)
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": validation_content}],
        )

        response = message.content[0].text.strip()
        print(f"[INFO] Validation response: '{response}'")

        # Parse response
        is_uncertain = response.lower().startswith("uncertain:")
        detected_type = response.replace("uncertain:", "").strip().lower()

        # Normalize detected type
        if "match" in detected_type:
            detected_type = "match"
        elif "ball work" in detected_type:
            detected_type = "ball_work"
        elif "speed" in detected_type and "agility" in detected_type:
            detected_type = "speed_agility"

        # Compare with claimed type
        is_valid = detected_type == claimed_type.lower()
        confidence = "uncertain" if is_uncertain else "confident"

        print(
            f"[INFO] Validation result - detected: {detected_type}, valid: {is_valid}, confidence: {confidence}"
        )
        return is_valid, detected_type, confidence

    except Exception as e:
        print(f"[WARNING] Validation failed: {str(e)}")
        # If validation fails, allow processing to continue (don't block on validation errors)
        return True, claimed_type, "validation_error"


@app.route("/process", methods=["POST"])
def process_images():
    """Process uploaded images and extract session data"""
    try:
        session_type = request.form.get("session_type")
        files = request.files.getlist("images")

        print(
            f"[INFO] [{API_KEY_MODE.upper()} MODE] Processing request - Session: {session_type}, Files: {len(files)}"
        )

        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        # Validate file count
        if len(files) > MAX_FILES:
            return jsonify({"error": f"Too many files. Maximum {MAX_FILES} allowed"}), 400

        # Validate session-type-specific image count (signature validation)
        session_type_limits = {
            "speed_agility": 2,
            "ball_work": 4,
            # Match sessions can have many images (no strict limit beyond MAX_FILES)
        }

        if session_type in session_type_limits:
            max_for_type = session_type_limits[session_type]
            if len(files) > max_for_type:
                type_display = session_type.replace("_", " ").title()
                return (
                    jsonify(
                        {
                            "error": f"{type_display} sessions should have at most {max_for_type} images. "
                            f"You uploaded {len(files)}. Please check your session type selection."
                        }
                    ),
                    400,
                )

        # Validate file sizes
        for file in files:
            file.seek(0, 2)
            size = file.tell()
            file.seek(0)
            if size > MAX_FILE_SIZE:
                return (
                    jsonify(
                        {"error": f"File {file.filename} exceeds {MAX_FILE_SIZE/1024/1024}MB limit"}
                    ),
                    400,
                )

        # Get API key based on mode
        if API_KEY_MODE == "server":
            api_key = SERVER_API_KEY
            if not api_key:
                print("[ERROR] Server mode but no API key configured")
                return jsonify({"error": "Server configuration error"}), 500
        else:
            # User mode - get from request
            api_key = request.form.get("api_key")
            if not api_key:
                return jsonify({"error": "API key is required"}), 400

        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Validate that images match the claimed session type
        is_valid, detected_type, confidence = validate_session_type(client, files, session_type)

        if not is_valid:
            # Special case: Match sessions include all Ball Work metrics, so Ball Work detection might be expected
            # Only block if we're confident AND it's not a Match/Ball Work confusion
            is_match_ballwork_confusion = (
                session_type == "match" and detected_type == "ball_work"
            ) or (session_type == "ball_work" and detected_type == "match")

            if confidence == "uncertain" or is_match_ballwork_confusion:
                # Allow uncertain matches or Match/Ball Work confusion to proceed
                print(
                    f"[WARNING] Validation uncertain or Match/Ball Work overlap - detected: {detected_type}, claimed: {session_type}, proceeding with user's selection"
                )
            else:
                # Only block if we're confident and it's a clear mismatch (e.g., Speed & Agility vs Match)
                display_detected = detected_type.replace("_", " ").title()
                display_claimed = session_type.replace("_", " ").title()

                error_msg = (
                    f"Session type mismatch: Images appear to be '{display_detected}' "
                    f"but you selected '{display_claimed}'. Please verify your selection."
                )
                print(f"[ERROR] {error_msg}")
                return jsonify({"error": error_msg}), 400

        elif confidence == "uncertain":
            print(
                "[WARNING] Session type detection was uncertain, proceeding with user's selection"
            )

        # Create detailed prompt based on session type - ask for JSON directly
        if session_type == "match":
            extraction_prompt = """Extract all data from these soccer match session screenshots and return as a single JSON object. Look across ALL images to find these fields. Return ONLY valid JSON, no other text.

Required format:
{
  "date": "YYYY-MM-DD",
  "session_name": "Date + time of day",
  "duration_minutes": number,
  "position": "2-3 letter position code (like RM, CM, LW)",
  "goals": number,
  "assists": number,
  "athlete_team_score": number (athlete's team score),
  "opposing_team_score": number (opponent's team score),
  "opposing_team_name": "opponent name",
  "two_footed_score": number,
  "dribbling_score": number,
  "first_touch_score": number,
  "agility_score": number,
  "speed_score": number,
  "power_score": number or null (if shows "no data", use null or 0),
  "work_rate": number (yd/min),
  "ball_possessions": number,
  "total_distance": number (miles),
  "sprint_distance": number (yards),
  "top_speed": number (mph),
  "kicking_power": number (mph),
  "left_touches": number,
  "left_touches_pct": number,
  "right_touches": number,
  "right_touches_pct": number,
  "left_releases": number,
  "left_releases_pct": number,
  "right_releases": number,
  "right_releases_pct": number,
  "left_receives": number,
  "left_receives_pct": number,
  "right_receives": number,
  "right_receives_pct": number,
  "left_kicking_power": number (mph - look for "Left Foot Kicking Power" on two-footed detail screen),
  "right_kicking_power": number (mph - look for "Right Foot Kicking Power" on two-footed detail screen),
  "distance_with_ball": number (yards),
  "top_speed_with_ball": number (mph),
  "intense_turns_with_ball": number,
  "one_touch_poss": number,
  "multiple_touch_poss": number,
  "total_duration_sec": number,
  "laces": number,
  "inside": number,
  "other_footzone": number,
  "left_turns": number,
  "back_turns": number,
  "right_turns": number,
  "intense_turns": number,
  "avg_turn_entry": number (mph),
  "avg_turn_exit": number (mph),
  "num_sprints": number (look for "Sprints" count, often shown with top speed),
  "first_step_accel": number,
  "intense_accel": number
}

Return only the JSON object with all available fields. Use null for missing values."""
        elif session_type == "ball_work":
            extraction_prompt = """Extract all data from these soccer ball work session screenshots and return as JSON. Return ONLY valid JSON.

Required format:
{
  "date": "YYYY-MM-DD",
  "session_name": "Date + time",
  "duration_minutes": number,
  "training_type": "Technical/Physical/Tactical",
  "intensity": "Low/Moderate/High",
  "ball_touches": number,
  "total_distance": number (miles),
  "sprint_distance": number (yards),
  "accelerations": number,
  "kicking_power": number (mph),
  "left_touches": number,
  "left_pct": number,
  "right_touches": number,
  "right_pct": number,
  "left_releases": number,
  "left_release_pct": number,
  "right_releases": number,
  "right_release_pct": number,
  "left_kicking_power": number (mph),
  "right_kicking_power": number (mph),
  "top_speed": number (mph),
  "num_sprints": number,
  "left_turns": number,
  "back_turns": number,
  "right_turns": number,
  "intense_turns": number,
  "avg_turn_entry": number (mph),
  "avg_turn_exit": number (mph)
}"""
        else:  # speed_agility
            extraction_prompt = """Extract all data from these speed & agility session screenshots and return as JSON. Return ONLY valid JSON.

Required format:
{
  "date": "YYYY-MM-DD",
  "session_name": "Date + time",
  "duration_minutes": number,
  "training_type": "Physical",
  "intensity": "Low/Moderate/High",
  "total_distance": number (miles),
  "sprint_distance": number (yards),
  "accelerations": number,
  "top_speed": number (mph),
  "num_sprints": number,
  "left_turns": number,
  "back_turns": number,
  "right_turns": number,
  "intense_turns": number,
  "avg_turn_entry": number (mph),
  "avg_turn_exit": number (mph)
}"""

        # Build content with all images + extraction prompt
        content = []

        # Add all images first
        for i, file in enumerate(files):
            print(f"[INFO] Preparing image {i+1}/{len(files)}: {file.filename}")
            image_data = base64.standard_b64encode(file.read()).decode("utf-8")

            # Determine media type
            media_type = "image/jpeg"
            if file.filename.lower().endswith(".png"):
                media_type = "image/png"

            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                }
            )

        # Add extraction prompt at the end
        content.append({"type": "text", "text": extraction_prompt})

        # Call Claude Vision API with ALL images at once
        print(f"[INFO] Calling Claude API with {len(files)} images...")
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Try Sonnet first for better extraction
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            print("[INFO] Using Claude 3.5 Sonnet")
        except Exception as e:
            # Fallback to Haiku if Sonnet not available
            print(f"[INFO] Sonnet not available ({str(e)}), falling back to Haiku")
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            print("[INFO] Using Claude 3 Haiku")

        response_text = message.content[0].text
        print(f"[INFO] Received response: {len(response_text)} characters")

        # Parse JSON response directly
        try:
            # Extract JSON from response (handle if Claude adds any wrapper text)
            import json

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                extracted_json = json.loads(json_str)
                print(f"[SUCCESS] Extracted {len(extracted_json)} fields from JSON")
            else:
                print("[ERROR] No JSON found in response")
                return jsonify({"error": "Failed to extract JSON from response"}), 500
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            print(f"[ERROR] Response text: {response_text[:500]}")
            return jsonify({"error": f"Invalid JSON response: {str(e)}"}), 500

        # Format result based on session type (convert flat JSON to nested structure)
        print(f"[INFO] Formatting result for {session_type}...")
        if session_type == "ball_work":
            result = format_ball_work_result(extracted_json)
        elif session_type == "speed_agility":
            result = format_speed_agility_result(extracted_json)
        elif session_type == "match":
            result = format_match_result(extracted_json)
        else:
            return jsonify({"error": "Invalid session type"}), 400

        print("[SUCCESS] Extraction complete!")
        return jsonify({"success": True, "data": result, "ocr_text": response_text})

    except anthropic.AuthenticationError as e:
        print(f"[ERROR] Authentication error: {str(e)}")
        error_msg = "Invalid API key" if API_KEY_MODE == "user" else "Server API key invalid"
        return jsonify({"error": error_msg}), 401
    except anthropic.RateLimitError as e:
        print(f"[ERROR] Rate limit: {str(e)}")
        return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
    except Exception as e:
        print(f"[ERROR] Unexpected: {type(e).__name__}: {str(e)}")
        if FLASK_ENV == "development":
            import traceback

            traceback.print_exc()
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
        else:
            return jsonify({"error": "Processing error occurred"}), 500


def extract_ball_work_data(text):
    """Extract Ball Work session data from OCR text"""
    text = text.lower()

    # Session info
    session_name_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening)[^,]*)",
        text,
        re.IGNORECASE,
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})",
        text,
        re.IGNORECASE,
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), "%B %d, %Y")
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text, r"(\d+)\s*min")
    training_type = extract_value(text, r"(technical|physical|tactical)", default="Technical")
    intensity = extract_value(text, r"(low|moderate|high)", default="Moderate")

    # Highlights
    ball_touches = extract_number(text, r"ball\s*touches[:\s]*(\d+)")
    total_distance = extract_number(text, r"total\s*distance[:\s]*([\d.]+)")
    sprint_distance = extract_number(text, r"sprint\s*distance[:\s]*([\d.]+)")
    accl_decl = extract_number(text, r"accl\s*/\s*decl[:\s]*(\d+)")
    kicking_power = extract_number(text, r"kicking\s*power[:\s]*([\d.]+)")

    # Two-footed - More flexible patterns (with or without parentheses)
    left_touches = extract_number(text, r"left\s*foot[^:]*touch[^:]*[:\s]*(\d+)")
    left_touches_pct = extract_number(
        text, r"left\s*foot[^:]*touch[^:]*\(?(\d+)%\)?"
    ) or extract_number(text, r"(\d+)%.*?left.*?touch")
    right_touches = extract_number(text, r"right\s*foot[^:]*touch[^:]*[:\s]*(\d+)")
    right_touches_pct = extract_number(
        text, r"right\s*foot[^:]*touch[^:]*\(?(\d+)%\)?"
    ) or extract_number(text, r"(\d+)%.*?right.*?touch")

    left_releases = extract_number(text, r"left\s*foot[^:]*release[^:]*[:\s]*(\d+)")
    left_releases_pct = extract_number(
        text, r"left\s*foot[^:]*release[^:]*\(?(\d+)%\)?"
    ) or extract_number(text, r"(\d+)%.*?left.*?release")
    right_releases = extract_number(text, r"right\s*foot[^:]*release[^:]*[:\s]*(\d+)")
    right_releases_pct = extract_number(
        text, r"right\s*foot[^:]*release[^:]*\(?(\d+)%\)?"
    ) or extract_number(text, r"(\d+)%.*?right.*?release")

    left_kicking = extract_number(text, r"left\s*foot\s*kicking\s*power[:\s]*([\d.]+)")
    right_kicking = extract_number(text, r"right\s*foot\s*kicking\s*power[:\s]*([\d.]+)")

    # Speed
    top_speed = extract_number(text, r"top\s*speed[:\s]*([\d.]+)")
    sprints = extract_number(text, r"sprints[:\s]*(\d+)")

    # Agility
    left_turns = extract_number(text, r"left\s*turns[:\s]*(\d+)")
    back_turns = extract_number(text, r"back\s*turns[:\s]*(\d+)")
    right_turns = extract_number(text, r"right\s*turns[:\s]*(\d+)")
    intense_turns = extract_number(text, r"intense\s*turns[:\s]*(\d+)")
    entry_speed = extract_number(text, r"(?:average\s*)?turn\s*entry\s*speed[:\s]*([\d.]+)")
    exit_speed = extract_number(text, r"(?:average\s*)?turn\s*exit\s*speed[:\s]*([\d.]+)")

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": training_type,
            "intensity": intensity,
        },
        "highlights": {
            "ball_touches": ball_touches,
            "total_distance_miles": total_distance,
            "sprint_distance_yards": sprint_distance,
            "accl_decl": accl_decl,
            "kicking_power_mph": kicking_power,
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
            "right_foot_kicking_power_mph": right_kicking,
        },
        "speed": {"top_speed_mph": top_speed, "sprints": sprints},
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "average_turn_entry_speed_mph": entry_speed,
            "average_turn_exit_speed_mph": exit_speed,
        },
    }


def extract_speed_agility_data(text):
    """Extract Speed & Agility session data from OCR text"""
    text = text.lower()

    # Session info
    session_name_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening)[^,]*)",
        text,
        re.IGNORECASE,
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})",
        text,
        re.IGNORECASE,
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), "%B %d, %Y")
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text, r"(\d+)\s*min")
    training_type = extract_value(text, r"(technical|physical|tactical)", default="Physical")
    intensity = extract_value(text, r"(low|moderate|high)", default="Moderate")

    # Highlights
    total_distance = extract_number(text, r"total\s*distance[:\s]*([\d.]+)")
    sprint_distance = extract_number(text, r"sprint\s*distance[:\s]*([\d.]+)")
    accl_decl = extract_number(text, r"accl\s*/\s*decl[:\s]*(\d+)")

    # Speed
    top_speed = extract_number(text, r"top\s*speed[:\s]*([\d.]+)")
    sprints = extract_number(text, r"sprints[:\s]*(\d+)")

    # Agility
    left_turns = extract_number(text, r"left\s*turns[:\s]*(\d+)")
    back_turns = extract_number(text, r"back\s*turns[:\s]*(\d+)")
    right_turns = extract_number(text, r"right\s*turns[:\s]*(\d+)")
    intense_turns = extract_number(text, r"intense\s*turns[:\s]*(\d+)")
    entry_speed = extract_number(text, r"(?:average\s*)?turn\s*entry\s*speed[:\s]*([\d.]+)")
    exit_speed = extract_number(text, r"(?:average\s*)?turn\s*exit\s*speed[:\s]*([\d.]+)")

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": training_type,
            "intensity": intensity,
        },
        "highlights": {
            "total_distance_miles": total_distance,
            "sprint_distance_yards": sprint_distance,
            "accl_decl": accl_decl,
        },
        "speed": {"top_speed_mph": top_speed, "sprints": sprints},
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "average_turn_entry_speed_mph": entry_speed,
            "average_turn_exit_speed_mph": exit_speed,
        },
    }


def extract_match_data(text):
    """Extract Match session data from OCR text"""
    text_lower = text.lower()

    # Session info
    session_name_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\s+(?:morning|afternoon|evening))",
        text_lower,
        re.IGNORECASE,
    )
    session_name = session_name_match.group(1).strip() if session_name_match else None

    date_match = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4})",
        text_lower,
        re.IGNORECASE,
    )
    date_str = None
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), "%B %d, %Y")
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError as e:
            print(f"[WARNING] Date parsing failed for '{date_match.group(1)}': {e}")
            date_str = None

    duration = extract_number(text_lower, r"(\d+)\s*min")

    # Match overview
    position = extract_value(text_lower, r"([a-z]{1,3})\s+position", default=None)
    goals = extract_number(text_lower, r"goals\s+(\d+)")
    assists = extract_number(text_lower, r"assists\s+(\d+)")

    # Team scores - looking for pattern like "cityplay fc 1 : 4 fc westlake"
    score_match = re.search(r"(\d+)\s*:\s*(\d+)", text_lower)
    if score_match:
        athlete_score = int(score_match.group(1))
        opposing_score = int(score_match.group(2))
    else:
        athlete_score = None
        opposing_score = None

    # Opponent name - text after the score pattern
    opponent_match = re.search(r"\d+\s*:\s*\d+\s+(.+?)(?:\n|$)", text_lower)
    opposing_team_name = opponent_match.group(1).strip() if opponent_match else None

    # Skills scores - simpler patterns matching "two-footed 55" format
    two_footed = extract_number(text_lower, r"two-?footed\s+(\d+)")
    dribbling = extract_number(text_lower, r"dribbling\s+(\d+)")
    first_touch = extract_number(text_lower, r"first\s+touch\s+(\d+)")
    agility_score = extract_number(text_lower, r"agility\s+(\d+)")
    speed_score = extract_number(text_lower, r"speed\s+(\d+)")
    power_score = extract_number(text_lower, r"power\s+(\d+)")

    # Highlights - simpler patterns matching actual format
    work_rate = extract_number(text_lower, r"work\s+rate\s+([\d.]+)")
    ball_possessions = extract_number(text_lower, r"ball\s+possessions\s+(\d+)")
    total_distance = extract_number(text_lower, r"total\s+distance\s+([\d.]+)")
    sprint_distance = extract_number(text_lower, r"sprint\s+distance\s+([\d.]+)")
    top_speed = extract_number(text_lower, r"top\s+speed\s+([\d.]+)")
    kicking_power = extract_number(text_lower, r"kicking\s+power\s+([\d.]+)")

    # Two-footed - More flexible patterns (with or without parentheses)
    left_touches = extract_number(text_lower, r"left\s+foot[^:]*touch[^:]*[:\s]*(\d+)")
    left_touches_pct = extract_number(
        text_lower, r"left\s+foot[^:]*touch[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?left.*?touch")
    right_touches = extract_number(text_lower, r"right\s+foot[^:]*touch[^:]*[:\s]*(\d+)")
    right_touches_pct = extract_number(
        text_lower, r"right\s+foot[^:]*touch[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?right.*?touch")

    left_releases = extract_number(text_lower, r"left\s+foot[^:]*release[^:]*[:\s]*(\d+)")
    left_releases_pct = extract_number(
        text_lower, r"left\s+foot[^:]*release[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?left.*?release")
    right_releases = extract_number(text_lower, r"right\s+foot[^:]*release[^:]*[:\s]*(\d+)")
    right_releases_pct = extract_number(
        text_lower, r"right\s+foot[^:]*release[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?right.*?release")

    left_receives = extract_number(text_lower, r"left\s+foot[^:]*receive[^:]*[:\s]*(\d+)")
    left_receives_pct = extract_number(
        text_lower, r"left\s+foot[^:]*receive[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?left.*?receive")
    right_receives = extract_number(text_lower, r"right\s+foot[^:]*receive[^:]*[:\s]*(\d+)")
    right_receives_pct = extract_number(
        text_lower, r"right\s+foot[^:]*receive[^:]*\(?(\d+)%\)?"
    ) or extract_number(text_lower, r"(\d+)%.*?right.*?receive")

    left_kicking = extract_number(text_lower, r"left\s+foot\s*kicking\s*power[:\s]*(\d+\.?\d*)")
    right_kicking = extract_number(text_lower, r"right\s+foot\s*kicking\s*power[:\s]*(\d+\.?\d*)")

    # Dribbling
    distance_with_ball = extract_number(text_lower, r"distance\s+with\s+ball[:\s]*(\d+\.?\d*)\s*yd")
    top_speed_with_ball = extract_number(
        text_lower, r"top\s+speed\s+with\s+ball[:\s]*(\d+\.?\d*)\s*mph"
    )
    intense_turns_with_ball = extract_number(
        text_lower, r"intense\s+turns\s+with\s+ball[:\s]*(\d+)"
    )

    # First touch - possessions
    one_touch_poss = extract_number(text_lower, r"one[- ]touch[:\s]*(\d+)")
    multiple_touch_poss = extract_number(text_lower, r"multiple[- ]touch[:\s]*(\d+)")
    total_duration_sec = extract_number(text_lower, r"total\s+duration[:\s]*(\d+\.?\d*)\s*s")

    # First touch - ball release footzone
    laces = extract_number(text_lower, r"laces[:\s]*(\d+)")
    inside = extract_number(text_lower, r"inside[:\s]*(\d+)")
    other_footzone = extract_number(text_lower, r"other[:\s]*(\d+)")

    # Agility
    left_turns = extract_number(
        text_lower,
        r"(\d+)[^\d]*\d+[^\d]*\d+[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?",
    )
    back_turns = extract_number(
        text_lower,
        r"\d+[^\d]*(\d+)[^\d]*\d+[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?",
    )
    right_turns = extract_number(
        text_lower,
        r"\d+[^\d]*\d+[^\d]*(\d+)[^\d]*left\s+turns?[^\d]*back\s+turns?[^\d]*right\s+turns?",
    )
    intense_turns = extract_number(text_lower, r"intense\s+turns?(?!\s+with\s+ball)\s*[:\s#]*(\d+)")
    entry_speed = extract_number(
        text_lower, r"(?:average\s*)?(?:turn|tum)\s+entry\s+speed[:\s]*(\d+\.?\d*)"
    )
    exit_speed = extract_number(
        text_lower, r"(?:average\s*)?(?:turn|tum)\s+exit\s+speed[:\s]*(\d+\.?\d*)"
    )

    # Speed
    sprints = extract_number(text_lower, r"sprints?\s*[:\s#]*(\d+)")

    # Power
    first_step_accel = extract_number(text_lower, r"first[- ]step[:\s]*(\d+)")
    intense_accel = extract_number(text_lower, r"intense\s+(?:accel|acceleration)[:\s]*(\d+)")

    return {
        "session": {
            "session_name": session_name,
            "date": date_str,
            "duration_minutes": duration,
            "training_type": "Match",
        },
        "overview": {
            "position": position,
            "goals": goals,
            "assists": assists,
            "athlete_team_score": athlete_score,
            "opposing_team_score": opposing_score,
            "opposing_team_name": opposing_team_name,
        },
        "skills": {
            "two_footed_score": two_footed,
            "dribbling_score": dribbling,
            "first_touch_score": first_touch,
            "agility_score": agility_score,
            "speed_score": speed_score,
            "power_score": power_score,
        },
        "highlights": {
            "work_rate_yd_per_min": work_rate,
            "ball_possessions": ball_possessions,
            "total_distance_mi": total_distance,
            "sprint_distance_yd": sprint_distance,
            "top_speed_mph": top_speed,
            "kicking_power_mph": kicking_power,
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
            "right_foot_kicking_power_mph": right_kicking,
        },
        "dribbling": {
            "distance_with_ball_yd": distance_with_ball,
            "top_speed_with_ball_mph": top_speed_with_ball,
            "intense_turns_with_ball": intense_turns_with_ball,
        },
        "first_touch": {
            "ball_possessions": {
                "total": ball_possessions,
                "one_touch": one_touch_poss,
                "multiple_touch": multiple_touch_poss,
                "total_duration_sec": total_duration_sec,
            },
            "ball_release_footzone": {"laces": laces, "inside": inside, "other": other_footzone},
        },
        "agility": {
            "left_turns": left_turns,
            "back_turns": back_turns,
            "right_turns": right_turns,
            "intense_turns": intense_turns,
            "avg_turn_entry_speed_mph": entry_speed,
            "avg_turn_exit_speed_mph": exit_speed,
        },
        "speed": {"top_speed_mph": top_speed, "sprints": sprints},
        "power": {
            "first_step_accelerations": first_step_accel,
            "intense_accelerations": intense_accel,
        },
    }


def extract_number(text, pattern):
    """Extract a number from text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) if "." in match.group(1) else int(match.group(1))
        except ValueError:
            return None
    return None


def extract_value(text, pattern, default=None):
    """Extract a text value from text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return default


def format_match_result(data):
    """Format flat JSON into nested Match structure"""
    return {
        "session": {
            "session_name": data.get("session_name"),
            "date": data.get("date"),
            "duration_minutes": data.get("duration_minutes"),
            "training_type": "Match",
        },
        "overview": {
            "position": data.get("position"),
            "goals": data.get("goals"),
            "assists": data.get("assists"),
            "athlete_team_score": data.get("athlete_team_score"),
            "opposing_team_score": data.get("opposing_team_score"),
            "opposing_team_name": data.get("opposing_team_name"),
        },
        "skills": {
            "two_footed_score": data.get("two_footed_score"),
            "dribbling_score": data.get("dribbling_score"),
            "first_touch_score": data.get("first_touch_score"),
            "agility_score": data.get("agility_score"),
            "speed_score": data.get("speed_score"),
            "power_score": data.get("power_score"),
        },
        "highlights": {
            "work_rate_yd_per_min": data.get("work_rate"),
            "ball_possessions": data.get("ball_possessions"),
            "total_distance_mi": data.get("total_distance"),
            "sprint_distance_yd": data.get("sprint_distance"),
            "top_speed_mph": data.get("top_speed"),
            "kicking_power_mph": data.get("kicking_power"),
        },
        "two_footed": {
            "left_foot_touches": data.get("left_touches"),
            "left_foot_touches_pct": data.get("left_touches_pct"),
            "right_foot_touches": data.get("right_touches"),
            "right_foot_touches_pct": data.get("right_touches_pct"),
            "left_foot_releases": data.get("left_releases"),
            "left_foot_releases_pct": data.get("left_releases_pct"),
            "right_foot_releases": data.get("right_releases"),
            "right_foot_releases_pct": data.get("right_releases_pct"),
            "left_foot_receives": data.get("left_receives"),
            "left_foot_receives_pct": data.get("left_receives_pct"),
            "right_foot_receives": data.get("right_receives"),
            "right_foot_receives_pct": data.get("right_receives_pct"),
            "left_foot_kicking_power_mph": data.get("left_kicking_power"),
            "right_foot_kicking_power_mph": data.get("right_kicking_power"),
        },
        "dribbling": {
            "distance_with_ball_yd": data.get("distance_with_ball"),
            "top_speed_with_ball_mph": data.get("top_speed_with_ball"),
            "intense_turns_with_ball": data.get("intense_turns_with_ball"),
        },
        "first_touch": {
            "ball_possessions": {
                "total": data.get("ball_possessions"),
                "one_touch": data.get("one_touch_poss"),
                "multiple_touch": data.get("multiple_touch_poss"),
                "total_duration_sec": data.get("total_duration_sec"),
            },
            "ball_release_footzone": {
                "laces": data.get("laces"),
                "inside": data.get("inside"),
                "other": data.get("other_footzone"),
            },
        },
        "agility": {
            "left_turns": data.get("left_turns"),
            "back_turns": data.get("back_turns"),
            "right_turns": data.get("right_turns"),
            "intense_turns": data.get("intense_turns"),
            "avg_turn_entry_speed_mph": data.get("avg_turn_entry"),
            "avg_turn_exit_speed_mph": data.get("avg_turn_exit"),
        },
        "speed": {"top_speed_mph": data.get("top_speed"), "sprints": data.get("num_sprints")},
        "power": {
            "first_step_accelerations": data.get("first_step_accel"),
            "intense_accelerations": data.get("intense_accel"),
        },
    }


def format_ball_work_result(data):
    """Format flat JSON into nested Ball Work structure"""
    return {
        "session": {
            "session_name": data.get("session_name"),
            "date": data.get("date"),
            "duration_minutes": data.get("duration_minutes"),
            "training_type": data.get("training_type"),
            "intensity": data.get("intensity"),
        },
        "highlights": {
            "ball_touches": data.get("ball_touches"),
            "total_distance_miles": data.get("total_distance"),
            "sprint_distance_yards": data.get("sprint_distance"),
            "accl_decl": data.get("accelerations"),
            "kicking_power_mph": data.get("kicking_power"),
        },
        "two_footed": {
            "left_foot_touches": data.get("left_touches"),
            "left_foot_touches_percentage": data.get("left_pct"),
            "right_foot_touches": data.get("right_touches"),
            "right_foot_touches_percentage": data.get("right_pct"),
            "left_foot_releases": data.get("left_releases"),
            "left_foot_releases_percentage": data.get("left_release_pct"),
            "right_foot_releases": data.get("right_releases"),
            "right_foot_releases_percentage": data.get("right_release_pct"),
            "left_foot_kicking_power_mph": data.get("left_kicking_power"),
            "right_foot_kicking_power_mph": data.get("right_kicking_power"),
        },
        "speed": {"top_speed_mph": data.get("top_speed"), "sprints": data.get("num_sprints")},
        "agility": {
            "left_turns": data.get("left_turns"),
            "back_turns": data.get("back_turns"),
            "right_turns": data.get("right_turns"),
            "intense_turns": data.get("intense_turns"),
            "average_turn_entry_speed_mph": data.get("avg_turn_entry"),
            "average_turn_exit_speed_mph": data.get("avg_turn_exit"),
        },
    }


def format_speed_agility_result(data):
    """Format flat JSON into nested Speed & Agility structure"""
    return {
        "session": {
            "session_name": data.get("session_name"),
            "date": data.get("date"),
            "duration_minutes": data.get("duration_minutes"),
            "training_type": data.get("training_type"),
            "intensity": data.get("intensity"),
        },
        "highlights": {
            "total_distance_miles": data.get("total_distance"),
            "sprint_distance_yards": data.get("sprint_distance"),
            "accl_decl": data.get("accelerations"),
        },
        "speed": {"top_speed_mph": data.get("top_speed"), "sprints": data.get("num_sprints")},
        "agility": {
            "left_turns": data.get("left_turns"),
            "back_turns": data.get("back_turns"),
            "right_turns": data.get("right_turns"),
            "intense_turns": data.get("intense_turns"),
            "average_turn_entry_speed_mph": data.get("avg_turn_entry"),
            "average_turn_exit_speed_mph": data.get("avg_turn_exit"),
        },
    }


if __name__ == "__main__":
    print("Starting Soccer OCR Server...")
    print(f"Mode: {API_KEY_MODE.upper()}")
    print(f"Environment: {FLASK_ENV}")
    if FLASK_ENV == "production":
        print("WARNING: Use gunicorn for production!")

    port = int(os.environ.get("PORT", 5000))
    debug = FLASK_ENV == "development"
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=debug)
