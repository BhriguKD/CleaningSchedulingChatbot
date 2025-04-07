import os
import json
import traceback
import datetime
from datetime import datetime, timedelta
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Flask app
app = Flask(__name__)

# Data storage (in a real app, use a database)
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
USER_DATA_FILE = DATA_DIR / 'user_data.json'
CONVERSATION_FILE = DATA_DIR / 'conversations.json'


def load_user_data():
    try:
        if USER_DATA_FILE.exists():
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        return {"users": {}}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading user data: {e}")
        return {"users": {}}


def save_user_data(data):
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Error saving user data: {e}")
        return False


def load_conversations():
    try:
        if CONVERSATION_FILE.exists():
            with open(CONVERSATION_FILE, 'r') as f:
                return json.load(f)
        return {"conversations": {}}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading conversations: {e}")
        return {"conversations": {}}


def save_conversations(data):
    try:
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Error saving conversations: {e}")
        return False


def get_conversation_history(user_id, max_history=10):
    """Get conversation history for a user"""
    conversations = load_conversations()
    if user_id not in conversations.get("conversations", {}):
        conversations["conversations"][user_id] = []

    user_conversations = conversations["conversations"][user_id]
    return user_conversations[-max_history:] if user_conversations else []


def add_to_conversation(user_id, message, is_user=True):
    """Add a message to the conversation history"""
    conversations = load_conversations()
    if user_id not in conversations.get("conversations", {}):
        conversations["conversations"][user_id] = []

    conversations["conversations"][user_id].append({
        "role": "user" if is_user else "model",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })

    save_conversations(conversations)


# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')


def get_gemini_response(user_id, prompt):
    """Get a response from the Gemini model with conversation history context"""
    # Get conversation history
    conversation_history = get_conversation_history(user_id)

    # Format conversation history for Gemini
    formatted_history = ""
    if conversation_history:
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"

    # Create context-aware prompt
    context_prompt = f"""
    You are a cleaning scheduling assistant. Your main capabilities include:
    - Scheduling cleaning tasks
    - Showing cleaning schedules
    - Marking tasks as complete
    
    Previous conversation:
    {formatted_history}
    
    The user's latest message: "{prompt}"
    
    Respond concisely (1-2 sentences). If they're trying to schedule something 
    but didn't specify when, ask when they would like to schedule it.
    """

    try:
        response = model.generate_content(context_prompt)
        # Add this interaction to the conversation history
        add_to_conversation(user_id, prompt, is_user=True)
        add_to_conversation(user_id, response.text, is_user=False)
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return "I'm having trouble connecting right now. Could you please try again?"


def extract_dates_from_text(text):
    """Simple date extraction logic - in a real app, use a more robust parser"""
    dates = []
    # This is a simplified version - would need more comprehensive parsing
    date_keywords = ['tomorrow', 'next week', 'on monday', 'tuesday', 'wednesday',
                     'thursday', 'friday', 'saturday', 'sunday']

    text_lower = text.lower()
    today = datetime.now()

    # Simple keyword matching
    if 'tomorrow' in text_lower:
        dates.append((today + timedelta(days=1)).strftime('%Y-%m-%d'))
    elif 'next week' in text_lower:
        dates.append((today + timedelta(days=7)).strftime('%Y-%m-%d'))

    # Day matching
    days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6}

    for day, day_num in days.items():
        if day in text_lower:
            # Calculate days until next occurrence of this day
            current_day_num = today.weekday()
            days_ahead = day_num - current_day_num
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            dates.append(target_date.strftime('%Y-%m-%d'))
            
    # Look for ISO format dates (YYYY-MM-DD)
    import re
    iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    if iso_dates:
        dates.extend(iso_dates)

    return dates


def add_schedule(user_id, task, date):
    data = load_user_data()

    if user_id not in data['users']:
        data['users'][user_id] = {"schedules": []}

    # Add a unique ID for each task
    task_id = len(data['users'][user_id]['schedules']) + 1

    data['users'][user_id]['schedules'].append({
        "id": task_id,
        "task": task,
        "date": date,
        "created_at": datetime.now().isoformat(),
        "completed": False
    })

    save_user_data(data)
    return task_id


def get_schedules(user_id):
    data = load_user_data()
    if user_id in data['users']:
        return data['users'][user_id]['schedules']
    return []


def format_schedules_for_response(schedules):
    """Format schedules for consistent API response"""
    return [
        {
            "id": s.get('id', idx+1),
            "task": s['task'],
            "date": s['date'],
            "completed": s['completed']
        }
        for idx, s in enumerate(schedules)
    ]


def mark_completed(user_id, task_id):
    data = load_user_data()
    if user_id in data['users']:
        for task in data['users'][user_id]['schedules']:
            if task.get('id') == task_id:
                task['completed'] = True
                task['completed_at'] = datetime.now().isoformat()
                save_user_data(data)
                return True
    return False


def extract_schedule_from_message(user_id, message):
    # Include conversation context in the prompt
    conversation_history = get_conversation_history(user_id)
    formatted_history = ""
    if conversation_history:
        # Use last 5 messages for context
        for msg in conversation_history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"

    prompt = f"""
    Previous conversation:
    {formatted_history}
    
    Extract cleaning task information from this message: "{message}"
    
    Respond in JSON format with these fields:
    - has_task: boolean
    - task: string (cleaning task description)
    - is_scheduling: boolean
    - date: string (YYYY-MM-DD format if mentioned)
    
    Example:
    {{
      "has_task": true,
      "task": "deep clean kitchen",
      "is_scheduling": true,
      "date": "2025-04-15"
    }}
    """

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing response: {e}")
        return {"has_task": False}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/schedules', methods=['GET'])
def get_user_schedules():
    """New endpoint to get user schedules"""
    user_id = request.args.get('user_id', 'default_user')
    schedules = get_schedules(user_id)
    
    # Make sure schedules maintain their original IDs
    formatted_schedules = format_schedules_for_response(schedules)
    
    return jsonify({
        "schedules": formatted_schedules
    })


@app.route('/api/message', methods=['POST'])
def process_message():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    message = data.get('message', '').strip()

    # First, check if we need to handle special commands
    message_lower = message.lower()

    # Check for schedule requests
    if 'show my schedule' in message_lower or 'my tasks' in message_lower:
        schedules = get_schedules(user_id)
        if not schedules:
            response = "You don't have any scheduled cleanings yet."
            add_to_conversation(user_id, message, is_user=True)
            add_to_conversation(user_id, response, is_user=False)
            return jsonify({
                "response": response,
                "schedules": []
            })

        # Format schedules for consistency
        formatted = format_schedules_for_response(schedules)

        response = "Here are your scheduled cleanings:"
        add_to_conversation(user_id, message, is_user=True)
        add_to_conversation(user_id, response, is_user=False)
        return jsonify({
            "response": response,
            "schedules": formatted
        })

    # Handle task completion
    if 'mark complete' in message_lower or 'done' in message_lower or 'mark as complete' in message_lower:
        try:
            # Extract task ID from message
            task_id = int(''.join(filter(str.isdigit, message)))
            if mark_completed(user_id, task_id):
                response = f"Marked task {task_id} as complete!"
                add_to_conversation(user_id, message, is_user=True)
                add_to_conversation(user_id, response, is_user=False)
                
                # Get updated schedules to return
                updated_schedules = get_schedules(user_id)
                formatted_schedules = format_schedules_for_response(updated_schedules)
                
                return jsonify({
                    "response": response,
                    "success": True,
                    "schedules": formatted_schedules  # Return updated schedules
                })
        except (ValueError, IndexError):
            pass

        response = "Couldn't find that task number. Please check your schedule."
        add_to_conversation(user_id, message, is_user=True)
        add_to_conversation(user_id, response, is_user=False)
        return jsonify({
            "response": response,
            "success": False
        })

    # Process new scheduling requests
    task_info = extract_schedule_from_message(user_id, message)
    if task_info.get('has_task') and task_info.get('is_scheduling'):
        task = task_info.get('task', 'Cleaning task')
        date = task_info.get('date') or extract_dates_from_text(message)

        if date:
            date_to_use = date[0] if isinstance(date, list) else date
            task_id = add_schedule(user_id, task, date_to_use)
            response = f"Scheduled: '{task}' for {date_to_use}. Anything else?"
            add_to_conversation(user_id, message, is_user=True)
            add_to_conversation(user_id, response, is_user=False)
            
            # Return success with the new task info but without detailed schedule data
            return jsonify({
                "response": response,
                "success": True,
                "new_task": {
                    "id": task_id,
                    "task": task,
                    "date": date_to_use
                }
            })

        response = "When would you like to schedule this cleaning?"
        add_to_conversation(user_id, message, is_user=True)
        add_to_conversation(user_id, response, is_user=False)
        return jsonify({
            "response": response,
            "needs_date": True
        })

    # Default response using Gemini with context
    response = get_gemini_response(user_id, message)
    return jsonify({"response": response})


@app.route('/api/export', methods=['GET'])
def export_data():
    user_id = request.args.get('user_id', 'default_user')
    data = load_user_data()
    conversations = load_conversations()

    # Get user's schedules
    schedules = data['users'].get(user_id, {}).get('schedules', [])

    # Get user's conversation history
    conversation_history = conversations.get(
        'conversations', {}).get(user_id, [])

    # Prepare export data
    export_data = {
        "user_id": user_id,
        "schedules": schedules,
        "conversations": conversation_history,
        "exported_at": datetime.now().isoformat()
    }

    return jsonify(export_data)


@app.route('/test_gemini')
def test_gemini():
    try:
        response = get_gemini_response("default_user", "Say hello")
        return f"Gemini API test successful. Response: {response}"
    except Exception as e:
        return f"Gemini API test failed: {str(e)}\n{traceback.format_exc()}"


# Run the Flask app
if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Ensure we have empty data files if they don't exist
    if not USER_DATA_FILE.exists():
        with open(USER_DATA_FILE, 'w') as f:
            json.dump({"users": {}}, f)

    if not CONVERSATION_FILE.exists():
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump({"conversations": {}}, f)

    app.run(debug=True)