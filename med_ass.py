import re
import os
import time
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from num2words import num2words
from typing import Dict, Optional, List
from pymongo import MongoClient
from bson import ObjectId


# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_VERSION  = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")
SPEECH_KEY      = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION   = os.getenv("AZURE_SPEECH_REGION")

# MongoDB Configuration
MONGODB_URI     = os.getenv("MONGODB_URI")
DATABASE_NAME   = os.getenv("DATABASE_NAME", "medical_device_db")

assert all([OPENAI_KEY, OPENAI_ENDPOINT, OPENAI_DEPLOY, OPENAI_VERSION]), "OpenAI vars missing"
assert all([SPEECH_KEY, SPEECH_REGION]), "Speech vars missing"
assert MONGODB_URI, "MongoDB URI missing"

# 2. INITIALIZE AZURE OPENAI AND SPEECH CLIENTS
client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)
speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

print("✔ Medical Device Assistant initialized")

# Global variable to store user's preferred language
user_language = "English"

# 3. MONGODB SETUP
def init_mongodb():
    """Initialize MongoDB connection and collections"""
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client[DATABASE_NAME]
        
        # Test connection
        mongo_client.admin.command('ping')
        print("✔ MongoDB Atlas connected successfully")
        
        # Get collections
        users_collection = db.user_profiles
        readings_collection = db.medical_readings
        
        # Create indexes for better performance
        users_collection.create_index([("fullname", 1)], unique=True, sparse=True)
        readings_collection.create_index([("user_id", 1), ("reading_date", -1)])
        
        return db, users_collection, readings_collection
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

# Initialize MongoDB
db, users_collection, readings_collection = init_mongodb()

# 4. SMART INFORMATION EXTRACTION USING AI
def extract_information_with_ai(user_input: str, info_type: str, context: str = "") -> Dict:
    """Use AI to extract specific information from natural speech"""
    try:
        extraction_prompt = f"""
        You are a medical assistant helping to extract specific information from patient responses.
        
        Patient said: "{user_input}"
        
        I need to extract: {info_type}
        Context: {context}
        
        Please analyze what the patient said and extract ONLY the specific information requested.
        
        Instructions based on information type:
        
        If extracting NAME:
        - Extract only the actual name (first name and last name)
        - Ignore phrases like "my name is", "I am", "call me", etc.
        - Example: "My name is John Smith" → extract "John Smith"
        - Example: "I am called Sarah Wilson" → extract "Sarah Wilson"
        
        If extracting AGE:
        - Extract only the numerical age
        - Example: "I am 25 years old" → extract "25"
        - Example: "My age is thirty-five" → extract "35"
        
        If extracting SYMPTOMS:
        - Extract the actual symptoms/problems described
        - Remove casual phrases and focus on medical concerns
        - Example: "I have been feeling dizzy and having headaches" → extract "dizzy, headaches"
        
        If extracting MEDICAL_VALUE (temperature, blood pressure, etc.):
        - Extract only the numerical value(s)
        - Example: "My temperature is 98.6 degrees" → extract "98.6"
        - Example: "Blood pressure is 120 over 80" → extract "120/80"
        
        If extracting YES_NO_RESPONSE:
        - Determine if the response indicates yes/no/unclear
        - Example: "Yes, that's correct" → extract "yes"
        - Example: "No, that's wrong" → extract "no"
        
        If extracting GENDER:
        - Extract the gender mentioned
        - Example: "I am a female" → extract "female"
        
        If extracting HEIGHT/WEIGHT:
        - Extract the numerical value and unit if mentioned
        - Example: "I am 5 feet 6 inches tall" → extract "5'6\""
        - Example: "My weight is 70 kilograms" → extract "70 kg"
        
        Return your response in this exact JSON format:
        {{
            "extracted_value": "the extracted information",
            "confidence": "high/medium/low",
            "needs_clarification": true/false,
            "clarification_question": "question to ask if needs clarification"
        }}
        
        If you cannot extract the information clearly, set needs_clarification to true and provide a helpful question.
        """
        
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": "You are an expert at extracting specific information from natural language. Always respond with valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            extracted_info = json.loads(result)
            return extracted_info
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "extracted_value": user_input,
                "confidence": "low",
                "needs_clarification": True,
                "clarification_question": f"I didn't understand clearly. Could you please repeat your {info_type}?"
            }
            
    except Exception as e:
        print(f"Error in AI extraction: {e}")
        return {
            "extracted_value": user_input,
            "confidence": "low",
            "needs_clarification": True,
            "clarification_question": f"I had trouble understanding. Could you please tell me your {info_type} again?"
        }

# 5. LANGUAGE SELECTION AT START
def select_user_language() -> str:
    """Ask user to select their preferred language at the beginning"""
    global user_language
    
    # Start with English greeting and language options
    language_options = """
    Hello! Welcome to your Medical Assistant. 
    To provide you with the best care, please tell me which language you'd like to communicate in:
    
    English - Say "English"
    हिंदी - Say "Hindi" 
    मराठी - Say "Marathi"
    தமிழ் - Say "Tamil"
    తెలుగు - Say "Telugu"
    ગુજરાતી - Say "Gujarati"
    বাংলা - Say "Bengali"
    ਪੰਜਾਬੀ - Say "Punjabi"
    ಕನ್ನಡ - Say "Kannada"
    മലയാളം - Say "Malayalam"
    اردو - Say "Urdu"
    
    Or simply speak in your preferred language and I'll understand.
    """
    
    print(f"🌐 {language_options}")
    speak(language_options)
    
    max_attempts = 3
    for attempt in range(max_attempts):
        print("🎙️ Listening for your language preference...")
        language_response = listen("your language preference")
        
        if language_response:
            # Use AI to determine the language
            detected_lang = detect_language_with_ai(language_response)
            
            if detected_lang != "unclear":
                user_language = detected_lang
                confirmation_msg = f"Perfect! I will communicate with you in {user_language}."
                
                # Translate confirmation to selected language
                translated_confirmation = translate_to_user_language(confirmation_msg)
                print(f"✅ {translated_confirmation}")
                speak(translated_confirmation)
                return user_language
            else:
                if attempt < max_attempts - 1:
                    retry_msg = "I didn't catch that clearly. Please say which language you prefer, like 'English', 'Hindi', or just speak in your language."
                    print(f"🔄 {retry_msg}")
                    speak(retry_msg)
        else:
            if attempt < max_attempts - 1:
                patience_msg = "No problem, let me try again. Please tell me your preferred language."
                print(f"🔄 {patience_msg}")
                speak(patience_msg)
    
    # Default to English if no clear response
    user_language = "English"
    default_msg = "I'll continue in English. We can change the language anytime during our conversation."
    print(f"🌐 {default_msg}")
    speak(default_msg)
    return user_language

def detect_language_with_ai(user_input: str) -> str:
    """Use AI to detect and classify the language with better accuracy"""
    try:
        detection_prompt = f"""
        Analyze this text and determine the language: "{user_input}"
        
        The user is selecting their preferred language for a medical consultation.
        They might say the language name in English (like "Hindi", "Tamil") or speak in their native language.
        
        Common languages and their indicators:
        - English: "English", "english", or English words
        - Hindi: "Hindi", "हिंदी", or Hindi words/script
        - Marathi: "Marathi", "मराठी", or Marathi words/script  
        - Tamil: "Tamil", "தமிழ்", or Tamil words/script
        - Telugu: "Telugu", "తెలుగు", or Telugu words/script
        - Gujarati: "Gujarati", "ગુજરાતી", or Gujarati words/script
        - Bengali: "Bengali", "বাংলা", or Bengali words/script
        - Punjabi: "Punjabi", "ਪੰਜਾਬੀ", or Punjabi words/script
        - Kannada: "Kannada", "ಕನ್ನಡ", or Kannada words/script
        - Malayalam: "Malayalam", "മലയാളം", or Malayalam words/script
        - Urdu: "Urdu", "اردو", or Urdu words/script
        
        Respond with just the language name in English (like "Hindi", "English", "Tamil", etc.).
        If unclear, respond with "unclear".
        """
        
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": "You are a language detection expert. Respond only with the language name in English or 'unclear'."},
                {"role": "user", "content": detection_prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        detected_language = response.choices[0].message.content.strip()
        print(f"🔍 Language detected: {detected_language}")
        return detected_language
        
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unclear"

# 6. ENHANCED TRANSLATION FUNCTIONS
def translate_to_user_language(text: str) -> str:
    """Translate text to user's preferred language with medical context"""
    global user_language
    
    if user_language.lower() in ['english', 'en']:
        return text
    
    try:
        translation_prompt = f"""
        You are a medical translator. Translate the following medical assistant text to {user_language}.
        
        Important guidelines:
        - Keep it natural, caring, and appropriate for a medical context
        - Maintain the same tone and meaning
        - Use respectful and formal language appropriate for healthcare
        - Keep medical terms accurate
        - Make it sound like a caring doctor speaking
        
        Text to translate: "{text}"
        
        Respond only with the translation in {user_language}, nothing else.
        """
        
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": f"You are a professional medical translator specializing in {user_language}. Provide accurate, respectful medical translations."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text
        
    except Exception as e:
        print(f"Error translating to {user_language}: {e}")
        return text

def translate_from_user_language(user_input: str) -> str:
    """Translate user input to English for processing"""
    global user_language
    
    if user_language.lower() in ['english', 'en']:
        return user_input
    
    try:
        translation_prompt = f"""
        Translate the following medical consultation text from {user_language} to English.
        This is from a patient consultation, so maintain medical accuracy and context.
        
        Text to translate: "{user_input}"
        
        Respond only with the English translation, nothing else.
        """
        
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": f"You are a medical translator. Translate from {user_language} to English while maintaining medical accuracy."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text
        
    except Exception as e:
        print(f"Error translating from {user_language}: {e}")
        return user_input

# 7. ENHANCED SPEECH FUNCTIONS
LOG_FILE = "medical_device.log"
def log_conversation(role: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role}: {message}\n")

def tts(text: str):
    speechsdk.SpeechSynthesizer(
        speech_config=speech_cfg,
        audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    ).speak_text_async(text).get()

def clean_for_speech(text: str) -> str:
    text = re.sub(r"【.*?】", "", text)
    return text

def speak(text: str):
    clean = clean_for_speech(text)
    tts(clean)

def listen(listening_for: str = "your response") -> str | None:
    recog = speechsdk.SpeechRecognizer(speech_config=speech_cfg)
    print(f"🎙️ Listening for: {listening_for}...")
    print("💬 Please speak clearly (say 'exit' to quit)")
    res = recog.recognize_once_async().get()
    if res.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_said = res.text.strip()
        print(f"👤 You said: '{user_said}'")
        return user_said
    print("⚠️ I didn't hear anything clearly")
    return None

# 8. SMART INFORMATION COLLECTION WITH AI
def collect_information_smartly(info_type: str, prompt: str, context: str = "", max_attempts: int = 3) -> str | None:
    """Collect information using AI to extract from natural speech"""
    
    for attempt in range(max_attempts):
        if attempt == 0:
            question = translate_to_user_language(prompt)
        else:
            retry_msg = f"Let me try again. {prompt}"
            question = translate_to_user_language(retry_msg)
        
        print(f"🩺 {question}")
        speak(question)
        
        user_response = listen(f"your {info_type}")
        
        if not user_response:
            if attempt < max_attempts - 1:
                patience_msg = "I'm having trouble hearing you. Let me listen more carefully."
                speak(translate_to_user_language(patience_msg))
                continue
            else:
                return None
        
        # Check for exit commands
        if any(exit_word in user_response.lower() for exit_word in ["exit", "quit", "bye", "stop", "goodbye"]):
            return "EXIT_COMMAND"
        
        # Translate to English for processing
        english_response = translate_from_user_language(user_response)
        
        # Use AI to extract information
        extraction_result = extract_information_with_ai(english_response, info_type, context)
        
        if not extraction_result["needs_clarification"] and extraction_result["confidence"] in ["high", "medium"]:
            # Confirm with user
            extracted_value = extraction_result["extracted_value"]
            confirmation_prompt = f"I understood that your {info_type} is: {extracted_value}. Is this correct?"
            
            confirmation_response = get_confirmation(confirmation_prompt)
            
            if confirmation_response == "EXIT_COMMAND":
                return "EXIT_COMMAND"
            elif confirmation_response == "yes":
                return extracted_value
            elif confirmation_response == "no":
                continue  # Ask again
        else:
            # Need clarification
            if extraction_result.get("clarification_question"):
                clarification = translate_to_user_language(extraction_result["clarification_question"])
                print(f"🤔 {clarification}")
                speak(clarification)
    
    return None

def get_confirmation(confirmation_prompt: str) -> str:
    """Get yes/no confirmation from user"""
    translated_prompt = translate_to_user_language(confirmation_prompt)
    print(f"🩺 {translated_prompt}")
    speak(translated_prompt)
    
    max_attempts = 2
    for attempt in range(max_attempts):
        response = listen("yes or no")
        
        if response:
            # Check for exit
            if any(exit_word in response.lower() for exit_word in ["exit", "quit", "bye", "stop", "goodbye"]):
                return "EXIT_COMMAND"
            
            # Translate and extract yes/no
            english_response = translate_from_user_language(response)
            confirmation_result = extract_information_with_ai(english_response, "YES_NO_RESPONSE")
            
            extracted_answer = confirmation_result["extracted_value"].lower()
            if extracted_answer in ["yes", "yeah", "correct", "right", "true"]:
                return "yes"
            elif extracted_answer in ["no", "wrong", "incorrect", "false"]:
                return "no"
        
        if attempt < max_attempts - 1:
            clarify_msg = "Please say 'yes' if correct or 'no' if incorrect."
            speak(translate_to_user_language(clarify_msg))
    
    # Default to yes if unclear
    assume_msg = "I'll assume that's correct."
    speak(translate_to_user_language(assume_msg))
    return "yes"

# 9. USER PROFILE MANAGEMENT (Enhanced)
def check_user_exists(fullname: str) -> Optional[Dict]:
    """Check if user profile exists in MongoDB"""
    try:
        user = users_collection.find_one(
            {"fullname": {"$regex": f"^{re.escape(fullname)}$", "$options": "i"}}
        )
        if user:
            user['_id'] = str(user['_id'])
            return user
        return None
    except Exception as e:
        print(f"Error checking user: {e}")
        return None

def create_user_profile(profile_data: Dict) -> str:
    """Create new user profile in MongoDB"""
    try:
        profile_data['created_at'] = datetime.utcnow()
        profile_data['updated_at'] = datetime.utcnow()
        profile_data['preferred_language'] = user_language
        
        result = users_collection.insert_one(profile_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating user profile: {e}")
        return None

def get_profile_info_smartly():
    """Create profile using smart information extraction"""
    profile = {}
    
    welcome_msg = """I need to create your medical profile to provide you with the best care. I'll ask you a few questions. 
    Please speak naturally - you can say things like 'My name is John' or 'I am 25 years old' and I'll understand."""
    
    translated_welcome = translate_to_user_language(welcome_msg)
    print(f"🩺 {translated_welcome}")
    speak(translated_welcome)
    
    # Collect name
    name_prompt = "Could you please tell me your full name?"
    name = collect_information_smartly("NAME", name_prompt, "Patient's full name for medical records")
    if name == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    if name:
        profile['fullname'] = name
        confirm_msg = f"Thank you, {name}. I've recorded your name."
        speak(translate_to_user_language(confirm_msg))
    else:
        return None
    
    # Collect age
    age_prompt = "What is your age? You can say it however you're comfortable."
    age_str = collect_information_smartly("AGE", age_prompt, "Patient's age in years")
    if age_str == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    if age_str:
        try:
            age = int(age_str)
            if 1 <= age <= 120:
                profile['age'] = age
                confirm_msg = f"I've recorded your age as {age} years."
                speak(translate_to_user_language(confirm_msg))
        except:
            error_msg = "I had trouble understanding your age. Let me ask again."
            speak(translate_to_user_language(error_msg))
            return get_profile_info_smartly()
    
    # Collect gender
    gender_prompt = "What is your gender?"
    gender = collect_information_smartly("GENDER", gender_prompt, "Patient's gender")
    if gender == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    if gender:
        profile['gender'] = gender
        confirm_msg = "Thank you, I've recorded your gender."
        speak(translate_to_user_language(confirm_msg))
    
    # Collect height
    height_prompt = "What is your height? You can tell me in any unit you prefer."
    height_str = collect_information_smartly("HEIGHT", height_prompt, "Patient's height")
    if height_str == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    if height_str:
        # Process height with AI to extract numerical value
        height_info = extract_information_with_ai(height_str, "MEDICAL_VALUE", "height measurement")
        try:
            height_value = float(re.findall(r'\d+\.?\d*', height_info["extracted_value"])[0])
            # Convert to cm if needed
            if height_value < 10:  # Likely feet
                height_value *= 30.48
            profile['height'] = height_value
            confirm_msg = f"I've recorded your height as {height_value:.1f} centimeters."
            speak(translate_to_user_language(confirm_msg))
        except:
            profile['height'] = height_str
    
    # Medical history
    history_prompt = "Do you have any important medical conditions or medical history I should know about?"
    history = collect_information_smartly("MEDICAL_HISTORY", history_prompt, "Patient's medical history")
    if history == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    profile['medical_history'] = history if history else "No significant medical history reported"
    
    # Allergies
    allergy_prompt = "Do you have any allergies to medications, foods, or other substances?"
    allergies = collect_information_smartly("ALLERGIES", allergy_prompt, "Patient's allergies")
    if allergies == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    profile['allergies'] = allergies if allergies else "No known allergies"
    
    # Medications
    med_prompt = "Are you currently taking any medications?"
    medications = collect_information_smartly("MEDICATIONS", med_prompt, "Patient's current medications")
    if medications == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    profile['medications'] = medications if medications else "No current medications"
    
    # Emergency contact
    emergency_prompt = "Could you please provide an emergency contact phone number?"
    emergency = collect_information_smartly("EMERGENCY_CONTACT", emergency_prompt, "Emergency contact information")
    if emergency == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    profile['emergency_contact'] = emergency if emergency else "Not provided"
    
    completion_msg = "Perfect! I've created your medical profile successfully."
    speak(translate_to_user_language(completion_msg))
    
    return profile

# 10. SMART VITAL SIGNS COLLECTION
def get_vital_sign_smartly(vital_type: str, prompt: str, expected_range: str) -> float | None:
    """Collect vital signs using smart extraction"""
    vital_prompt = f"{prompt} {expected_range}"
    
    vital_str = collect_information_smartly("MEDICAL_VALUE", vital_prompt, f"{vital_type} measurement")
    
    if vital_str == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    
    if vital_str:
        try:
            # Extract numerical value using AI
            extraction_result = extract_information_with_ai(vital_str, "MEDICAL_VALUE", f"{vital_type} reading")
            
            if vital_type == "blood_pressure":
                # Special handling for blood pressure
                bp_values = re.findall(r'\d+', extraction_result["extracted_value"])
                if len(bp_values) >= 2:
                    return (int(bp_values[0]), int(bp_values[1]))
            else:
                # Single numerical value
                numbers = re.findall(r'\d+\.?\d*', extraction_result["extracted_value"])
                if numbers:
                    return float(numbers[0])
        except:
            pass
    
    return None

def collect_vital_signs_smartly(manual_mode=False) -> Dict:
    """Collect vital signs with smart extraction"""
    collecting_msg = "Now I'll collect your vital signs. Please have your measurements ready."
    speak(translate_to_user_language(collecting_msg))
    
    vitals = {}
    
    if manual_mode:
        # Temperature
        temp = get_vital_sign_smartly("temperature", 
            "Please tell me your body temperature.", 
            "Normal range is usually 96-104°F.")
        if temp == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        vitals['temperature'] = temp if temp else 98.6
        
        # Heart Rate
        hr = get_vital_sign_smartly("heart_rate",
            "Please tell me your heart rate.",
            "Normal range is usually 60-100 beats per minute.")
        if hr == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        vitals['heart_rate'] = hr if hr else 72
        
        # Pulse Rate
        pulse = get_vital_sign_smartly("pulse_rate",
            "Please tell me your pulse rate.",
            "This is usually similar to heart rate.")
        if pulse == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        vitals['pulse_rate'] = pulse if pulse else 70
        
        # Oxygen Saturation
        o2 = get_vital_sign_smartly("oxygen_saturation",
            "Please tell me your oxygen saturation percentage.",
            "Normal is usually above 95%.")
        if o2 == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        vitals['oxygen_saturation'] = o2 if o2 else 98.5
        
        # Blood Pressure
        bp = get_vital_sign_smartly("blood_pressure",
            "Please tell me your blood pressure.",
            "Please give both systolic and diastolic readings.")
        if bp == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        if isinstance(bp, tuple):
            vitals['blood_pressure_systolic'] = bp[0]
            vitals['blood_pressure_diastolic'] = bp[1]
        else:
            vitals['blood_pressure_systolic'] = 120
            vitals['blood_pressure_diastolic'] = 80
        
        # Weight
        weight = get_vital_sign_smartly("weight",
            "Please tell me your weight.",
            "You can use any unit you prefer.")
        if weight == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        vitals['weight'] = weight if weight else 70.0
        
    else:
        # Sensor readings (mock functions - replace with actual sensor code)
        vitals['temperature'] = 98.6
        vitals['heart_rate'] = 72
        vitals['pulse_rate'] = 70
        vitals['oxygen_saturation'] = 98.5
        vitals['blood_pressure_systolic'] = 120
        vitals['blood_pressure_diastolic'] = 80
        vitals['weight'] = 70.0
    
    return vitals

# 11. ENHANCED HEALTH ANALYSIS
def analyze_health_data(user_profile: Dict, symptoms: str, vitals: Dict) -> Dict:
    """Analyze health data with comprehensive AI analysis"""
    
    symptoms_english = translate_from_user_language(symptoms)
    
    health_context = f"""
    You are an experienced medical AI assistant analyzing patient data. Please provide a comprehensive but careful analysis.
    
    PATIENT PROFILE:
    Name: {user_profile['fullname']}
    Age: {user_profile['age']} years
    Gender: {user_profile['gender']}
    Height: {user_profile.get('height', 'Not specified')} cm
    Medical History: {user_profile.get('medical_history', 'None specified')}
    Allergies: {user_profile.get('allergies', 'None specified')}
    Current Medications: {user_profile.get('medications', 'None specified')}
    
    CURRENT SYMPTOMS:
    {symptoms_english}
    
    VITAL SIGNS:
    • Temperature: {vitals['temperature']}°F
    • Heart Rate: {vitals['heart_rate']} bpm
    • Pulse Rate: {vitals['pulse_rate']} bpm
    • Oxygen Saturation: {vitals['oxygen_saturation']}%
    • Blood Pressure: {vitals['blood_pressure_systolic']}/{vitals['blood_pressure_diastolic']} mmHg
    • Weight: {vitals['weight']} kg
    
    ANALYSIS REQUIREMENTS:
    1. Assess if vital signs are within normal ranges for the patient's age and gender
    2. Correlate symptoms with vital sign patterns
    3. Identify any concerning trends or red flags
    4. Provide preliminary assessment with appropriate caveats
    5. Determine urgency level: Normal/Monitor/Consult Doctor/Emergency
    6. Give specific, actionable recommendations
    7. Respond in {user_language} for patient understanding
    
    IMPORTANT MEDICAL DISCLAIMERS TO INCLUDE:
    - This is a preliminary assessment only
    - Not a substitute for professional medical diagnosis
    - Recommend consulting healthcare provider for persistent or concerning symptoms
    - Advise immediate medical attention for emergency symptoms
    
    Please provide a caring, professional response in {user_language} that includes:
    - Assessment of current condition
    - Explanation of vital signs in simple terms
    - Specific recommendations
    - When to seek professional help
    """
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": f"""You are a compassionate medical AI assistant. Provide thorough but cautious health analysis in {user_language}. 
                Always emphasize that this is preliminary guidance and professional medical consultation is important for proper diagnosis and treatment.
                Be caring, professional, and clear in your explanations. Use simple language that patients can understand."""},
                {"role": "user", "content": health_context}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        
        analysis = response.choices[0].message.content
        
        # Determine severity level using AI
        severity_prompt = f"""
        Based on the symptoms: "{symptoms_english}" and vital signs analysis, determine the urgency level.
        
        Respond with only one word:
        - Normal: Minor issues, routine monitoring
        - Monitor: Keep watching, not immediately concerning
        - Consult: Should see doctor within few days
        - Emergency: Needs immediate medical attention
        """
        
        severity_response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": "You are a medical triage assistant. Respond with only the urgency level word."},
                {"role": "user", "content": severity_prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        severity = severity_response.choices[0].message.content.strip()
        
        return {
            'diagnosis': analysis,
            'recommendations': analysis,
            'severity': severity
        }
        
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        error_analysis = """I apologize, but I'm having technical difficulties analyzing your health data right now. 
        
        For your safety, I recommend:
        1. If you're experiencing severe symptoms, please seek immediate medical attention
        2. For ongoing concerns, please consult with a healthcare provider
        3. Keep monitoring your vital signs
        4. You can try this analysis again later
        
        Your health and safety are most important."""
        
        return {
            'diagnosis': translate_to_user_language(error_analysis),
            'recommendations': translate_to_user_language(error_analysis),
            'severity': 'Unknown'
        }

# 12. SAVE MEDICAL READING TO MONGODB
def save_medical_reading(user_id: str, vitals: Dict, symptoms: str, diagnosis: str, recommendations: str):
    """Save medical reading to MongoDB with enhanced data"""
    try:
        reading_data = {
            'user_id': user_id,
            'temperature': vitals['temperature'],
            'heart_rate': vitals['heart_rate'],
            'pulse_rate': vitals['pulse_rate'],
            'oxygen_saturation': vitals['oxygen_saturation'],
            'blood_pressure_systolic': vitals['blood_pressure_systolic'],
            'blood_pressure_diastolic': vitals['blood_pressure_diastolic'],
            'weight': vitals['weight'],
            'symptoms': symptoms,
            'symptoms_english': translate_from_user_language(symptoms),
            'diagnosis': diagnosis,
            'recommendations': recommendations,
            'patient_language': user_language,
            'reading_date': datetime.utcnow(),
            'device_id': os.getenv('DEVICE_ID', 'medical_assistant_v2'),
            'location': os.getenv('DEVICE_LOCATION', 'clinic')
        }
        
        result = readings_collection.insert_one(reading_data)
        print(f"✔ Medical reading saved with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error saving medical reading: {e}")
        return None

# 13. MAIN APPLICATION
def main():
    global user_language
    
    # Welcome message
    welcome = """🩺 Welcome to your Advanced Medical Assistant! 
    
    I'm here to help you with professional health guidance. I can:
    ✅ Understand your natural speech patterns
    ✅ Communicate in your preferred language
    ✅ Analyze your symptoms and vital signs
    ✅ Provide personalized health recommendations
    ✅ Maintain your medical records securely
    
    Important: This system provides preliminary health guidance. Always consult healthcare professionals for medical concerns."""
    
    print(f"🏥 {welcome}")
    speak(welcome)
    
    # Select language first
    select_user_language()
    
    # Main consultation loop
    while True:
        try:
            # Get patient name with smart extraction
            name_prompt = "To start your consultation, could you please tell me your name?"
            patient_name = collect_information_smartly("NAME", name_prompt, "Patient identification")
            
            if patient_name == "EXIT_COMMAND":
                break
            
            if not patient_name:
                continue_msg = "I need your name to access your medical records. Let me try again."
                speak(translate_to_user_language(continue_msg))
                continue
            
            print(f"👤 Patient: {patient_name}")
            log_conversation("User", patient_name)
            
            # Check if user exists
            user_profile = check_user_exists(patient_name)
            
            if user_profile:
                # Set language from profile
                if 'preferred_language' in user_profile:
                    user_language = user_profile['preferred_language']
                
                welcome_back_msg = f"Welcome back, {user_profile['fullname']}! I have your medical profile ready."
                translated_welcome = translate_to_user_language(welcome_back_msg)
                print(f"🩺 {translated_welcome}")
                speak(translated_welcome)
                
            else:
                # Create new profile
                new_patient_msg = f"Hello {patient_name}! I'll need to create your medical profile first."
                translated_new = translate_to_user_language(new_patient_msg)
                print(f"🩺 {translated_new}")
                speak(translated_new)
                
                profile_data = get_profile_info_smartly()
                
                if profile_data == "EXIT_COMMAND":
                    break
                
                if not profile_data:
                    error_msg = "I couldn't create your profile. Let me try again."
                    speak(translate_to_user_language(error_msg))
                    continue
                
                user_id = create_user_profile(profile_data)
                
                if user_id:
                    user_profile = check_user_exists(patient_name)
                    success_msg = "Your medical profile has been created successfully!"
                    speak(translate_to_user_language(success_msg))
                else:
                    error_msg = "There was an issue creating your profile. Please try again."
                    speak(translate_to_user_language(error_msg))
                    continue
            
            # Get current symptoms
            symptoms_prompt = """Now, please tell me about your current health concerns or symptoms. 
            You can describe them naturally, like 'I have a headache and feel dizzy' or 'My chest hurts when I breathe'."""
            
            symptoms = collect_information_smartly("SYMPTOMS", symptoms_prompt, "Current health symptoms and concerns")
            
            if symptoms == "EXIT_COMMAND":
                break
            
            if not symptoms:
                no_symptoms_msg = "I need to understand your symptoms to help you properly. Let me ask again."
                speak(translate_to_user_language(no_symptoms_msg))
                continue
            
            print(f"📋 Symptoms: {symptoms}")
            log_conversation("User", symptoms)
            
            # Acknowledge symptoms
            symptoms_acknowledged = "I understand your concerns. Now let me collect your vital signs to get a complete picture of your health."
            speak(translate_to_user_language(symptoms_acknowledged))
            
            # Ask for reading mode
            mode_prompt = """Would you like to provide your vital sign readings manually, or should I take them from connected sensors? 
            Say 'manual' if you'll tell me the readings, or 'sensor' for automatic measurement."""
            
            mode_response = collect_information_smartly("READING_MODE", mode_prompt, "Method for collecting vital signs")
            
            if mode_response == "EXIT_COMMAND":
                break
            
            manual_mode = mode_response and 'manual' in mode_response.lower()
            
            # Collect vital signs
            vitals = collect_vital_signs_smartly(manual_mode)
            
            if vitals == "EXIT_COMMAND":
                break
            
            # Display collected vitals
            vitals_summary = f"""
            📊 Your Vital Signs:
            🌡️ Temperature: {vitals['temperature']}°F
            💓 Heart Rate: {vitals['heart_rate']} bpm
            🫀 Pulse Rate: {vitals['pulse_rate']} bpm
            🫁 Oxygen Saturation: {vitals['oxygen_saturation']}%
            🩸 Blood Pressure: {vitals['blood_pressure_systolic']}/{vitals['blood_pressure_diastolic']} mmHg
            ⚖️ Weight: {vitals['weight']} kg
            """
            
            print(vitals_summary)
            log_conversation("Assistant", vitals_summary)
            
            analyzing_msg = "I have all your information. Let me analyze your health data now..."
            speak(translate_to_user_language(analyzing_msg))
            
            # Analyze health data
            analysis = analyze_health_data(user_profile, symptoms, vitals)
            
            # Present analysis
            print(f"🩺 Health Analysis:\n{analysis['diagnosis']}")
            log_conversation("Assistant", analysis['diagnosis'])
            speak(analysis['diagnosis'])  # Already translated
            
            # Save to MongoDB
            reading_id = save_medical_reading(
                user_profile['_id'], vitals, symptoms, 
                analysis['diagnosis'], analysis['recommendations']
            )
            
            # Final care message
            care_msg = f"""Your health consultation is complete and has been securely saved to your medical record.
            
            Severity Level: {analysis['severity']}
            
            Remember: This is preliminary guidance to help you understand your health better. For proper medical diagnosis and treatment, please consult with qualified healthcare professionals.
            
            Take care of yourself, and don't hesitate to seek medical attention if your symptoms worsen or if you have any concerns."""
            
            translated_care = translate_to_user_language(care_msg)
            print(f"🩺 {translated_care}")
            speak(translated_care)
            
            # Ask for next action
            next_action_prompt = """What would you like to do next?
            - Say 'another consultation' for a new health check
            - Say 'new patient' to help someone else
            - Say 'exit' to finish
            
            How can I continue to help you?"""
            
            next_action = collect_information_smartly("NEXT_ACTION", next_action_prompt, "User's choice for next steps")
            
            if next_action == "EXIT_COMMAND":
                break
            elif next_action and any(word in next_action.lower() for word in ["new", "different", "patient", "someone"]):
                new_patient_msg = "I'll help you with a new patient consultation."
                speak(translate_to_user_language(new_patient_msg))
                continue
            elif next_action and any(word in next_action.lower() for word in ["another", "more", "again", "consultation"]):
                another_msg = "I'm ready for another consultation with you."
                speak(translate_to_user_language(another_msg))
                # Continue with same user profile
                continue
            else:
                break
                
        except Exception as e:
            error_msg = f"I encountered a technical issue: {e}. Let me restart your consultation."
            print(f"❌ {error_msg}")
            log_conversation("System", error_msg)
            
            restart_msg = "I apologize for the technical difficulty. Let me restart your consultation."
            speak(translate_to_user_language(restart_msg))
            continue
    
    # Farewell
    farewell = """Thank you for using your Advanced Medical Assistant. 
    
    Remember to take care of your health and consult healthcare professionals when needed. 
    
    I'm always here whenever you need health guidance. Stay healthy and take care!"""
    
    translated_farewell = translate_to_user_language(farewell)
    print(f"🏥 {translated_farewell}")
    speak(translated_farewell)
    log_conversation("Assistant", farewell)

if __name__ == "__main__":
    print("=" * 80)
    print("🏥 ADVANCED MEDICAL ASSISTANT - Enhanced with AI Intelligence")
    print("=" * 80)
    print("🎯 KEY FEATURES:")
    print("   ✅ Smart Information Extraction - Understands natural speech")
    print("   ✅ Multi-language Support - Choose your preferred language")
    print("   ✅ AI-Powered Analysis - Comprehensive health assessment")
    print("   ✅ Secure Cloud Storage - Your data is safely stored")
    print("   ✅ Professional Medical Guidance - Like talking to a doctor")
    print("   ✅ Continuous Operation - Multiple consultations supported")
    print("=" * 80)
    print("🌟 IMPROVEMENTS:")
    print("   • Language selection at startup")
    print("   • Smart name extraction (ignores 'my name is')")
    print("   • Natural conversation understanding")
    print("   • Enhanced AI analysis with medical reasoning")
    print("   • Better error handling and user guidance")
    print("   • Professional medical consultation experience")
    print("=" * 80)
    print("🎙️ USAGE TIPS:")
    print("   • Speak naturally - say 'My name is John' or 'I'm feeling dizzy'")
    print("   • The assistant extracts key information automatically")
    print("   • Choose your language at the beginning")
    print("   • All information is confirmed before proceeding")
    print("   • Say 'exit' anytime to quit")
    print("=" * 80)
    print("🚀 Starting Enhanced Medical Assistant...")
    print("=" * 80)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Medical Assistant stopped by user")
        farewell_msg = "Medical consultation ended. Take care of your health!"
        speak(farewell_msg)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        error_msg = "I apologize for the technical issue. Please restart the application and contact support if the problem continues."
        speak(error_msg)
        log_conversation("System", f"Critical Error: {e}")
    finally:
        print("🏥 Medical Assistant session completed")
        print("💙 Thank you for using Advanced Medical Assistant!")
        log_conversation("System", "Medical Assistant session ended")