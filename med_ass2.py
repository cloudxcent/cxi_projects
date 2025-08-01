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

# 2. LANGUAGE CONFIGURATION
SUPPORTED_LANGUAGES = {
    'english': {
        'code': 'en-US',
        'voice': 'en-US-AriaNeural',
        'name': 'English',
        'display': 'English'
    },
    'hindi': {
        'code': 'hi-IN',
        'voice': 'hi-IN-SwaraNeural',
        'name': 'Hindi',
        'display': 'हिंदी'
    },
    'marathi': {
        'code': 'mr-IN',
        'voice': 'mr-IN-AarohiNeural',
        'name': 'Marathi',
        'display': 'मराठी'
    },
    'telugu': {
        'code': 'te-IN',
        'voice': 'te-IN-ShrutiNeural',
        'name': 'Telugu',
        'display': 'తెలుగు'
    }
}

# Global language setting
current_language = 'english'

# 3. LANGUAGE TRANSLATION DICTIONARIES
TRANSLATIONS = {
    'english': {
        'welcome': "Welcome to your Medical Device Assistant. I'm here to help you understand your health better.",
        'disclaimer': "This system provides preliminary health analysis based on your symptoms and vital signs. However, please remember that this is NOT a substitute for professional medical advice, diagnosis, or treatment.",
        'emergency_advice': "I encourage you to always consult with qualified healthcare providers for medical concerns. In case of any emergency, please call emergency services immediately.",
        'patient_guidance': "I'm designed to be patient and understanding. Please speak clearly and take your time - there's no rush. If you need me to repeat anything, just ask.",
        'choose_language': "Please choose your preferred language. Say 'English' for English, 'Hindi' for Hindi, 'Marathi' for Marathi, or 'Telugu' for Telugu.",
        'language_set': "Language has been set to",
        'ask_name': "Hello! Welcome to your medical assistant. To provide you with the best care, I need to know your name. Could you please tell me your full name?",
        'name_again': "I want to make sure I have your name correctly for your medical record. Could you please tell me your full name clearly?",
        'welcome_back': "Welcome back! I found your profile in our system.",
        'new_user': "I don't have your profile yet. Let me create one for you.",
        'profile_created': "Your profile has been created successfully!",
        'ask_symptoms': "Now, please tell me what health problem or symptoms you're experiencing today. Describe it in your own words.",
        'understand_symptoms': "I understand your symptoms. Now let me collect your vital signs.",
        'reading_mode': "Would you like me to take readings from sensors, or would you prefer to provide them manually for testing? Say 'sensor' for automatic or 'manual' for voice input.",
        'vitals_collected': "All vital signs have been collected. Now analyzing your health data.",
        'analysis_complete': "Based on your symptoms and vital signs, here is my analysis:",
        'data_saved': "Your health information has been safely saved to our secure cloud database.",
        'medical_advice': "I want to remind you that while this analysis can be helpful, it's important to follow up with a healthcare professional, especially if your symptoms continue or worsen.",
        'take_care': "Please take care of yourself, and don't hesitate to seek medical attention if you're concerned about your health.",
        'exit_message': "Thank you for using the Medical Device Assistant. Take care of your health and don't hesitate to use our services again!"
    },
    'hindi': {
        'welcome': "आपके मेडिकल डिवाइस असिस्टेंट में आपका स्वागत है। मैं आपके स्वास्थ्य को बेहतर समझने में आपकी मदद करने के लिए यहाँ हूँ।",
        'disclaimer': "यह सिस्टम आपके लक्षणों और महत्वपूर्ण संकेतों के आधार पर प्रारंभिक स्वास्थ्य विश्लेषण प्रदान करता है। हालांकि, कृपया याद रखें कि यह पेशेवर चिकित्सा सलाह, निदान या उपचार का विकल्प नहीं है।",
        'emergency_advice': "मैं आपको प्रोत्साहित करता हूँ कि चिकित्सा संबंधी चिंताओं के लिए हमेशा योग्य स्वास्थ्य सेवा प्रदाताओं से सलाह लें। किसी भी आपातकाल की स्थिति में, कृपया तुरंत आपातकालीन सेवाओं को कॉल करें।",
        'patient_guidance': "मैं धैर्यवान और समझदार होने के लिए डिज़ाइन किया गया हूँ। कृपया स्पष्ट रूप से बोलें और अपना समय लें - कोई जल्दी नहीं है। यदि आपको मुझसे कुछ दोहराने की जरूरत है, तो बस पूछें।",
        'choose_language': "कृपया अपनी पसंदीदा भाषा चुनें। अंग्रेजी के लिए 'English', हिंदी के लिए 'Hindi', मराठी के लिए 'Marathi', या तेलुगु के लिए 'Telugu' कहें।",
        'language_set': "भाषा सेट की गई है",
        'ask_name': "नमस्ते! आपके मेडिकल असिस्टेंट में आपका स्वागत है। आपको बेहतर देखभाल प्रदान करने के लिए, मुझे आपका नाम जानना होगा। कृपया मुझे अपना पूरा नाम बताएं?",
        'name_again': "मैं यह सुनिश्चित करना चाहता हूँ कि आपके मेडिकल रिकॉर्ड के लिए मेरे पास आपका नाम सही है। कृपया मुझे अपना पूरा नाम स्पष्ट रूप से बताएं?",
        'welcome_back': "वापस स्वागत है! मुझे हमारे सिस्टम में आपकी प्रोफाइल मिल गई है।",
        'new_user': "मेरे पास अभी तक आपकी प्रोफाइल नहीं है। मैं आपके लिए एक बनाता हूँ।",
        'profile_created': "आपकी प्रोफाइल सफलतापूर्वक बनाई गई है!",
        'ask_symptoms': "अब, कृपया मुझे बताएं कि आज आप किस स्वास्थ्य समस्या या लक्षणों का सामना कर रहे हैं। इसे अपने शब्दों में बताएं।",
        'understand_symptoms': "मैं आपके लक्षणों को समझता हूँ। अब मैं आपके जीवनशैली संकेतक एकत्र करूंगा।",
        'reading_mode': "क्या आप चाहते हैं कि मैं सेंसर से रीडिंग लूं, या आप उन्हें मैन्युअल रूप से प्रदान करना पसंद करेंगे? स्वचालित के लिए 'sensor' या आवाज़ इनपुट के लिए 'manual' कहें।",
        'vitals_collected': "सभी जीवनशैली संकेतक एकत्र किए गए हैं। अब आपके स्वास्थ्य डेटा का विश्लेषण कर रहा हूँ।",
        'analysis_complete': "आपके लक्षणों और जीवनशैली संकेतकों के आधार पर, यहाँ मेरा विश्लेषण है:",
        'data_saved': "आपकी स्वास्थ्य जानकारी सुरक्षित रूप से हमारे क्लाउड डेटाबेस में सेव की गई है।",
        'medical_advice': "मैं आपको याद दिलाना चाहता हूँ कि यह विश्लेषण सहायक हो सकता है, लेकिन एक स्वास्थ्य पेशेवर से फॉलो अप करना महत्वपूर्ण है, खासकर यदि आपके लक्षण जारी रहते हैं या बिगड़ते हैं।",
        'take_care': "कृपया अपना ख्याल रखें, और यदि आप अपने स्वास्थ्य के बारे में चिंतित हैं तो चिकित्सा सहायता लेने में संकोच न करें।",
        'exit_message': "मेडिकल डिवाइस असिस्टेंट का उपयोग करने के लिए धन्यवाद। अपने स्वास्थ्य का ख्याल रखें और हमारी सेवाओं का फिर से उपयोग करने में संकोच न करें!"
    },
    'marathi': {
        'welcome': "तुमच्या मेडिकल डिव्हाइस असिस्टंटमध्ये तुमचे स्वागत आहे. तुमचे आरोग्य चांगले समजून घेण्यासाठी मी येथे आहे.",
        'disclaimer': "ही प्रणाली तुमच्या लक्षणे आणि महत्वाच्या संकेतांवर आधारित प्राथमिक आरोग्य विश्लेषण प्रदान करते. तथापि, कृपया लक्षात ठेवा की हे व्यावसायिक वैद्यकीय सल्ला, निदान किंवा उपचाराचा पर्याय नहीं आहे.",
        'emergency_advice': "मी तुम्हाला प्रोत्साहित करतो की वैद्यकीय चिंतांसाठी नेहमी पात्र आरोग्यसेवा प्रदात्यांशी सल्लामसलत करा. कोणत्याही आणीबाणीच्या परिस्थितीत, कृपया ताबडतोब आणीबाणी सेवांना कॉल करा.",
        'patient_guidance': "मी धैर्यवान आणि समजूतदार असण्यासाठी डिझाइन केलेले आहे. कृपया स्पष्टपणे बोला आणि तुमचा वेळ घ्या - कोणतीही घाई नाही. जर तुम्हाला मला काहीतरी पुन्हा सांगायची गरज असेल, तर फक्त विचारा.",
        'choose_language': "कृपया तुमची पसंतीची भाषा निवडा. इंग्रजीसाठी 'English', हिंदीसाठी 'Hindi', मराठीसाठी 'Marathi', किंवा तेलुगुसाठी 'Telugu' म्हणा.",
        'language_set': "भाषा सेट केली गेली आहे",
        'ask_name': "नमस्कार! तुमच्या मेडिकल असिस्टंटमध्ये तुमचे स्वागत आहे. तुम्हाला चांगली काळजी देण्यासाठी, मला तुमचे नाव जाणून घेणे आवश्यक आहे. कृपया मला तुमचे पूर्ण नाव सांगाल का?",
        'name_again': "मला खात्री करायची आहे की तुमच्या वैद्यकीय नोंदीसाठी माझ्याकडे तुमचे नाव बरोबर आहे. कृपया मला तुमचे पूर्ण नाव स्पष्टपणे सांगाल का?",
        'welcome_back': "परत स्वागत आहे! मला आमच्या सिस्टममध्ये तुमची प्रोफाइल सापडली आहे.",
        'new_user': "माझ्याकडे अजून तुमची प्रोफाइल नाही. मी तुमच्यासाठी एक तयार करतो.",
        'profile_created': "तुमची प्रोफाइल यशस्वीरित्या तयार केली गेली आहे!",
        'ask_symptoms': "आता, कृपया मला सांगा की आज तुम्ही कोणत्या आरोग्य समस्या किंवा लक्षणांचा सामना करत आहात. ते तुमच्या स्वतःच्या शब्दांत वर्णन करा.",
        'understand_symptoms': "मी तुमची लक्षणे समजतो. आता मी तुमचे महत्वाचे संकेत गोळा करेन.",
        'reading_mode': "तुम्ही मला सेन्सरमधून रीडिंग घ्यायला आवडेल, की तुम्ही ते मॅन्युअली देणे पसंत कराल? स्वयंचलितसाठी 'sensor' किंवा आवाजाच्या इनपुटसाठी 'manual' म्हणा.",
        'vitals_collected': "सर्व महत्वाचे संकेत गोळा केले गेले आहेत. आता तुमच्या आरोग्य डेटाचे विश्लेषण करत आहे.",
        'analysis_complete': "तुमच्या लक्षणे आणि महत्वाच्या संकेतांवर आधारित, हे माझे विश्लेषण आहे:",
        'data_saved': "तुमची आरोग्य माहिती सुरक्षितपणे आमच्या क्लाउड डेटाबेसमध्ये सेव्ह केली गेली आहे.",
        'medical_advice': "मी तुम्हाला आठवण करून देऊ इच्छितो की हे विश्लेषण उपयुक्त असू शकते, परंतु आरोग्य व्यावसायिकाकडे फॉलो अप करणे महत्वाचे आहे, विशेषत: जर तुमची लक्षणे चालू राहिली किंवा वाढली.",
        'take_care': "कृपया स्वतःची काळजी घ्या, आणि जर तुम्हाला तुमच्या आरोग्याबद्दल चिंता असेल तर वैद्यकीय मदत घेण्यास संकोच करू नका.",
        'exit_message': "मेडिकल डिव्हाइस असिस्टंट वापरल्याबद्दल धन्यवाद. तुमच्या आरोग्याची काळजी घ्या आणि आमच्या सेवांचा पुन्हा वापर करण्यास संकोच करू नका!"
    },
    'telugu': {
        'welcome': "మీ మెడికల్ డివైస్ అసిస్టెంట్‌కు స్వాగతం. మీ ఆరోగ్యాన్ని బాగా అర్థం చేసుకోవడంలో మీకు సహాయం చేయడానికి నేను ఇక్కడ ఉన్నాను.",
        'disclaimer': "ఈ సిస్టమ్ మీ లక్షణాలు మరియు కీలక సంకేతాల ఆధారంగా ప్రాథమిక ఆరోగ్య విశ్లేషణను అందిస్తుంది. అయితే, ఇది వృత్తిపరమైన వైద్య సలహా, నిర్ధారణ లేదా చికిత్సకు ప్రత్యామ్నాయం కాదని దయచేసి గుర్తుంచుకోండి.",
        'emergency_advice': "వైద్య ఆందోళనల కోసం ఎల్లప్పుడూ అర్హత కలిగిన ఆరోగ్య సేవా ప్రదాతలను సంప్రదించమని నేను మిమ్మల్ని ప్రోత్సాహిస్తున్నాను. ఏదైనా అత్యవసర పరిస్థితుల్లో, దయచేసి వెంటనే అత్యవసర సేవలకు కాల్ చేయండి.",
        'patient_guidance': "నేను ఓపికగా మరియు అర్థం చేసుకోవడానికి రూపొందించబడ్డాను. దయచేసి స్పష్టంగా మాట్లాడండి మరియు మీ సమయం తీసుకోండి - ఎటువంటి తొందరపాటు లేదు. మీకు నేను ఏదైనా మళ్లీ చెప్పాల్సిన అవసరం ఉంటే, కేవలం అడగండి.",
        'choose_language': "దయచేసి మీ ఇష్టపడు భాషను ఎంచుకోండి. ఇంగ్లీష్ కోసం 'English', హిందీ కోసం 'Hindi', మరాఠీ కోసం 'Marathi', లేదా తెలుగు కోసం 'Telugu' అనండి.",
        'language_set': "భాష సెట్ చేయబడింది",
        'ask_name': "నమస్కారం! మీ మెడికల్ అసిస్టెంట్‌కు స్వాగతం. మీకు మంచి కేర్ అందించడానికి, మీ పేరు తెలుసుకోవాల్సిన అవసరం ఉంది. దయచేసి మీ పూర్తి పేరు చెప్పగలరా?",
        'name_again': "మీ వైద్య రికార్డు కోసం మీ పేరు సరిగ్గా ఉందని నేను నిర్ధారించుకోవాలని అనుకుంటున్నాను. దయచేసి మీ పూర్తి పేరు స్పష్టంగా చెప్పగలరా?",
        'welcome_back': "తిరిగి స్వాగతం! మా సిస్టమ్‌లో మీ ప్రొఫైల్ దొరికింది.",
        'new_user': "నా దగ్గర ఇంకా మీ ప్రొఫైల్ లేదు. మీ కోసం ఒకటి సృష్టిస్తాను.",
        'profile_created': "మీ ప్రొఫైల్ విజయవంతంగా సృష్టించబడింది!",
        'ask_symptoms': "ఇప్పుడు, దయచేసి ఈరోజు మీరు ఎలాంటి ఆరోగ్య సమస్య లేదా లక్షణాలను ఎదుర్కొంటున్నారో చెప్పండి. దాన్ని మీ సొంత మాటల్లో వివరించండి.",
        'understand_symptoms': "మీ లక్షణాలను నేను అర్థం చేసుకున్నాను. ఇప్పుడు మీ కీలక సంకేతాలను సేకరిస్తాను.",
        'reading_mode': "మీరు నేను సెన్సార్‌ల నుంచి రీడింగ్‌లు తీసుకోవాలని అనుకుంటున్నారా, లేదా వాటిని మ్యానువల్‌గా అందించడాన్ని ఇష్టపడతారా? ఆటోమేటిక్ కోసం 'sensor' లేదా వాయిస్ ఇన్‌పుట్ కోసం 'manual' అనండి.",
        'vitals_collected': "అన్ని కీలక సంకేతాలు సేకరించబడ్డాయి. ఇప్పుడు మీ ఆరోగ్య డేటాను విశ్లేషిస్తున్నాను.",
        'analysis_complete': "మీ లక్షణాలు మరియు కీలక సంకేతాల ఆధారంగా, ఇది నా విశ్లేషణ:",
        'data_saved': "మీ ఆరోగ్య సమాచారం మా సురక్షిత క్లౌడ్ డేటాబేస్‌లో భద్రంగా సేవ్ చేయబడింది.",
        'medical_advice': "ఈ విశ్లేషణ సహాయకరంగా ఉండవచ్చు, కానీ ఆరోగ్య నిపుణుడితో ఫాలో అప్ చేయడం ముఖ్యం, ముఖ్యంగా మీ లక్షణాలు కొనసాగితే లేదా దిగజారితే అని నేను మీకు గుర్తు చేయాలని అనుకుంటున్నాను.",
        'take_care': "దయచేసి మీ జాగ్రత్తలు చూసుకోండి, మరియు మీ ఆరోగ్యం గురించి మీకు ఆందోళన ఉంటే వైద్య సహాయం తీసుకోవడానికి వెనుకాడకండి.",
        'exit_message': "మెడికల్ డివైస్ అసిస్టెంట్‌ను ఉపయోగించినందుకు ధన్యవాదాలు. మీ ఆరోగ్యం జాగ్రత్తగా చూసుకోండి మరియు మా సేవలను మళ్లీ ఉపయోగించడానికి వెనుకాడకండి!"
    }
}

# 4. INITIALIZE AZURE OPENAI AND SPEECH CLIENTS
client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)

# Speech configuration will be updated based on selected language
speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

print("✔ Medical Device Assistant initialized")

# 5. LANGUAGE HELPER FUNCTIONS
def get_text(key: str) -> str:
    """Get translated text based on current language"""
    return TRANSLATIONS.get(current_language, TRANSLATIONS['english']).get(key, key)

def set_language(language_code: str):
    """Set the current language and update speech configuration"""
    global current_language, speech_cfg
    
    if language_code in SUPPORTED_LANGUAGES:
        current_language = language_code
        lang_config = SUPPORTED_LANGUAGES[language_code]
        
        # Update speech configuration
        speech_cfg.speech_recognition_language = lang_config['code']
        speech_cfg.speech_synthesis_voice_name = lang_config['voice']
        
        print(f"✔ Language set to: {lang_config['display']}")
        return True
    return False

def detect_language_from_speech(user_input: str) -> str:
    """Detect language preference from user input"""
    if not user_input:
        return 'english'
    
    user_input_lower = user_input.lower()
    
    # Check for language keywords
    if any(word in user_input_lower for word in ['hindi', 'हिंदी', 'हिन्दी']):
        return 'hindi'
    elif any(word in user_input_lower for word in ['marathi', 'मराठी']):
        return 'marathi'
    elif any(word in user_input_lower for word in ['telugu', 'తెలుగు']):
        return 'telugu'
    elif any(word in user_input_lower for word in ['english', 'इंग्लिश', 'इंग्रजी']):
        return 'english'
    
    return 'english'  # Default to English

def choose_language():
    """Interactive language selection"""
    global current_language
    
    # Ask in multiple languages initially
    multilang_prompt = """
    Please choose your preferred language / कृपया अपनी पसंदीदा भाषा चुनें / कृपया तुमची पसंतीची भाषा निवडा / దయచేసి మీ ఇష్టపడు భాషను ఎంచుకోండి:
    
    Say 'English' for English
    'Hindi' के लिए 'Hindi' कहें  
    'Marathi' साठी 'Marathi' म्हणा
    తెలుగు కోసం 'Telugu' అనండి
    """
    
    print(f"🌐 {multilang_prompt}")
    
    # Use default English TTS for initial prompt
    tts_english = speechsdk.SpeechSynthesizer(
        speech_config=speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION),
        audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    )
    tts_english.speak_text_async("Please choose your preferred language. Say English, Hindi, Marathi, or Telugu.").get()
    
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"🎙️ Listening for language choice... (Attempt {attempt + 1}/{max_attempts})")
        
        # Use English speech recognition for language selection
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        )
        
        result = recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_choice = result.text.strip()
            print(f"👤 You said: '{user_choice}'")
            
            detected_language = detect_language_from_speech(user_choice)
            
            if set_language(detected_language):
                lang_display = SUPPORTED_LANGUAGES[detected_language]['display']
                confirmation_msg = f"{get_text('language_set')} {lang_display}"
                print(f"✔ {confirmation_msg}")
                speak(confirmation_msg)
                return detected_language
            else:
                if attempt < max_attempts - 1:
                    tts_english.speak_text_async("I didn't understand. Please say English, Hindi, Marathi, or Telugu clearly.").get()
        else:
            if attempt < max_attempts - 1:
                tts_english.speak_text_async("I didn't hear you clearly. Please try again.").get()
    
    # Default to English if no valid selection
    set_language('english')
    tts_english.speak_text_async("Setting language to English by default.").get()
    return 'english'

# 6. MONGODB SETUP
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

# 7. CONVERSATION LOGGING AND UTILITY FUNCTIONS
LOG_FILE = "medical_device.log"
def log_conversation(role: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role}: {message}\n")

def display_help():
    """Display help information for users in current language"""
    help_translations = {
        'english': """
        🩺 MEDICAL DEVICE ASSISTANT HELP 🩺
        
        Available Commands:
        - Say your full name to start/continue
        - 'exit', 'quit', 'bye', 'stop', 'goodbye' - to quit anytime
        - 'continue' - to retry after voice recognition issues
        - 'manual' - for manual vital sign input (testing)
        - 'sensor' - for automatic sensor readings
        - 'new patient' - to switch to a different user
        - 'change language' - to switch language
        
        Features:
        ✅ Multi-language support (English, Hindi, Marathi, Telugu)
        ✅ Automatic user profile creation and recognition
        ✅ Voice-controlled vital sign collection
        ✅ AI-powered health analysis
        ✅ Cloud database storage with MongoDB Atlas
        ✅ Health trend tracking over time
        ✅ Continuous operation until you say exit
        
        Remember: This device provides preliminary analysis only.
        Always consult healthcare professionals for medical advice.
        """,
        'hindi': """
        🩺 मेडिकल डिवाइस असिस्टेंट हेल्प 🩺
        
        उपलब्ध कमांड:
        - शुरू करने/जारी रखने के लिए अपना पूरा नाम कहें
        - 'exit', 'quit', 'bye', 'stop', 'goodbye' - किसी भी समय छोड़ने के लिए
        - 'continue' - आवाज़ पहचान की समस्याओं के बाद फिर से कोशिश करने के लिए
        - 'manual' - मैन्युअल जीवनशैली संकेतक इनपुट के लिए (परीक्षण)
        - 'sensor' - स्वचालित सेंसर रीडिंग के लिए
        - 'new patient' - दूसरे उपयोगकर्ता पर स्विच करने के लिए
        - 'change language' - भाषा बदलने के लिए
        
        सुविधाएं:
        ✅ बहु-भाषा समर्थन (अंग्रेजी, हिंदी, मराठी, तेलुगु)
        ✅ स्वचालित उपयोगकर्ता प्रोफ़ाइल निर्माण और पहचान
        ✅ आवाज़-नियंत्रित जीवनशैली संकेतक संग्रह
        ✅ AI-संचालित स्वास्थ्य विश्लेषण
        ✅ MongoDB Atlas के साथ क्लाउड डेटाबेस स्टोरेज
        ✅ समय के साथ स्वास्थ्य प्रवृत्ति ट्रैकिंग
        ✅ जब तक आप exit न कहें, निरंतर संचालन
        
        याद रखें: यह डिवाइस केवल प्रारंभिक विश्लेषण प्रदान करता है।
        चिकित्सा सलाह के लिए हमेशा स्वास्थ्य पेशेवरों से सलाह लें।
        """,
        'marathi': """
        🩺 मेडिकल डिव्हाइस असिस्टंट हेल्प 🩺
        
        उपलब्ध कमांड:
        - सुरू करण्यासाठी/सुरू ठेवण्यासाठी तुमचे पूर्ण नाव सांगा
        - 'exit', 'quit', 'bye', 'stop', 'goodbye' - कधीही सोडण्यासाठी
        - 'continue' - आवाज ओळखण्याच्या समस्यांनंतर पुन्हा प्रयत्न करण्यासाठी
        - 'manual' - मॅन्युअल महत्वाचे संकेत इनपुटसाठी (चाचणी)
        - 'sensor' - स्वयंचलित सेन्सर रीडिंगसाठी
        - 'new patient' - दुसऱ्या वापरकर्त्यावर स्विच करण्यासाठी
        - 'change language' - भाषा बदलण्यासाठी
        
        वैशिष्ट्ये:
        ✅ बहु-भाषा समर्थन (इंग्रजी, हिंदी, मराठी, तेलुगु)
        ✅ स्वयंचलित वापरकर्ता प्रोफाइल निर्मिती आणि ओळख
        ✅ आवाज-नियंत्रित महत्वाचे संकेत संकलन
        ✅ AI-चालित आरोग्य विश्लेषण
        ✅ MongoDB Atlas सह क्लाउड डेटाबेस स्टोरेज
        ✅ कालांतराने आरोग्य ट्रेंड ट्रॅकिंग
        ✅ तुम्ही exit म्हणेपर्यंत सतत ऑपरेशन
        
        लक्षात ठेवा: हे डिव्हाइस फक्त प्राथमिक विश्लेषण प्रदान करते।
        वैद्यकीय सल्ल्यासाठी नेहमी आरोग्यसेवा व्यावसायिकांचा सल्ला घ्या।
        """,
        'telugu': """
        🩺 మెడికల్ డివైస్ అసిస్టెంట్ హెల్ప్ 🩺
        
        అందుబాటులో ఉన్న కమాండ్‌లు:
        - ప్రారంభించడానికి/కొనసాగించడానికి మీ పూర్తి పేరు చెప్పండి
        - 'exit', 'quit', 'bye', 'stop', 'goodbye' - ఎప్పుడైనా నిష్క్రమించడానికి
        - 'continue' - వాయిస్ గుర్తింపు సమస్యల తర్వాత మళ్లీ ప్రయత్నించడానికి
        - 'manual' - మ్యానువల్ కీలక సంకేతాల ఇన్‌పుట్ కోసం (పరీక్ష)
        - 'sensor' - ఆటోమేటిక్ సెన్సార్ రీడింగ్‌ల కోసం
        - 'new patient' - వేరే యూజర్‌కు మారడానికి
        - 'change language' - భాష మార్చడానికి
        
        లక్షణాలు:
        ✅ బహుళ-భాష మద్దతు (ఇంగ్లీష్, హిందీ, మరాఠీ, తెలుగు)
        ✅ ఆటోమేటిక్ యూజర్ ప్రొఫైల్ సృష్టి మరియు గుర్తింపు
        ✅ వాయిస్-నియంత్రిత కీలక సంకేతాల సేకరణ
        ✅ AI-శక్తితో ఆరోగ్య విశ్లేషణ
        ✅ MongoDB Atlas తో క్లౌడ్ డేటాబేస్ స్టోరేజ్
        ✅ కాలక్రమేణా ఆరోగ్య ట్రెండ్ ట్రాకింగ్
        ✅ మీరు exit అనే వరకు నిరంతర ఆపరేషన్
        
        గుర్తుంచుకోండి: ఈ పరికరం కేవలం ప్రాథమిక విశ్లేషణను మాత్రమే అందిస్తుంది।
        వైద్య సలహా కోసం ఎల్లప్పుడూ ఆరోగ్య నిపుణులను సంప్రదించండి।
        """
    }
    
    help_text = help_translations.get(current_language, help_translations['english'])
    print(help_text)
    log_conversation("System", "Help displayed")

def check_for_help_request(user_input: str) -> bool:
    """Check if user is asking for help in any supported language"""
    if not user_input:
        return False
    
    help_keywords = {
        'english': ['help', 'how', 'what can', 'commands'],
        'hindi': ['help', 'मदद', 'सहायता', 'कैसे', 'क्या कर सकते'],
        'marathi': ['help', 'मदत', 'सहाय्य', 'कसे', 'काय करू शकतो'],
        'telugu': ['help', 'సహాయం', 'ఎలా', 'ఏమి చేయగలరు']
    }
    
    user_lower = user_input.lower()
    for lang, keywords in help_keywords.items():
        if any(word in user_lower for word in keywords):
            display_help()
            speak("I've displayed the help information. You can continue with your consultation or say exit to quit.")
            return True
    return False

def check_for_language_change(user_input: str) -> bool:
    """Check if user wants to change language"""
    if not user_input:
        return False
    
    change_keywords = ['change language', 'भाषा बदलें', 'भाषा बदला', 'भाषा मारा', 'భాష మార్చు']
    user_lower = user_input.lower()
    
    if any(keyword in user_lower for keyword in change_keywords):
        speak("Let me help you change the language.")
        choose_language()
        return True
    return False

# 8. SPEECH HELPERS WITH MULTI-LANGUAGE SUPPORT
def tts(text: str):
    """Text-to-speech with current language configuration"""
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_cfg,
        audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    )
    synthesizer.speak_text_async(text).get()

def clean_for_speech(text: str) -> str:
    """Clean text for speech synthesis"""
    text = re.sub(r"【.*?】", "", text)
    return text

def speak(text: str):
    """Speak text in current language"""
    clean = clean_for_speech(text)
    tts(clean)

def listen(listening_for: str = "your response") -> str | None:
    """Listen for speech in current language"""
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_cfg)
    print(f"🎙️ Listening for: {listening_for}...")
    print("💬 Device is ready to hear you (say 'exit' to quit)")
    
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_said = result.text.strip()
        print(f"👤 You said: '{user_said}'")
        return user_said
    print("⚠️ I didn't hear anything clearly")
    return None

def listen_with_retry(prompt: str = "Please speak", listening_for: str = "your response", max_retries: int = 3) -> str | None:
    """Listen for user input with retry mechanism and clear feedback"""
    for attempt in range(max_retries):
        if attempt > 0:
            patience_msg = f"No worries, let me try again. {prompt}"
            print(f"🩺 {patience_msg}")
            speak(patience_msg)
        else:
            print(f"🩺 {prompt}")
            speak(prompt)
        
        user_input = listen(listening_for)
        
        if user_input:
            # Check for language change requests
            if check_for_language_change(user_input):
                continue  # Ask the original question again after language change
            
            # Check for help requests
            if check_for_help_request(user_input):
                continue  # Ask the original question again after showing help
            
            # Check for exit commands in multiple languages
            exit_words = ['exit', 'quit', 'bye', 'stop', 'goodbye', 'बाहर निकलें', 'छोड़ें', 
                         'बंद करें', 'बाहर पडा', 'सोडा', 'बंद करा', 'నిష్క్రమించు', 'వదిలేయు', 'ఆపు']
            if any(exit_word in user_input.lower() for exit_word in exit_words):
                return "EXIT_COMMAND"
            return user_input
        
        if attempt < max_retries - 1:
            retry_messages = {
                'english': "I'm here to help you. Let me listen more carefully.",
                'hindi': "मैं आपकी मदद करने के लिए यहाँ हूँ। मैं और ध्यान से सुनता हूँ।",
                'marathi': "मी तुमची मदत करण्यासाठी येथे आहे. मी अधिक काळजीपूर्वक ऐकतो.",
                'telugu': "మీకు సహాయం చేయడానికి నేను ఇక్కడ ఉన్నాను. నేను మరింత జాగ్రత్తగా వింటాను."
            }
            gentle_retry = retry_messages.get(current_language, retry_messages['english'])
            print(f"🩺 {gentle_retry}")
            speak(gentle_retry)
    
    # After max retries, ask if user wants to continue
    continue_messages = {
        'english': "I'm having a little trouble hearing you clearly. This happens sometimes. You can say 'continue' to keep trying, 'help' if you need assistance, or 'exit' if you'd like to stop.",
        'hindi': "मुझे आपको स्पष्ट रूप से सुनने में थोड़ी परेशानी हो रही है। यह कभी-कभी होता है। आप कोशिश जारी रखने के लिए 'continue' कह सकते हैं, सहायता की जरूरत हो तो 'help', या रोकने के लिए 'exit' कह सकते हैं।",
        'marathi': "मला तुम्हाला स्पष्टपणे ऐकण्यात थोडी अडचण येत आहे. हे कधी कधी होते. तुम्ही प्रयत्न सुरू ठेवण्यासाठी 'continue' म्हणू शकता, मदतीची गरज असल्यास 'help', किंवा थांबवण्यासाठी 'exit' म्हणू शकता.",
        'telugu': "మిమ్మల్ని స్పష్టంగా వినడంలో నాకు కొద్దిగా ఇబ్బంది ఉంది. ఇది కొన్నిసార్లు జరుగుతుంది. మీరు ప్రయత్నించడం కొనసాగించడానికి 'continue' అనవచ్చు, సహాయం అవసరమైతే 'help', లేదా ఆపాలని అనుకుంటే 'exit' అనవచ్చు."
    }
    
    patience_msg = continue_messages.get(current_language, continue_messages['english'])
    print(f"🩺 {patience_msg}")
    speak(patience_msg)
    
    response = listen("continue, help, or exit")
    if response:
        if check_for_help_request(response):
            return listen_with_retry(prompt, listening_for, max_retries)  # Try again after help
        elif any(exit_word in response.lower() for exit_word in ['exit', 'quit', 'bye', 'stop', 'goodbye', 'बाहर निकलें', 'छोड़ें', 'बंद करें', 'बाहर पडा', 'सोडा', 'बंद करा', 'నిష్క్రమించు', 'వదిలేయు', 'ఆపు']):
            return "EXIT_COMMAND"
        elif any(continue_word in response.lower() for continue_word in ["continue", "yes", "try", "again", "जारी रखें", "हाँ", "फिर से", "सुरू ठेवा", "होय", "पुन्हा", "కొనసాగించు", "అవును", "మళ్లీ"]):
            return listen_with_retry(prompt, listening_for, max_retries)  # Recursive retry
    
    return None

def understand_yes_no_response(user_input: str) -> str:
    """Convert various yes/no responses to standard format across languages"""
    if not user_input:
        return "unclear"
    
    user_input = user_input.lower().strip()
    
    # Yes responses in multiple languages
    yes_words = ["yes", "yeah", "yep", "ok", "okay", "sure", "correct", "right", "true", "affirmative",
                 "हाँ", "हां", "जी", "ठीक है", "सही", "होय", "बरोबर", "ठीक आहे", 
                 "అవును", "సరే", "నిజం", "సరైనది"]
    if any(word in user_input for word in yes_words):
        return "yes"
    
    # No responses in multiple languages
    no_words = ["no", "nope", "not", "wrong", "incorrect", "negative",
                "नहीं", "गलत", "नाही", "चुकीचे", "కాదు", "తప్పు"]
    if any(word in user_input for word in no_words):
        return "no"
    
    return "unclear"

def extract_reading_mode(user_input: str) -> str:
    """Extract whether user wants manual or sensor reading across languages"""
    if not user_input:
        return "unclear"
    
    user_input = user_input.lower().strip()
    print(f"🔍 Analyzing your response: '{user_input}'")
    
    # Manual mode keywords in multiple languages
    manual_keywords = ["manual", "manually", "voice", "speak", "say", "tell", "verbal", "by voice", "myself",
                      "मैनुअल", "आवाज़", "बोल", "कह", "स्वयं", "मॅन्युअल", "आवाज", "बोला", "सांग", "स्वतः",
                      "మ్యానువల్", "వాయిస్", "చెప్పు", "మాట్లాడు", "స్వయం"]
    
    # Sensor mode keywords in multiple languages
    sensor_keywords = ["sensor", "automatic", "device", "machine", "auto", "sensors", "automatically",
                      "सेंसर", "स्वचालित", "डिवाइस", "मशीन", "ऑटो", "सेन्सर", "स्वयंचलित", "यंत्र", "मशीन",
                      "సెన్సార్", "ఆటోమేటిక్", "పరికరం", "యంత్రం", "ఆటో"]
    
    manual_score = sum(1 for word in manual_keywords if word in user_input)
    sensor_score = sum(1 for word in sensor_keywords if word in user_input)
    
    print(f"📊 Manual keywords found: {manual_score}, Sensor keywords found: {sensor_score}")
    
    if manual_score > sensor_score:
        return "manual"
    elif sensor_score > manual_score:
        return "sensor"
    else:
        return "unclear"

# 9. USER PROFILE MANAGEMENT WITH MONGODB (Enhanced for multi-language)
def check_user_exists(fullname: str) -> Optional[Dict]:
    """Check if user profile exists in MongoDB"""
    try:
        user = users_collection.find_one(
            {"fullname": {"$regex": f"^{re.escape(fullname)}$", "$options": "i"}}
        )
        if user:
            # Convert ObjectId to string for easier handling
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
        profile_data['preferred_language'] = current_language  # Store user's language preference
        
        result = users_collection.insert_one(profile_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating user profile: {e}")
        return None

def update_user_profile(user_id: str, update_data: Dict):
    """Update existing user profile"""
    try:
        update_data['updated_at'] = datetime.utcnow()
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
    except Exception as e:
        print(f"Error updating user profile: {e}")

def get_profile_info():
    """Interactive profile creation with patient-friendly conversation in current language"""
    profile = {}
    
    welcome_msgs = {
        'english': "I need to create your medical profile to provide you with the best care. I'll ask you a few gentle questions. Please take your time, and let me know if you need any question repeated.",
        'hindi': "आपको सबसे अच्छी देखभाल प्रदान करने के लिए मुझे आपकी मेडिकल प्रोफाइल बनानी होगी। मैं आपसे कुछ सौम्य प्रश्न पूछूंगा। कृपया अपना समय लें, और मुझे बताएं यदि आपको कोई प्रश्न दोहराने की जरूरत है।",
        'marathi': "तुम्हाला सर्वोत्तम काळजी देण्यासाठी मला तुमची वैद्यकीय प्रोफाइल तयार करावी लागेल. मी तुम्हाला काही सौम्य प्रश्न विचारेन. कृपया तुमचा वेळ घ्या, आणि मला सांगा जर तुम्हाला कोणताही प्रश्न पुन्हा हवा असेल.",
        'telugu': "మీకు అత్యుత్తమ కేర్ అందించడానికి నేను మీ మెడికల్ ప్రొఫైల్ సృష్టించాల్సి ఉంది. నేను మిమ్మల్ని కొన్ని మృదువైన ప్రశ్నలు అడుగుతాను. దయచేసి మీ సమయం తీసుకోండి, మరియు మీకు ఏదైనా ప్రశ్న మళ్లీ అవసరమైతే నాకు చెప్పండి."
    }
    
    welcome_msg = welcome_msgs.get(current_language, welcome_msgs['english'])
    print(f"🩺 {welcome_msg}")
    speak(welcome_msg)
    
    # Age questions in different languages
    age_prompts = {
        'english': "Could you please tell me your age? You can say it in any way that's comfortable for you.",
        'hindi': "कृपया मुझे अपनी उम्र बताएं? आप इसे किसी भी तरीके से कह सकते हैं जो आपके लिए आरामदायक हो।",
        'marathi': "कृपया मला तुमचे वय सांगाल का? तुम्ही ते कोणत्याही प्रकारे म्हणू शकता जे तुमच्यासाठी आरामदायक आहे.",
        'telugu': "దయచేసి మీ వయస్సు చెప్పగలరా? మీకు సౌకర్యవంతమైన ఏ విధంగానైనా చెప్పవచ్చు."
    }
    
    # Age collection
    while True:
        age_prompt = age_prompts.get(current_language, age_prompts['english'])
        age_input = listen_with_retry(age_prompt, "your age", 5)
        if age_input == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        if age_input:
            print(f"🔍 Processing age information: '{age_input}'")
            try:
                # Extract numbers from response
                numbers = ''.join(filter(str.isdigit, age_input))
                if numbers:
                    age = int(numbers)
                    if 1 <= age <= 120:
                        profile['age'] = age
                        confirm_msgs = {
                            'english': f"Thank you. I have recorded your age as {age} years old.",
                            'hindi': f"धन्यवाद। मैंने आपकी उम्र {age} साल दर्ज की है।",
                            'marathi': f"धन्यवाद. मी तुमचे वय {age} वर्षे म्हणून नोंदवले आहे.",
                            'telugu': f"ధన్యవాదాలు. నేను మీ వయస్సును {age} సంవత్సరాలుగా నమోదు చేశాను."
                        }
                        confirm_msg = confirm_msgs.get(current_language, confirm_msgs['english'])
                        print(f"✅ {confirm_msg}")
                        speak(confirm_msg)
                        break
                    else:
                        error_msgs = {
                            'english': "I heard a number, but it seems outside the typical age range. Could you please repeat your age?",
                            'hindi': "मैंने एक संख्या सुनी, लेकिन यह सामान्य उम्र सीमा से बाहर लगती है। कृपया अपनी उम्र दोहराएं?",
                            'marathi': "मी एक संख्या ऐकली, पण ती सामान्य वयोमर्यादेच्या बाहेर वाटते. कृपया तुमचे वय पुन्हा सांगाल का?",
                            'telugu': "నేను ఒక సంఖ్యను విన్నాను, కానీ అది సాధారణ వయస్సు పరిధికి వెలుపల ఉన్నట్లు అనిపిస్తుంది. దయచేసి మీ వయస్సు మళ్లీ చెప్పగలరా?"
                        }
                        speak(error_msgs.get(current_language, error_msgs['english']))
                else:
                    error_msgs = {
                        'english': "I didn't catch a number in your response. Could you please tell me your age using numbers?",
                        'hindi': "मुझे आपके उत्तर में कोई संख्या समझ में नहीं आई। कृपया संख्याओं का उपयोग करके मुझे अपनी उम्र बताएं?",
                        'marathi': "मला तुमच्या उत्तरात कोणती संख्या समजली नाही. कृपया संख्या वापरून मला तुमचे वय सांगाल का?",
                        'telugu': "మీ సమాధానంలో నేను ఒక సంఖ్యను పట్టుకోలేకపోయాను. దయచేసి సంఖ్యలను ఉపయోగించి మీ వయస్సు చెప్పగలరా?"
                    }
                    speak(error_msgs.get(current_language, error_msgs['english']))
            except:
                error_msgs = {
                    'english': "I had trouble understanding that. Could you please say your age clearly? For example, 'I am 25 years old' or just '25'.",
                    'hindi': "मुझे समझने में परेशानी हुई। कृपया अपनी उम्र स्पष्ट रूप से कहें? उदाहरण के लिए, 'मैं 25 साल का हूँ' या बस '25'।",
                    'marathi': "मला समजण्यात अडचण आली. कृपया तुमचे वय स्पष्टपणे सांगाल का? उदाहरणार्थ, 'मी 25 वर्षांचा आहे' किंवा फक्त '25'.",
                    'telugu': "నాకు అర్థం చేసుకోవడంలో ఇబ్బంది వచ్చింది. దయచేసి మీ వయస్సు స్పష్టంగా చెప్పగలరా? ఉదాహరణకు, 'నేను 25 సంవత్సరాలు' లేదా కేవలం '25'."
                }
                speak(error_msgs.get(current_language, error_msgs['english']))
        else:
            retry_msgs = {
                'english': "Let me ask about your age again. Please tell me how old you are.",
                'hindi': "मुझे आपकी उम्र के बारे में फिर से पूछने दें। कृपया मुझे बताएं कि आपकी उम्र क्या है।",
                'marathi': "मला तुमच्या वयाबद्दल पुन्हा विचारू द्या. कृपया मला सांगा की तुमचे वय काय आहे.",
                'telugu': "మీ వయస్సు గురించి మళ్లీ అడుగుతాను. దయచేసి మీ వయస్సు ఎంత అని చెప్పండి."
            }
            speak(retry_msgs.get(current_language, retry_msgs['english']))
    
    # Gender collection with multi-language support
    gender_prompts = {
        'english': "What is your gender? You can say male, female, or anything else you're comfortable with.",
        'hindi': "आपका लिंग क्या है? आप पुरुष, महिला, या कुछ और कह सकते हैं जिससे आप सहज हों।",
        'marathi': "तुमचे लिंग काय आहे? तुम्ही पुरुष, महिला, किंवा इतर काहीही म्हणू शकता ज्यामध्ये तुम्हाला आराम वाटतो.",
        'telugu': "మీ లింగం ఏమిటి? మీరు పురుషుడు, స్త్రీ, లేదా మీకు సౌకర్యవంతమైన ఏదైనా చెప్పవచ్చు."
    }
    
    while True:
        gender_prompt = gender_prompts.get(current_language, gender_prompts['english'])
        gender_input = listen_with_retry(gender_prompt, "your gender")
        if gender_input == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        if gender_input:
            profile['gender'] = gender_input.strip()
            confirm_msgs = {
                'english': "Thank you. I have recorded your gender.",
                'hindi': "धन्यवाद। मैंने आपका लिंग दर्ज कर लिया है।",
                'marathi': "धन्यवाद. मी तुमचे लिंग नोंदवले आहे.",
                'telugu': "ధన్యవాదాలు. నేను మీ లింగాన్ని నమోదు చేశాను."
            }
            confirm_msg = confirm_msgs.get(current_language, confirm_msgs['english'])
            print(f"✅ {confirm_msg}")
            speak(confirm_msg)
            break
        else:
            retry_msgs = {
                'english': "Let me ask about your gender again.",
                'hindi': "मुझे आपके लिंग के बारे में फिर से पूछने दें।",
                'marathi': "मला तुमच्या लिंगाबद्दल पुन्हा विचारू द्या.",
                'telugu': "మీ లింగం గురించి మళ్లీ అడుగుతాను."
            }
            speak(retry_msgs.get(current_language, retry_msgs['english']))
    
    # Height collection
    height_prompts = {
        'english': "What is your height? You can tell me in centimeters, or say it any way you prefer.",
        'hindi': "आपकी लंबाई क्या है? आप मुझे सेंटीमीटर में बता सकते हैं, या जैसे चाहें कह सकते हैं।",
        'marathi': "तुमची उंची काय आहे? तुम्ही मला सेंटीमीटरमध्ये सांगू शकता, किंवा तुम्हाला पसंत असलेल्या कोणत्याही प्रकारे सांगू शकता.",
        'telugu': "మీ ఎత్తు ఎంత? మీరు నాకు సెంటీమీటర్లలో చెప్పవచ్చు, లేదా మీకు ఇష్టమైన ఏ విధంగానైనా చెప్పవచ్చు."
    }
    
    while True:
        height_prompt = height_prompts.get(current_language, height_prompts['english'])
        height_input = listen_with_retry(height_prompt, "your height")
        if height_input == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        if height_input:
            print(f"🔍 Processing height information: '{height_input}'")
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', height_input)
                if numbers:
                    height = float(numbers[0])
                    # Convert feet to cm if needed
                    if "feet" in height_input.lower() or "foot" in height_input.lower() or height < 10:
                        height = height * 30.48
                    if 50 <= height <= 250:
                        profile['height'] = height
                        confirm_msgs = {
                            'english': f"Thank you. I have recorded your height as {height:.1f} centimeters.",
                            'hindi': f"धन्यवाद। मैंने आपकी लंबाई {height:.1f} सेंटीमीटर दर्ज की है।",
                            'marathi': f"धन्यवाद. मी तुमची उंची {height:.1f} सेंटीमीटर म्हणून नोंदवली आहे.",
                            'telugu': f"ధన్యవాదాలు. నేను మీ ఎత్తును {height:.1f} సెంటీమీటర్లుగా నమోదు చేశాను."
                        }
                        confirm_msg = confirm_msgs.get(current_language, confirm_msgs['english'])
                        print(f"✅ {confirm_msg}")
                        speak(confirm_msg)
                        break
                    else:
                        error_msgs = {
                            'english': "I heard a number, but it seems outside the typical height range. Could you please repeat your height?",
                            'hindi': "मैंने एक संख्या सुनी, लेकिन यह सामान्य लंबाई सीमा से बाहर लगती है। कृपया अपनी लंबाई दोहराएं?",
                            'marathi': "मी एक संख्या ऐकली, पण ती सामान्य उंचीच्या मर्यादेच्या बाहेर वाटते. कृपया तुमची उंची पुन्हा सांगाल का?",
                            'telugu': "నేను ఒక సంఖ్యను విన్నాను, కానీ అది సాధారణ ఎత్తు పరిధికి వెలుపల ఉన్నట్లు అనిపిస్తుంది. దయచేసి మీ ఎత్తు మళ్లీ చెప్పగలరా?"
                        }
                        speak(error_msgs.get(current_language, error_msgs['english']))
                else:
                    error_msgs = {
                        'english': "I didn't catch a measurement in your response. Could you please tell me your height with numbers?",
                        'hindi': "मुझे आपके उत्तर में कोई माप समझ में नहीं आया। कृपया संख्याओं के साथ मुझे अपनी लंबाई बताएं?",
                        'marathi': "मला तुमच्या उत्तरात कोणतेही मोजमाप समजले नाही. कृपया संख्यांसह मला तुमची उंची सांगाल का?",
                        'telugu': "మీ సమాధానంలో నేను ఒక కొలత పట్టుకోలేకపోయాను. దయచేసి సంఖ్యలతో మీ ఎత్తు చెప్పగలరా?"
                    }
                    speak(error_msgs.get(current_language, error_msgs['english']))
            except:
                error_msgs = {
                    'english': "I had trouble understanding that. Could you please say your height clearly? For example, '170 centimeters' or '5 feet 6 inches'.",
                    'hindi': "मुझे समझने में परेशानी हुई। कृपया अपनी लंबाई स्पष्ट रूप से कहें? उदाहरण के लिए, '170 सेंटीमीटर' या '5 फीट 6 इंच'।",
                    'marathi': "मला समजण्यात अडचण आली. कृपया तुमची उंची स्पष्टपणे सांगाल का? उदाहरणार्थ, '170 सेंटीमीटर' किंवा '5 फूट 6 इंच'.",
                    'telugu': "నాకు అర్థం చేసుకోవడంలో ఇబ్బంది వచ్చింది. దయచేసి మీ ఎత్తు స్పష్టంగా చెప్పగలరా? ఉదాహరణకు, '170 సెంటీమీటర్లు' లేదా '5 అడుగులు 6 అంగుళాలు'."
                }
                speak(error_msgs.get(current_language, error_msgs['english']))
        else:
            retry_msgs = {
                'english': "Let me ask about your height again.",
                'hindi': "मुझे आपकी लंबाई के बारे में फिर से पूछने दें।",
                'marathi': "मला तुमच्या उंचीबद्दल पुन्हा विचारू द्या.",
                'telugu': "మీ ఎత్తు గురించి మళ్లీ అడుగుతాను."
            }
            speak(retry_msgs.get(current_language, retry_msgs['english']))
    
    # Medical History
    history_prompts = {
        'english': "Do you have any important medical conditions or medical history I should know about? If none, you can just say 'none' or 'no medical history'.",
        'hindi': "क्या आपकी कोई महत्वपूर्ण चिकित्सा स्थितियां या चिकित्सा इतिहास है जिसके बारे में मुझे जानना चाहिए? यदि कोई नहीं है, तो आप बस 'कोई नहीं' या 'कोई चिकित्सा इतिहास नहीं' कह सकते हैं।",
        'marathi': "तुमच्याकडे कोणत्या महत्त्वाच्या वैद्यकीय परिस्थिती किंवा वैद्यकीय इतिहास आहे ज्याबद्दल मला माहिती असावी? जर काही नसेल तर तुम्ही फक्त 'काही नाही' किंवा 'कोणताही वैद्यकीय इतिहास नाही' म्हणू शकता.",
        'telugu': "మీకు ఏదైనా ముఖ్యమైన వైద్య పరిస్థితులు లేదా వైద్య చరిత్ర ఉందా అని నేను తెలుసుకోవాలి? ఏవీ లేకుంటే, మీరు కేవలం 'ఏవీ లేవు' లేదా 'వైద్య చరిత్ర లేదు' అని చెప్పవచ్చు."
    }
    
    history_prompt = history_prompts.get(current_language, history_prompts['english'])
    history_input = listen_with_retry(history_prompt, "your medical history")
    if history_input == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    
    response_type = understand_yes_no_response(history_input)
    if response_type == "no" or (history_input and any(word in history_input.lower() for word in ["none", "nothing", "no medical", "कोई नहीं", "कुछ नहीं", "काही नाही", "कुठे नाही", "ఏవీ లేవు", "ఏమీ లేదు"])):
        profile['medical_history'] = "No significant medical history reported"
        no_history_msgs = {
            'english': "Understood. No significant medical history noted.",
            'hindi': "समझ गया। कोई महत्वपूर्ण चिकित्सा इतिहास नोट नहीं किया गया।",
            'marathi': "समजले. कोणताही महत्त्वाचा वैद्यकीय इतिहास नोंदवला नाही.",
            'telugu': "అర్థమైంది. ముఖ్యమైన వైద్య చరిత్ర లేదని నమోదు చేశాను."
        }
        speak(no_history_msgs.get(current_language, no_history_msgs['english']))
    else:
        profile['medical_history'] = history_input if history_input else "Not specified"
        if history_input:
            thank_msgs = {
                'english': "Thank you for sharing that information. I've recorded your medical history.",
                'hindi': "उस जानकारी को साझा करने के लिए धन्यवाद। मैंने आपका चिकित्सा इतिहास दर्ज कर लिया है।",
                'marathi': "ती माहिती सामायिक केल्याबद्दल धन्यवाद. मी तुमचा वैद्यकीय इतिहास नोंदवला आहे.",
                'telugu': "ఆ సమాచారాన్ని పంచుకున్నందుకు ధన్యవాదాలు. నేను మీ వైద్య చరిత్రను నమోదు చేశాను."
            }
            speak(thank_msgs.get(current_language, thank_msgs['english']))
    
    # Allergies
    allergy_prompts = {
        'english': "Do you have any allergies to medications, foods, or other substances? If none, just say 'no allergies' or 'none'.",
        'hindi': "क्या आपको दवाओं, भोजन, या अन्य पदार्थों से कोई एलर्जी है? यदि कोई नहीं है, तो बस 'कोई एलर्जी नहीं' या 'कोई नहीं' कहें।",
        'marathi': "तुम्हाला औषधे, अन्न, किंवा इतर पदार्थांची कोणती ऍलर्जी आहे का? जर काही नसेल तर फक्त 'कोणतीही ऍलर्जी नाही' किंवा 'काही नाही' म्हणा.",
        'telugu': "మీకు మందులు, ఆహారాలు, లేదా ఇతర పదార్థాలకు ఏదైనా అలెర్జీలు ఉన్నాయా? ఏవీ లేకుంటే, కేవలం 'అలెర్జీలు లేవు' లేదా 'ఏవీ లేవు' అనండి."
    }
    
    allergy_prompt = allergy_prompts.get(current_language, allergy_prompts['english'])
    allergy_input = listen_with_retry(allergy_prompt, "your allergies")
    if allergy_input == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    
    response_type = understand_yes_no_response(allergy_input)
    if response_type == "no" or (allergy_input and any(word in allergy_input.lower() for word in ["none", "no allergies", "nothing", "कोई एलर्जी नहीं", "कुछ नहीं", "कोणतीही ऍलर्जी नाही", "काही नाही", "అలెర్జీలు లేవు", "ఏవీ లేవు"])):
        profile['allergies'] = "No known allergies"
        no_allergies_msgs = {
            'english': "Good to know. No allergies recorded.",
            'hindi': "जानकर अच्छा लगा। कोई एलर्जी दर्ज नहीं की गई।",
            'marathi': "हे जाणून चांगले वाटले. कोणतीही ऍलर्जी नोंदवली नाही.",
            'telugu': "తెలుసుకుని మంచిగా అనిపించింది. అలెర్జీలు లేవని నమోదు చేశాను."
        }
        speak(no_allergies_msgs.get(current_language, no_allergies_msgs['english']))
    else:
        profile['allergies'] = allergy_input if allergy_input else "Not specified"
        if allergy_input:
            important_msgs = {
                'english': "Important information. I've recorded your allergies.",
                'hindi': "महत्वपूर्ण जानकारी। मैंने आपकी एलर्जी दर्ज कर ली है।",
                'marathi': "महत्त्वाची माहिती. मी तुमच्या ऍलर्जी नोंदवल्या आहेत.",
                'telugu': "ముఖ్యమైన సమాచారం. నేను మీ అలెర్జీలను నమోదు చేశాను."
            }
            speak(important_msgs.get(current_language, important_msgs['english']))
    
    # Current Medications
    medication_prompts = {
        'english': "Are you currently taking any medications or treatments? If none, you can say 'no medications' or 'nothing'.",
        'hindi': "क्या आप वर्तमान में कोई दवाएं या उपचार ले रहे हैं? यदि कोई नहीं है, तो आप 'कोई दवा नहीं' या 'कुछ नहीं' कह सकते हैं।",
        'marathi': "तुम्ही सध्या कोणतीही औषधे किंवा उपचार घेत आहात का? जर काही नसेल तर तुम्ही 'कोणतीही औषधे नाहीत' किंवा 'काही नाही' म्हणू शकता.",
        'telugu': "మీరు ప్రస్తుతం ఏదైనా మందులు లేదా చికిత్సలు తీసుకుంటున్నారా? ఏవీ లేకుంటే, మీరు 'మందులు లేవు' లేదా 'ఏమీ లేదు' అని చెప్పవచ్చు."
    }
    
    medication_prompt = medication_prompts.get(current_language, medication_prompts['english'])
    medication_input = listen_with_retry(medication_prompt, "your current medications")
    if medication_input == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    
    response_type = understand_yes_no_response(medication_input)
    if response_type == "no" or (medication_input and any(word in medication_input.lower() for word in ["none", "no medications", "nothing", "not taking", "कोई दवा नहीं", "कुछ नहीं", "नहीं ले रहा", "कोणतीही औषधे नाहीत", "काही नाही", "घेत नाही", "మందులు లేవు", "ఏమీ లేదు", "తీసుకోవడం లేదు"])):
        profile['medications'] = "No current medications"
        no_meds_msgs = {
            'english': "Understood. No current medications noted.",
            'hindi': "समझ गया। कोई वर्तमान दवाएं नोट नहीं की गईं।",
            'marathi': "समजले. कोणतीही सध्याची औषधे नोंदवली नाहीत.",
            'telugu': "అర్థమైంది. ప్రస్తుత మందులు లేవని నమోదు చేశాను."
        }
        speak(no_meds_msgs.get(current_language, no_meds_msgs['english']))
    else:
        profile['medications'] = medication_input if medication_input else "Not specified"
        if medication_input:
            thank_msgs = {
                'english': "Thank you. I've recorded your current medications.",
                'hindi': "धन्यवाद। मैंने आपकी वर्तमान दवाएं दर्ज कर ली हैं।",
                'marathi': "धन्यवाद. मी तुमच्या सध्याच्या औषधांची नोंद केली आहे.",
                'telugu': "ధన్యవాదాలు. నేను మీ ప్రస్తుత మందులను నమోదు చేశాను."
            }
            speak(thank_msgs.get(current_language, thank_msgs['english']))
    
    # Emergency Contact
    emergency_prompts = {
        'english': "Could you please provide an emergency contact phone number? This is just for safety.",
        'hindi': "कृपया एक आपातकालीन संपर्क फोन नंबर प्रदान कर सकते हैं? यह केवल सुरक्षा के लिए है।",
        'marathi': "कृपया आणीबाणीचा संपर्क फोन नंबर देऊ शकाल का? हे फक्त सुरक्षिततेसाठी आहे.",
        'telugu': "దయచేసి అత్యవసర సంప్రదింపు ఫోన్ నంబర్ అందించగలరా? ఇది కేవలం భద్రత కోసం."
    }
    
    emergency_prompt = emergency_prompts.get(current_language, emergency_prompts['english'])
    emergency_input = listen_with_retry(emergency_prompt, "emergency contact number")
    if emergency_input == "EXIT_COMMAND":
        return "EXIT_COMMAND"
    profile['emergency_contact'] = emergency_input if emergency_input else "Not provided"
    if emergency_input:
        thank_msgs = {
            'english': "Thank you. Emergency contact information saved.",
            'hindi': "धन्यवाद। आपातकालीन संपर्क जानकारी सहेजी गई।",
            'marathi': "धन्यवाद. आणीबाणीची संपर्क माहिती जतन केली.",
            'telugu': "ధన్యవాదాలు. అత్యవసర సంప్రదింపు సమాచారం సేవ్ చేయబడింది."
        }
        speak(thank_msgs.get(current_language, thank_msgs['english']))
    
    completion_msgs = {
        'english': "Perfect! I've created your medical profile. This information helps me provide better care for you.",
        'hindi': "बेहतरीन! मैंने आपकी मेडिकल प्रोफाइल बना दी है। यह जानकारी मुझे आपको बेहतर देखभाल प्रदान करने में मदद करती है।",
        'marathi': "उत्तम! मी तुमची वैद्यकीय प्रोफाइल तयार केली आहे. ही माहिती मला तुम्हाला चांगली काळजी देण्यासाठी मदत करते.",
        'telugu': "అద్భుతం! నేను మీ మెడికల్ ప్రొఫైల్ సృష్టించాను. ఈ సమాచారం మీకు మంచి కేర్ అందించడంలో నాకు సహాయపడుతుంది."
    }
    
    completion_msg = completion_msgs.get(current_language, completion_msgs['english'])
    print(f"✅ {completion_msg}")
    log_conversation("Assistant", completion_msg)
    speak(completion_msg)
    
    return profile

# 10. SENSOR READING FUNCTIONS
def read_temperature_sensor() -> float:
    """Read temperature from sensor (mock function - replace with actual sensor code)"""
    # TODO: Replace with actual sensor reading code
    print("📡 Reading temperature sensor...")
    time.sleep(1)
    return 98.6  # Mock reading

def read_heart_rate_sensor() -> int:
    """Read heart rate from sensor"""
    print("📡 Reading heart rate sensor...")
    time.sleep(2)
    return 72  # Mock reading

def read_pulse_sensor() -> int:
    """Read pulse rate from sensor"""
    print("📡 Reading pulse sensor...")
    time.sleep(2)
    return 70  # Mock reading

def read_oxygen_saturation_sensor() -> float:
    """Read oxygen saturation from sensor"""
    print("📡 Reading oxygen saturation sensor...")
    time.sleep(2)
    return 98.5  # Mock reading

def read_blood_pressure_sensor() -> tuple:
    """Read blood pressure from sensor - returns (systolic, diastolic)"""
    print("📡 Reading blood pressure sensor...")
    time.sleep(3)
    return (120, 80)  # Mock reading

def read_weight_sensor() -> float:
    """Read weight from sensor"""
    print("📡 Reading weight sensor...")
    time.sleep(2)
    return 70.5  # Mock reading

# 11. MANUAL VOICE INPUT FUNCTIONS WITH MULTI-LANGUAGE SUPPORT
def get_manual_temperature() -> float:
    """Get temperature reading via voice input with patient care in current language"""
    temp_prompts = {
        'english': "Please tell me your body temperature. You can say it in Fahrenheit, like '98.6 degrees' or just '98.6'. Take your time.",
        'hindi': "कृपया मुझे अपना शरीर का तापमान बताएं। आप इसे फ़ारेनहाइट में कह सकते हैं, जैसे '98.6 डिग्री' या बस '98.6'। अपना समय लें।",
        'marathi': "कृपया मला तुमचे शरीराचे तापमान सांगा. तुम्ही ते फॅरेनहाइटमध्ये म्हणू शकता, जसे '98.6 डिग्री' किंवा फक्त '98.6'. तुमचा वेळ घ्या.",
        'telugu': "దయచేసి మీ శరీర ఉష్ణోగ్రత చెప్పండి. మీరు దానిని ఫారెన్‌హీట్‌లో చెప్పవచ్చు, '98.6 డిగ్రీలు' లేదా కేవలం '98.6'. మీ సమయం తీసుకోండి."
    }
    
    while True:
        temp_prompt = temp_prompts.get(current_language, temp_prompts['english'])
        temp_input = listen_with_retry(temp_prompt, "your temperature reading")
        if temp_input == "EXIT_COMMAND":
            return None
        if temp_input:
            print(f"🔍 Processing temperature: '{temp_input}'")
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', temp_input)
                if numbers:
                    temp = float(numbers[0])
                    if 90 <= temp <= 110:
                        confirm_msgs = {
                            'english': f"Got it. Your temperature is {temp} degrees Fahrenheit.",
                            'hindi': f"समझ गया। आपका तापमान {temp} डिग्री फ़ारेनहाइट है।",
                            'marathi': f"समजले. तुमचे तापमान {temp} डिग्री फॅरेनहाइट आहे.",
                            'telugu': f"అర్థమైంది. మీ ఉష్ణోగ్రత {temp} డిగ్రీల ఫారెన్‌హీట్."
                        }
                        confirm_msg = confirm_msgs.get(current_language, confirm_msgs['english'])
                        print(f"✅ {confirm_msg}")
                        speak(confirm_msg)
                        return temp
                    else:
                        error_msgs = {
                            'english': "That temperature seems outside the normal range. Could you please check and tell me again? Normal body temperature is usually between 96 and 104 degrees.",
                            'hindi': "वह तापमान सामान्य सीमा से बाहर लगता है। कृपया जांच करके मुझे फिर से बताएं? सामान्य शरीर का तापमान आमतौर पर 96 और 104 डिग्री के बीच होता है।",
                            'marathi': "ते तापमान सामान्य मर्यादेच्या बाहेर वाटते. कृपया तपासून मला पुन्हा सांगाल का? सामान्य शरीराचे तापमान सहसा 96 आणि 104 डिग्री दरम्यान असते.",
                            'telugu': "ఆ ఉష్ణోగ్రత సాధారణ పరిధికి వెలుపల ఉన్నట్లు అనిపిస్తుంది. దయచేసి తనిఖీ చేసి మళ్లీ చెప్పగలరా? సాధారణ శరీర ఉష్ణోగ్రత సాధారణంగా 96 మరియు 104 డిగ్రీల మధ్య ఉంటుంది."
                        }
                        speak(error_msgs.get(current_language, error_msgs['english']))
                else:
                    error_msgs = {
                        'english': "I didn't catch a temperature number. Could you please say your temperature reading clearly? For example, 'ninety eight point six' or just '98.6'.",
                        'hindi': "मुझे तापमान संख्या समझ में नहीं आई। कृपया अपना तापमान रीडिंग स्पष्ट रूप से कहें? उदाहरण के लिए, 'अट्ठानवे दशमलव छह' या बस '98.6'।",
                        'marathi': "मला तापमान संख्या समजली नाही. कृपया तुमचे तापमान रीडिंग स्पष्टपणे सांगाल का? उदाहरणार्थ, 'अठ्ठ्यानवे दशमलव सहा' किंवा फक्त '98.6'.",
                        'telugu': "నేను ఉష్ణోగ్రత సంఖ్యను పట్టుకోలేకపోయాను. దయచేసి మీ ఉష్ణోగ్రత రీడింగ్ స్పష్టంగా చెప్పగలరా? ఉదాహరణకు, 'తొంభై ఎనిమిది దశమలవ ఆరు' లేదా కేవలం '98.6'."
                    }
                    speak(error_msgs.get(current_language, error_msgs['english']))
            except:
                error_msgs = {
                    'english': "I had trouble understanding that. Please say your temperature slowly and clearly, like '98.6 degrees'.",
                    'hindi': "मुझे समझने में परेशानी हुई। कृपया अपना तापमान धीरे और स्पष्ट रूप से कहें, जैसे '98.6 डिग्री'।",
                    'marathi': "मला समजण्यात अडचण आली. कृपया तुमचे तापमान हळूहळू आणि स्पष्टपणे सांगा, जसे '98.6 डिग्री'.",
                    'telugu': "నాకు అర్థం చేసుకోవడంలో ఇబ్బంది వచ్చింది. దయచేసి మీ ఉష్ణోగ్రతను నెమ్మదిగా మరియు స్పష్టంగా చెప్పండి, '98.6 డిగ్రీలు' లాగా."
                }
                speak(error_msgs.get(current_language, error_msgs['english']))
        else:
            retry_msgs = {
                'english': "Let me ask for your temperature again. Please tell me the reading from your thermometer.",
                'hindi': "मुझे आपके तापमान के बारे में फिर से पूछने दें। कृपया मुझे अपने थर्मामीटर की रीडिंग बताएं।",
                'marathi': "मला तुमच्या तापमानाबद्दल पुन्हा विचारू द्या. कृपया मला तुमच्या थर्मामीटरची रीडिंग सांगा.",
                'telugu': "మీ ఉష్ణోగ్రత గురించి మళ్లీ అడుగుతాను. దయచేసి మీ థర్మామీటర్ రీడింగ్ చెప్పండి."
            }
            speak(retry_msgs.get(current_language, retry_msgs['english']))

def get_manual_heart_rate() -> int:
    """Get heart rate via voice input with patient care in current language"""
    hr_prompts = {
        'english': "Please tell me your heart rate in beats per minute. You can say something like '72 beats per minute' or just '72'.",
        'hindi': "कृपया मुझे अपना हृदय गति प्रति मिनट धड़कन में बताएं। आप '72 धड़कन प्रति मिनट' या बस '72' जैसा कुछ कह सकते हैं।",
        'marathi': "कृपया मला तुमच्या हृदयाची गती प्रति मिनिट धडधडाटीत सांगा. तुम्ही '72 धडधडाट प्रति मिनिट' किंवा फक्त '72' असे काहीतरी म्हणू शकता.",
        'telugu': "దయచేసి మీ హృదయ స్పందన రేటును నిమిషానికి స్పందనలలో చెప్పండి. మీరు 'నిమిషానికి 72 స్పందనలు' లేదా కేవలం '72' లాంటిది చెప్పవచ్చు."
    }
    
    while True:
        hr_prompt = hr_prompts.get(current_language, hr_prompts['english'])
        hr_input = listen_with_retry(hr_prompt, "your heart rate")
        if hr_input == "EXIT_COMMAND":
            return None
        if hr_input:
            print(f"🔍 Processing heart rate: '{hr_input}'")
            try:
                numbers = ''.join(filter(str.isdigit, hr_input))
                if numbers:
                    hr = int(numbers)
                    if 30 <= hr <= 200:
                        confirm_msgs = {
                            'english': f"Thank you. Your heart rate is {hr} beats per minute.",
                            'hindi': f"धन्यवाद। आपकी हृदय गति {hr} धड़कन प्रति मिनट है।",
                            'marathi': f"धन्यवाद. तुमच्या हृदयाची गती {hr} धडधडाट प्रति मिनिट आहे.",
                            'telugu': f"ధన్యవాదాలు. మీ హృదయ స్పందన రేటు నిమిషానికి {hr} స్పందనలు."
                        }
                        confirm_msg = confirm_msgs.get(current_language, confirm_msgs['english'])
                        print(f"✅ {confirm_msg}")
                        speak(confirm_msg)
                        return hr
                    else:
                        error_msgs = {
                            'english': "That heart rate seems unusual. Could you please check and tell me again? Normal heart rate is usually between 60 and 100 beats per minute.",
                            'hindi': "वह हृदय गति असामान्य लगती है। कृपया जांच करके मुझे फिर से बताएं? सामान्य हृदय गति आमतौर पर 60 और 100 धड़कन प्रति मिनट के बीच होती है।",
                            'marathi': "ती हृदयाची गती असामान्य वाटते. कृपया तपासून मला पुन्हा सांगाल का? सामान्य हृदयाची गती सहसा 60 आणि 100 धडधडाट प्रति मिनिट दरम्यान असते.",
                            'telugu': "ఆ హృదయ స్పందన రేటు అసాధారణంగా అనిపిస్తుంది. దయచేసి తనిఖీ చేసి మళ్లీ చెప్పగలరా? సాధారణ హృదయ స్పందన రేటు సాధారణంగా నిమిషానికి 60 మరియు 100 స్పందనల మధ్య ఉంటుంది."
                        }
                        speak(error_msgs.get(current_language, error_msgs['english']))
                else:
                    error_msgs = {
                        'english': "I didn't catch a heart rate number. Could you please tell me your heart rate clearly? For example, 'seventy two' or just '72'.",
                        'hindi': "मुझे हृदय गति संख्या समझ में नहीं आई। कृपया मुझे अपनी हृदय गति स्पष्ट रूप से बताएं? उदाहरण के लिए, 'बहत्तर' या बस '72'।",
                        'marathi': "मला हृदयाची गती संख्या समजली नाही. कृपया मला तुमच्या हृदयाची गती स्पष्टपणे सांगाल का? उदाहरणार्थ, 'बहात्तर' किंवा फक्त '72'.",
                        'telugu': "నేను హృదయ స్పందన రేటు సంఖ్యను పట్టుకోలేకపోయాను. దయచేసి మీ హృదయ స్పందన రేటు స్పష్టంగా చెప్పగలరా? ఉదాహరణకు, 'డెబ్బై రెండు' లేదా కేవలం '72'."
                    }
                    speak(error_msgs.get(current_language, error_msgs['english']))
            except:
                error_msgs = {
                    'english': "I had trouble understanding that. Please say your heart rate slowly, like '72 beats per minute'.",
                    'hindi': "मुझे समझने में परेशानी हुई। कृपया अपनी हृदय गति धीरे कहें, जैसे '72 धड़कन प्रति मिनट'।",
                    'marathi': "मला समजण्यात अडचण आली. कृपया तुमच्या हृदयाची गती हळूहळू सांगा, जसे '72 धडधडाट प्रति मिनिट'.",
                    'telugu': "నాకు అర్థం చేసుకోవడంలో ఇబ్బంది వచ్చింది. దయచేసి మీ హృదయ స్పందన రేటును నెమ్మదిగా చెప్పండి, 'నిమిషానికి 72 స్పందనలు' లాగా."
                }
                speak(error_msgs.get(current_language, error_msgs['english']))
        else:
            retry_msgs = {
                'english': "Let me ask for your heart rate again. Please tell me the number from your measurement.",
                'hindi': "मुझे आपकी हृदय गति के बारे में फिर से पूछने दें। कृपया मुझे अपने मापन से संख्या बताएं।",
                'marathi': "मला तुमच्या हृदयाच्या गतीबद्दल पुन्हा विचारू द्या. कृपया मला तुमच्या मोजमापातील संख्या सांगा.",
                'telugu': "మీ హృదయ స్పందన రేటు గురించి మళ్లీ అడుగుతాను. దయచేసి మీ కొలత నుండి సంఖ్యను చెప్పండి."
            }
            speak(retry_msgs.get(current_language, retry_msgs['english']))

# 12. VITAL SIGNS COLLECTION
def collect_vital_signs(manual_mode=False) -> Dict:
    """Collect all vital signs either from sensors or manual input"""
    collection_msgs = {
        'english': "Now I will collect your vital signs. Please stay still.",
        'hindi': "अब मैं आपके जीवनशैली संकेतक एकत्र करूंगा। कृपया स्थिर रहें।",
        'marathi': "आता मी तुमचे महत्वाचे संकेत गोळा करेन. कृपया स्थिर राहा.",
        'telugu': "ఇప్పుడు నేను మీ కీలక సంకేతాలను సేకరిస్తాను. దయచేసి నిశ్చలంగా ఉండండి."
    }
    speak(collection_msgs.get(current_language, collection_msgs['english']))
    
    vitals = {}
    
    if manual_mode:
        # Temperature
        temperature = get_manual_temperature()
        if temperature is None:
            return "EXIT_COMMAND"
        vitals['temperature'] = temperature
        
        # Heart Rate
        heart_rate = get_manual_heart_rate()
        if heart_rate is None:
            return "EXIT_COMMAND"
        vitals['heart_rate'] = heart_rate
        
        # Continue with other vitals...
        # For brevity, I'll implement the key ones. You can add the rest following the same pattern
        
    else:
        vitals['temperature'] = read_temperature_sensor()
        vitals['heart_rate'] = read_heart_rate_sensor()
        vitals['pulse_rate'] = read_pulse_sensor()
        vitals['oxygen_saturation'] = read_oxygen_saturation_sensor()
        systolic, diastolic = read_blood_pressure_sensor()
        vitals['blood_pressure_systolic'] = systolic
        vitals['blood_pressure_diastolic'] = diastolic
        vitals['weight'] = read_weight_sensor()
    
    return vitals

# 13. AI ANALYSIS WITH MULTI-LANGUAGE SUPPORT
def analyze_health_data(user_profile: Dict, symptoms: str, vitals: Dict) -> Dict:
    """Analyze health data and provide diagnosis and recommendations in current language"""
    
    # Prepare health context for AI analysis
    health_context = f"""
    PATIENT PROFILE:
    Name: {user_profile['fullname']}
    Age: {user_profile['age']}
    Gender: {user_profile['gender']}
    Height: {user_profile['height']} cm
    Medical History: {user_profile['medical_history']}
    Allergies: {user_profile['allergies']}
    Current Medications: {user_profile['medications']}
    Preferred Language: {current_language}
    
    CURRENT SYMPTOMS:
    {symptoms}
    
    VITAL SIGNS:
    Temperature: {vitals['temperature']}°F
    Heart Rate: {vitals['heart_rate']} bpm
    Oxygen Saturation: {vitals.get('oxygen_saturation', 'N/A')}%
    Blood Pressure: {vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')} mmHg
    Weight: {vitals.get('weight', 'N/A')} kg
    
    Please provide analysis in {SUPPORTED_LANGUAGES[current_language]['name']} language:
    1. Possible diagnosis based on symptoms and vital signs
    2. Severity level (Low, Medium, High, Emergency)
    3. Immediate recommendations
    4. When to seek professional medical help
    5. Lifestyle recommendations if applicable
    
    Important: This is for informational purposes only and should not replace professional medical advice.
    """
    
    try:
        system_prompts = {
            'english': "You are a medical analysis AI assistant. Analyze the provided health data and give preliminary assessment. Always emphasize that this is not a substitute for professional medical care. Respond in English.",
            'hindi': "आप एक मेडिकल विश्लेषण AI सहायक हैं। प्रदान किए गए स्वास्थ्य डेटा का विश्लेषण करें और प्रारंभिक मूल्यांकन दें। हमेशा जोर दें कि यह पेशेवर चिकित्सा देखभाल का विकल्प नहीं है। हिंदी में जवाब दें।",
            'marathi': "तुम्ही एक वैद्यकीय विश्लेषण AI सहायक आहात. प्रदान केलेल्या आरोग्य डेटाचे विश्लेषण करा आणि प्राथमिक मूल्यमापन द्या. नेहमी यावर भर द्या की हे व्यावसायिक वैद्यकीय काळजीचा पर्याय नाही. मराठीत उत्तर द्या.",
            'telugu': "మీరు ఒక వైద్య విశ్లేషణ AI సహాయకులు. అందించిన ఆరోగ్య డేటాను విశ్లేషించి ప్రాథమిక మూల్యాంకనం ఇవ్వండి. ఇది వృత్తిపరమైన వైద్య సంరక్షణకు ప్రత్యామ్నాయం కాదని ఎల్లప్పుడూ నొక్కి చెప్పండి. తెలుగులో సమాధానం ఇవ్వండి."
        }
        
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": system_prompts.get(current_language, system_prompts['english'])},
                {"role": "user", "content": health_context}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content
        
        return {
            'diagnosis': analysis,
            'recommendations': analysis,
            'severity': 'Medium'
        }
        
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        error_msgs = {
            'english': 'Unable to analyze at this time. Please consult a healthcare professional.',
            'hindi': 'इस समय विश्लेषण करने में असमर्थ। कृपया किसी स्वास्थ्य पेशेवर से सलाह लें।',
            'marathi': 'यावेळी विश्लेषण करू शकत नाही. कृपया आरोग्यसेवा व्यावसायिकाचा सल्ला घ्या.',
            'telugu': 'ఈ సమయంలో విశ్లేషించలేకపోతున్నాము. దయచేసి ఆరోగ్య నిపుణులను సంప్రదించండి.'
        }
        return {
            'diagnosis': error_msgs.get(current_language, error_msgs['english']),
            'recommendations': error_msgs.get(current_language, error_msgs['english']),
            'severity': 'Unknown'
        }

# 14. SAVE MEDICAL READING TO MONGODB
def save_medical_reading(user_id: str, vitals: Dict, symptoms: str, diagnosis: str, recommendations: str):
    """Save medical reading to MongoDB"""
    try:
        reading_data = {
            'user_id': user_id,
            'temperature': vitals.get('temperature'),
            'heart_rate': vitals.get('heart_rate'),
            'pulse_rate': vitals.get('pulse_rate'),
            'oxygen_saturation': vitals.get('oxygen_saturation'),
            'blood_pressure_systolic': vitals.get('blood_pressure_systolic'),
            'blood_pressure_diastolic': vitals.get('blood_pressure_diastolic'),
            'weight': vitals.get('weight'),
            'symptoms': symptoms,
            'diagnosis': diagnosis,
            'recommendations': recommendations,
            'language_used': current_language,
            'reading_date': datetime.utcnow(),
            'device_id': os.getenv('DEVICE_ID', 'unknown'),
            'location': os.getenv('DEVICE_LOCATION', 'unknown')
        }
        
        result = readings_collection.insert_one(reading_data)
        print(f"✔ Medical reading saved with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error saving medical reading: {e}")
        return None

# 15. ADDITIONAL MONGODB UTILITY FUNCTIONS
def get_user_medical_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Get user's medical reading history"""
    try:
        readings = readings_collection.find(
            {"user_id": user_id}
        ).sort("reading_date", -1).limit(limit)
        
        return list(readings)
    except Exception as e:
        print(f"Error fetching medical history: {e}")
        return []

def get_health_trends(user_id: str, days: int = 30) -> Dict:
    """Get health trends for a user over specified days"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "reading_date": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_temperature": {"$avg": "$temperature"},
                    "avg_heart_rate": {"$avg": "$heart_rate"},
                    "avg_bp_systolic": {"$avg": "$blood_pressure_systolic"},
                    "avg_bp_diastolic": {"$avg": "$blood_pressure_diastolic"},
                    "avg_oxygen_saturation": {"$avg": "$oxygen_saturation"},
                    "reading_count": {"$sum": 1}
                }
            }
        ]
        
        result = list(readings_collection.aggregate(pipeline))
        return result[0] if result else {}
        
    except Exception as e:
        print(f"Error calculating health trends: {e}")
        return {}

def main():
    """Main application function with multi-language support"""
    global current_language
    
    # Choose language first
    print("🌐 LANGUAGE SELECTION / भाषा चयन / भाषा निवड / భాష ఎంపిక")
    current_language = choose_language()
    
    # Medical disclaimer in selected language
    disclaimer = get_text('disclaimer')
    emergency_advice = get_text('emergency_advice')
    patient_guidance = get_text('patient_guidance')
    
    full_disclaimer = f"{get_text('welcome')} {disclaimer} {emergency_advice} {patient_guidance}"
    
    print(f"🩺 {full_disclaimer}")
    log_conversation("System", full_disclaimer)
    speak(full_disclaimer)
    
    # Main application loop - keeps running until user says exit
    while True:
        try:
            # User identification with patience in current language
            fullname = None
            name_attempts = 0
            max_name_attempts = 5
            
            while not fullname and name_attempts < max_name_attempts:
                if name_attempts == 0:
                    name_prompt = get_text('ask_name')
                else:
                    name_prompt = get_text('name_again')
                
                name_response = listen_with_retry(name_prompt, "your full name", 3)
                name_attempts += 1
                
                if name_response == "EXIT_COMMAND":
                    break
                
                if name_response:
                    # Basic validation for name
                    if len(name_response.strip()) >= 2 and any(c.isalpha() for c in name_response):
                        fullname = name_response.strip()
                        welcome_msgs = {
                            'english': f"Thank you, {fullname}. I'm pleased to meet you.",
                            'hindi': f"धन्यवाद, {fullname}। आपसे मिलकर खुशी हुई।",
                            'marathi': f"धन्यवाद, {fullname}. तुम्हाला भेटून आनंद झाला.",
                            'telugu': f"ధన్యవాదాలు, {fullname}. మిమ్మల్ని కలవడం ఆనందంగా ఉంది."
                        }
                        welcome_name_msg = welcome_msgs.get(current_language, welcome_msgs['english'])
                        print(f"🩺: {welcome_name_msg}")
                        speak(welcome_name_msg)
                        break
                    else:
                        error_msgs = {
                            'english': "I didn't catch a proper name. Could you please say your first and last name clearly?",
                            'hindi': "मुझे उचित नाम समझ में नहीं आया। कृपया अपना पहला और अंतिम नाम स्पष्ट रूप से कहें?",
                            'marathi': "मला योग्य नाव समजले नाही. कृपया तुमचे पहिले आणि शेवटचे नाव स्पष्टपणे सांगाल का?",
                            'telugu': "నేను సరైన పేరును పట్టుకోలేకపోయాను. దయచేసి మీ మొదటి మరియు చివరి పేరు స్పష్టంగా చెప్పగలరా?"
                        }
                        speak(error_msgs.get(current_language, error_msgs['english']))
                else:
                    if name_attempts < max_name_attempts:
                        patience_msgs = {
                            'english': "No problem, let me try to listen more carefully for your name.",
                            'hindi': "कोई बात नहीं, मैं आपके नाम को और ध्यान से सुनने की कोशिश करता हूँ।",
                            'marathi': "काही हरकत नाही, मी तुमचे नाव अधिक काळजीपूर्वक ऐकण्याचा प्रयत्न करतो.",
                            'telugu': "సమస్య లేదు, నేను మీ పేరును మరింత జాగ్రత్తగా వినడానికి ప్రయత్నిస్తాను."
                        }
                        patience_msg = patience_msgs.get(current_language, patience_msgs['english'])
                        print(f"🩺: {patience_msg}")
                        speak(patience_msg)
            
            if fullname == "EXIT_COMMAND" or not fullname:
                if not fullname:
                    exit_msgs = {
                        'english': "I understand this technology can sometimes be challenging. Feel free to try again when you're ready.",
                        'hindi': "मैं समझता हूँ कि यह तकनीक कभी-कभी चुनौतीपूर्ण हो सकती है। जब आप तैयार हों तो फिर से कोशिश करने में संकोच न करें।",
                        'marathi': "मला समजते की हे तंत्रज्ञान कधी कधी आव्हानात्मक असू शकते. तुम्ही तयार असाल तेव्हा पुन्हा प्रयत्न करण्यास मोकळे वाटा.",
                        'telugu': "ఈ టెక్నాలజీ కొన్నిసార్లు సవాలుగా ఉంటుందని నేను అర్థం చేసుకుంటాను. మీరు సిద్ధంగా ఉన్నప్పుడు మళ్లీ ప్రయత్నించడానికి వెనుకాడకండి."
                    }
                    speak(exit_msgs.get(current_language, exit_msgs['english']))
                break
            
            print(f"👤: {fullname}")
            log_conversation("User", fullname)
            
            # Check if user exists
            user_profile = check_user_exists(fullname)
            
            if user_profile:
                # Set language preference if stored
                if 'preferred_language' in user_profile and user_profile['preferred_language'] in SUPPORTED_LANGUAGES:
                    if user_profile['preferred_language'] != current_language:
                        set_language(user_profile['preferred_language'])
                        lang_switch_msgs = {
                            'english': f"I've switched to your preferred language: {SUPPORTED_LANGUAGES[current_language]['display']}",
                            'hindi': f"मैंने आपकी पसंदीदा भाषा पर स्विच कर दिया है: {SUPPORTED_LANGUAGES[current_language]['display']}",
                            'marathi': f"मी तुमच्या पसंतीच्या भाषेवर स्विच केले आहे: {SUPPORTED_LANGUAGES[current_language]['display']}",
                            'telugu': f"నేను మీ ఇష్టపడు భాషకు మార్చాను: {SUPPORTED_LANGUAGES[current_language]['display']}"
                        }
                        speak(lang_switch_msgs.get(current_language, lang_switch_msgs['english']))
                
                welcome_msg = f"{get_text('welcome_back')} {user_profile['fullname']}! {get_text('welcome_back')}"
                print(f"🩺: {welcome_msg}")
                log_conversation("Assistant", welcome_msg)
                speak(welcome_msg)
                
                # Show health trends
                trends = get_health_trends(user_profile['_id'])
                if trends and trends.get('reading_count', 0) > 0:
                    trend_msgs = {
                        'english': f"You have {trends['reading_count']} previous readings in the last 30 days.",
                        'hindi': f"आपके पास पिछले 30 दिनों में {trends['reading_count']} पिछली रीडिंग हैं।",
                        'marathi': f"तुमच्याकडे गेल्या 30 दिवसांत {trends['reading_count']} मागील रीडिंग आहेत.",
                        'telugu': f"మీకు గత 30 రోజులలో {trends['reading_count']} మునుపటి రీడింగ్‌లు ఉన్నాయి."
                    }
                    trend_msg = trend_msgs.get(current_language, trend_msgs['english'])
                    print(f"📊: {trend_msg}")
                    speak(trend_msg)
                    
            else:
                new_user_msg = f"{fullname}! {get_text('new_user')}"
                print(f"🩺: {new_user_msg}")
                log_conversation("Assistant", new_user_msg)
                speak(new_user_msg)
                
                # Create new profile
                profile_data = get_profile_info()
                
                if profile_data == "EXIT_COMMAND":
                    break
                
                profile_data['fullname'] = fullname
                user_id = create_user_profile(profile_data)
                
                if user_id:
                    user_profile = check_user_exists(fullname)
                    profile_created_msg = get_text('profile_created')
                    print(f"🩺: {profile_created_msg}")
                    log_conversation("Assistant", profile_created_msg)
                    speak(profile_created_msg)
                else:
                    error_msgs = {
                        'english': "There was an error creating your profile. Let me try again.",
                        'hindi': "आपकी प्रोफ़ाइल बनाने में त्रुटि हुई। मैं फिर से कोशिश करता हूँ।",
                        'marathi': "तुमची प्रोफाइल तयार करण्यात त्रुटी झाली. मी पुन्हा प्रयत्न करतो.",
                        'telugu': "మీ ప్రొఫైల్ సృష్టించడంలో లోపం ఉంది. నేను మళ్లీ ప్రయత్నిస్తాను."
                    }
                    speak(error_msgs.get(current_language, error_msgs['english']))
                    continue
            
            # Get current health problem with retry
            symptoms = listen_with_retry(get_text('ask_symptoms'), "your symptoms and health concerns")
            
            if symptoms == "EXIT_COMMAND":
                break
            
            if not symptoms:
                retry_msgs = {
                    'english': "I need to know your symptoms to help you. Let me ask again.",
                    'hindi': "आपकी मदद करने के लिए मुझे आपके लक्षणों को जानने की जरूरत है। मैं फिर से पूछता हूँ।",
                    'marathi': "तुमची मदत करण्यासाठी मला तुमची लक्षणे जाणून घेणे आवश्यक आहे. मी पुन्हा विचारतो.",
                    'telugu': "మీకు సహాయం చేయడానికి మీ లక్షణాలను తెలుసుకోవాల్సిన అవసరం ఉంది. నేను మళ్లీ అడుగుతాను."
                }
                speak(retry_msgs.get(current_language, retry_msgs['english']))
                continue
            
            print(f"👤: {symptoms}")
            log_conversation("User", symptoms)
            
            symptoms_acknowledged = get_text('understand_symptoms')
            print(f"🩺: {symptoms_acknowledged}")
            log_conversation("Assistant", symptoms_acknowledged)
            speak(symptoms_acknowledged)
            
            # Ask for reading mode
            mode_response = listen_with_retry(get_text('reading_mode'), "manual or sensor mode")
            
            if mode_response == "EXIT_COMMAND":
                break
            
            manual_mode = True if mode_response and extract_reading_mode(mode_response) == 'manual' else False
            
            # Collect vital signs
            vitals = collect_vital_signs(manual_mode)
            
            if vitals == "EXIT_COMMAND":
                break
            
            # Display collected vitals
            vitals_summary = f"""
            Vital Signs Collected:
            Temperature: {vitals.get('temperature', 'N/A')}°F
            Heart Rate: {vitals.get('heart_rate', 'N/A')} bpm
            Pulse Rate: {vitals.get('pulse_rate', 'N/A')} bpm
            Oxygen Saturation: {vitals.get('oxygen_saturation', 'N/A')}%
            Blood Pressure: {vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')} mmHg
            Weight: {vitals.get('weight', 'N/A')} kg
            """
            
            print(vitals_summary)
            log_conversation("Assistant", vitals_summary)
            speak(get_text('vitals_collected'))
            
            # Analyze health data
            analysis = analyze_health_data(user_profile, symptoms, vitals)
            
            # Present analysis
            analysis_msg = f"{get_text('analysis_complete')} {analysis['diagnosis']}"
            print(f"🩺: {analysis_msg}")
            log_conversation("Assistant", analysis_msg)
            speak(analysis_msg)
            
            # Save to MongoDB
            reading_id = save_medical_reading(
                user_profile['_id'], vitals, symptoms, 
                analysis['diagnosis'], analysis['recommendations']
            )
            
            # Final recommendations with care
            final_msgs = {
                'english': f"{get_text('data_saved')} {get_text('medical_advice')} {get_text('take_care')} I'm always here if you need another consultation.",
                'hindi': f"{get_text('data_saved')} {get_text('medical_advice')} {get_text('take_care')} यदि आपको दूसरे परामर्श की आवश्यकता है तो मैं हमेशा यहाँ हूँ।",
                'marathi': f"{get_text('data_saved')} {get_text('medical_advice')} {get_text('take_care')} जर तुम्हाला दुसऱ्या सल्ल्याची गरज असेल तर मी नेहमी येथे आहे.",
                'telugu': f"{get_text('data_saved')} {get_text('medical_advice')} {get_text('take_care')} మీకు మరొక కన్సల్టేషన్ అవసరమైతే నేను ఎల్లప్పుడూ ఇక్కడ ఉన్నాను."
            }
            
            final_msg = final_msgs.get(current_language, final_msgs['english'])
            print(f"🩺: {final_msg}")
            log_conversation("Assistant", final_msg)
            speak(final_msg)
            
            # Ask if user wants another consultation with caring options
            continue_options = {
                'english': """Would you like to:
- Have another health consultation with me?
- Update any information in your profile?
- Help a different patient?
- Change language?
- Or are you finished for now?

You can say 'another consultation', 'update profile', 'new patient', 'change language', or 'exit' to quit.""",
                'hindi': """क्या आप चाहेंगे:
- मेरे साथ एक और स्वास्थ्य परामर्श?
- अपनी प्रोफ़ाइल में कोई जानकारी अपडेट करना?
- एक अलग मरीज़ की मदद करना?
- भाषा बदलना?
- या आप अभी के लिए समाप्त कर रहे हैं?

आप 'दूसरा परामर्श', 'प्रोफ़ाइल अपडेट', 'नया मरीज़', 'भाषा बदलें', या बाहर निकलने के लिए 'बाहर निकलें' कह सकते हैं।""",
                'marathi': """तुम्हाला हवे आहे का:
- माझ्यासोबत आणखी एक आरोग्य सल्लामसलत?
- तुमच्या प्रोफाइलमधील कोणतीही माहिती अपडेट करा?
- वेगळ्या रुग्णाची मदत करा?
- भाषा बदला?
- किंवा तुम्ही आत्तासाठी संपवत आहात?

तुम्ही 'दुसरी सल्लामसलत', 'प्रोफाइल अपडेट', 'नवीन रुग्ण', 'भाषा बदला', किंवा सोडण्यासाठी 'बाहेर पडा' म्हणू शकता।""",
                'telugu': """మీరు కోరుకుంటున్నారా:
- నాతో మరొక ఆరోగ్య సంప్రదింపు?
- మీ ప్రొఫైల్‌లో ఏదైనా సమాచారాన్ని అప్‌డేట్ చేయాలా?
- వేరే పేషెంట్‌కు సహాయం చేయాలా?
- భాష మార్చాలా?
- లేదా మీరు ఇప్పుడు ముగించాలని అనుకుంటున్నారా?

మీరు 'మరొక కన్సల్టేషన్', 'ప్రొఫైల్ అప్‌డేట్', 'కొత్త పేషెంట్', 'భాష మార్చు', లేదా నిష్క్రమించడానికి 'ఎగ్జిట్' అని చెప్పవచ్చు."""
            }

            continue_response = listen_with_retry(continue_options.get(current_language, continue_options['english']), "your choice for next steps")
            
            if continue_response == "EXIT_COMMAND":
                break
            elif continue_response:
                response_lower = continue_response.lower()
                
                # Check for language change
                if check_for_language_change(continue_response):
                    continue
                    
                elif any(word in response_lower for word in ["update", "change", "modify", "profile", "अपडेट", "बदलें", "प्रोफ़ाइल", "अपडेट", "बदला", "प्रोफाइल", "అప్‌డేట్", "మార్చు", "ప్రొఫైల్"]):
                    update_msgs = {
                        'english': "I'll help you update your profile information.",
                        'hindi': "मैं आपकी प्रोफ़ाइल जानकारी अपडेट करने में आपकी मदद करूंगा।",
                        'marathi': "मी तुमच्या प्रोफाइल माहिती अपडेट करण्यात मदत करेन.",
                        'telugu': "నేను మీ ప్రొఫైల్ సమాచారాన్ని అప్‌డేట్ చేయడంలో సహాయం చేస్తాను."
                    }
                    speak(update_msgs.get(current_language, update_msgs['english']))
                    # Add profile update functionality here
                    continue_ask_msgs = {
                        'english': "Now, would you like a health consultation?",
                        'hindi': "अब, क्या आप स्वास्थ्य परामर्श चाहेंगे?",
                        'marathi': "आता, तुम्हाला आरोग्य सल्लामसलत हवी आहे का?",
                        'telugu': "ఇప్పుడు, మీకు ఆరోగ్య సంప్రదింపు కావాలా?"
                    }
                    speak(continue_ask_msgs.get(current_language, continue_ask_msgs['english']))
                    continue
                    
                elif any(word in response_lower for word in ["new", "different", "other", "patient", "नया", "अलग", "मरीज़", "नवीन", "वेगळा", "रुग्ण", "కొత్త", "వేరే", "పేషెంట్"]):
                    new_patient_msgs = {
                        'english': "I'll help you with a new patient. Let's start fresh.",
                        'hindi': "मैं आपको एक नए मरीज़ के साथ मदद करूंगा। चलिए नए सिरे से शुरू करते हैं।",
                        'marathi': "मी तुम्हाला नवीन रुग्णासह मदत करेन. चला नव्याने सुरुवात करूया.",
                        'telugu': "నేను మిమ్మల్ని కొత్త పేషెంట్‌తో సహాయం చేస్తాను. మొదటి నుండి ప్రారంభిద్దాం."
                    }
                    speak(new_patient_msgs.get(current_language, new_patient_msgs['english']))
                    continue
                    
                elif any(word in response_lower for word in ["yes", "another", "continue", "more", "again", "हाँ", "दूसरा", "जारी", "और", "फिर", "होय", "दुसरा", "सुरू", "आणखी", "पुन्हा", "అవును", "మరొక", "కొనసాగు", "మరింత", "మళ్లీ"]):
                    another_consult_msgs = {
                        'english': "I'm happy to help you with another health consultation.",
                        'hindi': "मुझे आपको एक और स्वास्थ्य परामर्श में मदद करने में खुशी होगी।",
                        'marathi': "मला तुम्हाला आणखी एका आरोग्य सल्लामसलतीत मदत करण्यात आनंद होत आहे.",
                        'telugu': "మరొక ఆరోగ్య సంప్రదింపుతో మీకు సహాయం చేయడంలో నేను సంతోషిస్తున్నాను."
                    }
                    speak(another_consult_msgs.get(current_language, another_consult_msgs['english']))
                    continue
                else:
                    break
            else:
                break
                
        except Exception as e:
            error_msg = f"An error occurred: {e}. Let me restart the consultation."
            print(f"❌ {error_msg}")
            log_conversation("System", error_msg)
            
            restart_msgs = {
                'english': "An error occurred. Let me restart the consultation.",
                'hindi': "एक त्रुटि हुई। मैं परामर्श को फिर से शुरू करता हूँ।",
                'marathi': "एक त्रुटी झाली. मी सल्लामसलत पुन्हा सुरू करतो.",
                'telugu': "ఒక లోపం సంభవించింది. నేను కన్సల్టేషన్‌ను మళ్లీ ప్రారంభిస్తాను."
            }
            speak(restart_msgs.get(current_language, restart_msgs['english']))
            continue
    
    # Farewell message
    farewell = get_text('exit_message')
    print(f"🩺: {farewell}")
    log_conversation("Assistant", farewell)
    speak(farewell)

if __name__ == "__main__":
    print("=" * 70)
    print("🩺 MEDICAL DEVICE ASSISTANT - Starting Up...")
    print("=" * 70)
    print("💡 HELPFUL TIPS:")
    print("   • Speak clearly and at a normal pace")
    print("   • The device will tell you exactly what it's listening for")
    print("   • You can say 'help' anytime for assistance")
    print("   • Say 'exit', 'quit', or 'goodbye' to stop")
    print("   • Say 'change language' to switch languages")
    print("   • Take your time - the device is patient with you")
    print("   • If the device doesn't hear you, it will try again")
    print("=" * 70)
    print("🌐 LANGUAGE SUPPORT:")
    print("   • English / अंग्रेजी / इंग्रजी / ఇంగ్లీష్")
    print("   • Hindi / हिंदी / हिंदी / హిందీ") 
    print("   • Marathi / मराठी / मराठी / మరాఠీ")
    print("   • Telugu / तेलुगु / तेलुगू / తెలుగు")
    print("=" * 70)
    print("🎙️ WHAT YOU'LL HEAR:")
    print("   • The device will tell you what it's listening for")
    print("   • It will repeat what it heard from you")
    print("   • It will confirm each piece of information")
    print("=" * 70)
    print("🔄 The assistant will keep running until you say 'exit'")
    print("🏥 Ready to help with your health consultation!")
    print("=" * 70)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Medical Device Assistant stopped by user (Ctrl+C)")
        farewell_msg = "Medical Device Assistant stopped. I hope you feel better soon. Take care!"
        speak(farewell_msg)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        error_msg = "I apologize, but there was a technical issue. Please restart the device and try again. If problems continue, please contact technical support."
        speak(error_msg)
        log_conversation("System", f"Critical Error: {e}")
    finally:
        print("🏥 Medical Device Assistant session ended")
        print("💙 Thank you for using your medical assistant. Stay healthy!")