#!/usr/bin/env python3
"""
Medical Device Assistant - Main Application
Multi-language voice-controlled medical consultation system
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.language_manager import LanguageManager
from core.speech_handler import SpeechHandler
from core.logger import Logger
from database.mongodb_client import MongoDBClient
from medical.vitals_collector import VitalsCollector
from medical.health_analyzer import HealthAnalyzer
from patient.profile_manager import ProfileManager
from patient.user_interface import UserInterface
from utils.helpers import Helpers

class MedicalAssistantApp:
    """Main Medical Assistant Application"""
    
    def __init__(self):
        # Validate configuration
        Settings.validate()
        
        # Initialize core components
        self.language_manager = LanguageManager()
        self.speech_handler = SpeechHandler(self.language_manager)
        self.logger = Logger()
        
        # Initialize database
        self.mongodb_client = MongoDBClient()
        
        # Initialize medical components
        self.vitals_collector = VitalsCollector(self.speech_handler, self.language_manager)
        self.health_analyzer = HealthAnalyzer(self.language_manager)
        
        # Initialize patient components
        self.profile_manager = ProfileManager(self.mongodb_client, self.speech_handler, self.language_manager)
        self.user_interface = UserInterface(self.speech_handler, self.language_manager)
        
        print("✔ Medical Device Assistant initialized")
    
    def choose_language(self):
        """Interactive language selection"""
        multilang_prompt = """
        Please choose your preferred language / कृपया अपनी पसंदीदा भाषा चुनें / कृपया तुमची पसंतीची भाषा निवडा / దయచేసి మీ ఇష్టపడు భాషను ఎంచుకోండి:
        
        Say 'English' for English
        'Hindi' के लिए 'Hindi' कहें  
        'Marathi' साठी 'Marathi' म्हणा
        తెలుగు కోసం 'Telugu' అనండి
        """
        
        print(f"🌐 {multilang_prompt}")
        
        # Use default English TTS for initial prompt
        import azure.cognitiveservices.speech as speechsdk
        tts_english = speechsdk.SpeechSynthesizer(
            speech_config=speechsdk.SpeechConfig(subscription=Settings.SPEECH_KEY, region=Settings.SPEECH_REGION),
            audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        )
        tts_english.speak_text_async("Please choose your preferred language. Say English, Hindi, Marathi, or Telugu.").get()
        
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"🎙️ Listening for language choice... (Attempt {attempt + 1}/{max_attempts})")
            
            # Use English speech recognition for language selection
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speechsdk.SpeechConfig(subscription=Settings.SPEECH_KEY, region=Settings.SPEECH_REGION)
            )
            
            result = recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                user_choice = result.text.strip()
                print(f"👤 You said: '{user_choice}'")
                
                detected_language = self.language_manager.detect_language_from_speech(user_choice)
                
                if self.language_manager.set_language(detected_language):
                    confirmation_msg = f"{self.language_manager.get_text('language_set')} {self.language_manager.get_language_config()['display']}"
                    print(f"✔ {confirmation_msg}")
                    self.speech_handler.speak(confirmation_msg)
                    return detected_language
                else:
                    if attempt < max_attempts - 1:
                        tts_english.speak_text_async("I didn't understand. Please say English, Hindi, Marathi, or Telugu clearly.").get()
            else:
                if attempt < max_attempts - 1:
                    tts_english.speak_text_async("I didn't hear you clearly. Please try again.").get()
        
        # Default to English if no valid selection
        self.language_manager.set_language('english')
        tts_english.speak_text_async("Setting language to English by default.").get()
        return 'english'
    
    def run(self):
        """Main application loop"""
        try:
            # Choose language first
            print("🌐 LANGUAGE SELECTION / भाषा चयन / भाषा निवड / భాష ఎంపిక")
            self.choose_language()
            
            # Medical disclaimer in selected language
            disclaimer = self.language_manager.get_text('disclaimer')
            emergency_advice = self.language_manager.get_text('emergency_advice')
            patient_guidance = self.language_manager.get_text('patient_guidance')
            
            full_disclaimer = f"{self.language_manager.get_text('welcome')} {disclaimer} {emergency_advice} {patient_guidance}"
            
            print(f"🩺 {full_disclaimer}")
            self.logger.log_conversation("System", full_disclaimer)
            self.speech_handler.speak(full_disclaimer)
            
            # Main application loop
            while True:
                try:
                    # User identification
                    fullname = self._get_user_name()
                    if fullname == "EXIT_COMMAND" or not fullname:
                        break
                    
                    self.logger.log_conversation("User", fullname)
                    
                    # Check if user exists or create new profile
                    user_profile = self._handle_user_profile(fullname)
                    if user_profile == "EXIT_COMMAND":
                        break
                    
                    # Get symptoms
                    symptoms = self._get_user_symptoms()
                    if symptoms == "EXIT_COMMAND":
                        break
                    
                    # Collect vital signs
                    vitals = self._collect_vitals()
                    if vitals == "EXIT_COMMAND":
                        break
                    
                    # Analyze health data
                    analysis = self._analyze_health(user_profile, symptoms, vitals)
                    
                    # Save results
                    self._save_results(user_profile, vitals, symptoms, analysis)
                    
                    # Ask for continuation
                    if not self._ask_for_continuation():
                        break
                        
                except Exception as e:
                    error_msg = f"An error occurred: {e}. Let me restart the consultation."
                    print(f"❌ {error_msg}")
                    self.logger.log_error(error_msg)
                    
                    restart_msgs = {
                        'english': "An error occurred. Let me restart the consultation.",
                        'hindi': "एक त्रुटि हुई। मैं परामर्श को फिर से शुरू करता हूँ।",
                        'marathi': "एक त्रुटी झाली. मी सल्लामसलत पुन्हा सुरू करतो.",
                        'telugu': "ఒక లోపం సంభవించింది. నేను కన్సల్టేషన్‌ను మళ్లీ ప్రారంభిస్తాను."
                    }
                    self.speech_handler.speak(restart_msgs.get(self.language_manager.current_language, restart_msgs['english']))
                    continue
            
            # Farewell message
            farewell = self.language_manager.get_text('exit_message')
            print(f"🩺: {farewell}")
            self.logger.log_conversation("Assistant", farewell)
            self.speech_handler.speak(farewell)
            
        except KeyboardInterrupt:
            print("\n👋 Medical Device Assistant stopped by user (Ctrl+C)")
            farewell_msg = "Medical Device Assistant stopped. I hope you feel better soon. Take care!"
            self.speech_handler.speak(farewell_msg)
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            error_msg = "I apologize, but there was a technical issue. Please restart the device and try again."
            self.speech_handler.speak(error_msg)
            self.logger.log_error(f"Critical Error: {e}")
        finally:
            print("🏥 Medical Device Assistant session ended")
            print("💙 Thank you for using your medical assistant. Stay healthy!")
    
    def _get_user_name(self):
        """Get user's full name with retry logic"""
        fullname = None
        name_attempts = 0
        max_name_attempts = Settings.MAX_NAME_ATTEMPTS
        
        while not fullname and name_attempts < max_name_attempts:
            if name_attempts == 0:
                name_prompt = self.language_manager.get_text('ask_name')
            else:
                name_prompt = self.language_manager.get_text('name_again')
            
            name_response = self.speech_handler.listen_with_retry(name_prompt, "your full name", 3)
            name_attempts += 1
            
            if name_response == "EXIT_COMMAND":
                return "EXIT_COMMAND"
            
            if name_response and len(name_response.strip()) >= 2 and any(c.isalpha() for c in name_response):
                fullname = name_response.strip()
                welcome_msgs = {
                    'english': f"Thank you, {fullname}. I'm pleased to meet you.",
                    'hindi': f"धन्यवाद, {fullname}। आपसे मिलकर खुशी हुई।",
                    'marathi': f"धन्यवाद, {fullname}. तुम्हाला भेटून आनंद झाला.",
                    'telugu': f"ధన్యవాదాలు, {fullname}. మిమ్మల్ని కలవడం ఆనందంగా ఉంది."
                }
                welcome_name_msg = welcome_msgs.get(self.language_manager.current_language, welcome_msgs['english'])
                print(f"🩺: {welcome_name_msg}")
                self.speech_handler.speak(welcome_name_msg)
                return fullname
        
        return None
    
    def _handle_user_profile(self, fullname: str):
        """Handle user profile creation or retrieval"""
        user_profile = self.profile_manager.check_user_exists(fullname)
        
        if user_profile:
            # Set language preference if stored
            if 'preferred_language' in user_profile and user_profile['preferred_language'] in ['english', 'hindi', 'marathi', 'telugu']:
                if user_profile['preferred_language'] != self.language_manager.current_language:
                    self.language_manager.set_language(user_profile['preferred_language'])
            
            welcome_msg = f"{self.language_manager.get_text('welcome_back')} {user_profile['fullname']}"
            print(f"🩺: {welcome_msg}")
            self.speech_handler.speak(welcome_msg)
            return user_profile
        else:
            new_user_msg = f"{fullname}! {self.language_manager.get_text('new_user')}"
            print(f"🩺: {new_user_msg}")
            self.speech_handler.speak(new_user_msg)
            
            # Create new profile
            profile_data = self.profile_manager.get_profile_info()
            if profile_data == "EXIT_COMMAND":
                return "EXIT_COMMAND"
            
            profile_data['fullname'] = fullname
            user_id = self.profile_manager.create_user_profile(profile_data)
            
            if user_id:
                user_profile = self.profile_manager.check_user_exists(fullname)
                profile_created_msg = self.language_manager.get_text('profile_created')
                print(f"🩺: {profile_created_msg}")
                self.speech_handler.speak(profile_created_msg)
                return user_profile
            else:
                error_msgs = {
                    'english': "There was an error creating your profile. Let me try again.",
                    'hindi': "आपकी प्रोफ़ाइल बनाने में त्रुटि हुई। मैं फिर से कोशिश करता हूँ।",
                    'marathi': "तुमची प्रोफाइल तयार करण्यात त्रुटी झाली. मी पुन्हा प्रयत्न करतो.",
                    'telugu': "మీ ప్రొఫైల్ సృష్టించడంలో లోపం ఉంది. నేను మళ్లీ ప్రయత్నిస్తాను."
                }
                self.speech_handler.speak(error_msgs.get(self.language_manager.current_language, error_msgs['english']))
                return "EXIT_COMMAND"
    
    def _get_user_symptoms(self):
        """Get user's symptoms"""
        symptoms = self.speech_handler.listen_with_retry(
            self.language_manager.get_text('ask_symptoms'), 
            "your symptoms and health concerns"
        )
        
        if symptoms == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        
        if not symptoms:
            retry_msgs = {
                'english': "I need to know your symptoms to help you. Let me ask again.",
                'hindi': "आपकी मदद करने के लिए मुझे आपके लक्षणों को जानने की जरूरत है। मैं फिर से पूछता हूँ।",
                'marathi': "तुमची मदत करण्यासाठी मला तुमची लक्षणे जाणून घेणे आवश्यक आहे. मी पुन्हा विचारतो.",
                'telugu': "మీకు సహాయం చేయడానికి మీ లక్షణాలను తెలుసుకోవాల్సిన అవసరం ఉంది. నేను మళ్లీ అడుగుతాను."
            }
            self.speech_handler.speak(retry_msgs.get(self.language_manager.current_language, retry_msgs['english']))
            return "EXIT_COMMAND"
        
        print(f"👤: {symptoms}")
        self.logger.log_conversation("User", symptoms)
        
        symptoms_acknowledged = self.language_manager.get_text('understand_symptoms')
        print(f"🩺: {symptoms_acknowledged}")
        self.logger.log_conversation("Assistant", symptoms_acknowledged)
        self.speech_handler.speak(symptoms_acknowledged)
        
        return symptoms
    
    def _collect_vitals(self):
        """Collect vital signs from user"""
        # Ask for reading mode
        mode_response = self.speech_handler.listen_with_retry(
            self.language_manager.get_text('reading_mode'), 
            "manual or sensor mode"
        )
        
        if mode_response == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        
        manual_mode = True if mode_response and self.vitals_collector.extract_reading_mode(mode_response) == 'manual' else False
        
        # Collect vital signs
        vitals = self.vitals_collector.collect_vital_signs(manual_mode)
        
        if vitals == "EXIT_COMMAND":
            return "EXIT_COMMAND"
        
        # Display collected vitals
        vitals_summary = Helpers.format_vitals_summary(vitals)
        print(vitals_summary)
        self.logger.log_conversation("Assistant", vitals_summary)
        self.speech_handler.speak(self.language_manager.get_text('vitals_collected'))
        
        return vitals
    
    def _analyze_health(self, user_profile, symptoms, vitals):
        """Analyze health data using AI"""
        analysis = self.health_analyzer.analyze_health_data(user_profile, symptoms, vitals)
        
        # Present analysis
        analysis_msg = f"{self.language_manager.get_text('analysis_complete')} {analysis['diagnosis']}"
        print(f"🩺: {analysis_msg}")
        self.logger.log_conversation("Assistant", analysis_msg)
        self.speech_handler.speak(analysis_msg)
        
        return analysis
    
    def _save_results(self, user_profile, vitals, symptoms, analysis):
        """Save medical reading results"""
        reading_id = self.mongodb_client.save_medical_reading(
            user_profile['_id'], vitals, symptoms, 
            analysis['diagnosis'], analysis['recommendations'],
            self.language_manager.current_language
        )
        
        # Final recommendations
        final_msgs = {
            'english': f"{self.language_manager.get_text('data_saved')} {self.language_manager.get_text('medical_advice')} {self.language_manager.get_text('take_care')} I'm always here if you need another consultation.",
            'hindi': f"{self.language_manager.get_text('data_saved')} {self.language_manager.get_text('medical_advice')} {self.language_manager.get_text('take_care')} यदि आपको दूसरे परामर्श की आवश्यकता है तो मैं हमेशा यहाँ हूँ।",
            'marathi': f"{self.language_manager.get_text('data_saved')} {self.language_manager.get_text('medical_advice')} {self.language_manager.get_text('take_care')} जर तुम्हाला दुसऱ्या सल्ल्याची गरज असेल तर मी नेहमी येथे आहे.",
            'telugu': f"{self.language_manager.get_text('data_saved')} {self.language_manager.get_text('medical_advice')} {self.language_manager.get_text('take_care')} మీకు మరొక కన్సల్టేషన్ అవసరమైతే నేను ఎల్లప్పుడూ ఇక్కడ ఉన్నాను."
        }
        
        final_msg = final_msgs.get(self.language_manager.current_language, final_msgs['english'])
        print(f"🩺: {final_msg}")
        self.logger.log_conversation("Assistant", final_msg)
        self.speech_handler.speak(final_msg)
    
    def _ask_for_continuation(self):
        """Ask user if they want to continue"""
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

आप 'दूसरा परामर्श', 'प्रोफ़ाइल अपडेट', 'नया मरीज़', 'भाषा बदलें', या 'exit' कह सकते हैं।""",
            'marathi': """तुम्हाला हवे आहे का:
- माझ्यासोबत आणखी एक आरोग्य सल्लामसलत?
- तुमच्या प्रोफाइलमधील कोणतीही माहिती अपडेट करा?
- वेगळ्या रुग्णाची मदत करा?
- भाषा बदला?
- किंवा तुम्ही आत्तासाठी संपवत आहात?

तुम्ही 'दुसरी सल्लामसलत', 'प्रोफाइल अपडेट', 'नवीन रुग्ण', 'भाषा बदला', किंवा 'exit' म्हणू शकता।""",
            'telugu': """మీరు కోరుకుంటున్నారా:
- నాతో మరొక ఆరోగ్య సంప్రదింపు?
- మీ ప్రొఫైల్‌లో ఏదైనా సమాచారాన్ని అప్‌డేట్ చేయాలా?
- వేరే పేషెంట్‌కు సహాయం చేయాలా?
- భాష మార్చాలా?
- లేదా మీరు ఇప్పుడు ముగించాలని అనుకుంటున్నారా?

మీరు 'మరొక కన్సల్టేషన్', 'ప్రొఫైల్ అప్‌డేట్', 'కొత్త పేషెంట్', 'భాష మార్చు', లేదా 'exit' అని చెప్పవచ్చు."""
        }

        continue_response = self.speech_handler.listen_with_retry(
            continue_options.get(self.language_manager.current_language, continue_options['english']), 
            "your choice for next steps"
        )
        
        if continue_response == "EXIT_COMMAND":
            return False
        elif continue_response:
            response_lower = continue_response.lower()
            
            # Check for language change
            if self.user_interface.check_for_language_change(continue_response):
                self.choose_language()
                return True
                
            elif any(word in response_lower for word in ["update", "change", "modify", "profile", "अपडेट", "बदलें", "प्रोफ़ाइल", "अपडेट", "बदला", "प्रोफाइल", "అప్‌డేట్", "మార్చు", "ప్రొఫైల్"]):
                update_msgs = {
                    'english': "Profile update functionality would be implemented here.",
                    'hindi': "प्रोफ़ाइल अपडेट कार्यक्षमता यहाँ लागू की जाएगी।",
                    'marathi': "प्रोफाइल अपडेट कार्यक्षमता येथे लागू केली जाईल.",
                    'telugu': "ప్రొఫైల్ అప్‌డేట్ కార్యాచరణ ఇక్కడ అమలు చేయబడుతుంది."
                }
                self.speech_handler.speak(update_msgs.get(self.language_manager.current_language, update_msgs['english']))
                return True
                
            elif any(word in response_lower for word in ["new", "different", "other", "patient", "नया", "अलग", "मरीज़", "नवीन", "वेगळा", "रुग्ण", "కొత్త", "వేరే", "పేషెంట్"]):
                new_patient_msgs = {
                    'english': "I'll help you with a new patient. Let's start fresh.",
                    'hindi': "मैं आपको एक नए मरीज़ के साथ मदद करूंगा। चलिए नए सिरे से शुरू करते हैं।",
                    'marathi': "मी तुम्हाला नवीन रुग्णासह मदत करेन. चला नव्याने सुरुवात करूया.",
                    'telugu': "నేను మిమ్మల్ని కొత్త పేషెంట్‌తో సహాయం చేస్తాను. మొదటి నుండి ప్రారంభిద్దాం."
                }
                self.speech_handler.speak(new_patient_msgs.get(self.language_manager.current_language, new_patient_msgs['english']))
                return True
                
            elif any(word in response_lower for word in ["yes", "another", "continue", "more", "again", "हाँ", "दूसरा", "जारी", "और", "फिर", "होय", "दुसरा", "सुरू", "आणखी", "पुन्हा", "అవును", "మరొక", "కొనసాగు", "మరింత", "మళ్లీ"]):
                another_consult_msgs = {
                    'english': "I'm happy to help you with another health consultation. Let's begin.",
                    'hindi': "मुझे आपको एक और स्वास्थ्य परामर्श में मदद करने में खुशी होगी। चलिए शुरू करते हैं।",
                    'marathi': "मला तुम्हाला आणखी एका आरोग्य सल्लामसलतीत मदत करण्यात आनंद होत आहे. चला सुरुवात करूया.",
                    'telugu': "మరొక ఆరోగ్య సంప్రదింపుతో మీకు సహాయం చేయడంలో నేను సంతోషిస్తున్నాను. మొదలుపెట్టండి."
                }
                self.speech_handler.speak(another_consult_msgs.get(self.language_manager.current_language, another_consult_msgs['english']))
                return True
            else:
                return False
        else:
            return False


def main():
    """Main entry point"""
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
    
    # Create and run the medical assistant app
    app = MedicalAssistantApp()
    app.run()


if __name__ == "__main__":
    main()