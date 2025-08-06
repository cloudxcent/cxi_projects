import re
import os
import time
import json
import string
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from num2words import num2words

# === Helper imports ===
from restaurant_menu_helper import RestaurantMenuHelper
from restaurant_db_helper_normalized import RestaurantDB

# ==== Define what "main course" means ====
MAIN_COURSE_CATEGORY_KEYWORDS = {
    "main", "indian", "sabzi", "rice", "noodle", "pasta",
    "bread", "roti", "dal", "curry", "biryani", "chinese",
    "continental", "south indian"
}
EXCLUDE_CATEGORY_KEYWORDS = {"starter", "dessert", "soup", "appetizer", "salad"}

# === Load environment variables ===
load_dotenv()
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
assert all([OPENAI_KEY, OPENAI_ENDPOINT, OPENAI_DEPLOY, OPENAI_VERSION]), "OpenAI vars missing"
assert all([SPEECH_KEY, SPEECH_REGION]), "Speech vars missing"

# === Initialize clients ===
client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)

speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_cfg.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
print("✔ Services initialized")

mic_audio_cfg = speechsdk.audio.AudioConfig(use_default_microphone=True)
speaker_audio_cfg = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

LOG_FILE = "restaurant_assistant.log"

# === Intent Classification Constants ===
INTENT_ORDER = "place_order"
INTENT_MENU = "ask_menu"
INTENT_STATUS = "check_status"
INTENT_GENERAL = "general_question"
INTENT_GREETING = "greeting"
INTENT_GOODBYE = "goodbye"
INTENT_PRICE = "ask_price"
INTENT_DESCRIPTION = "ask_description"
INTENT_STARTER = "ask_starter"
INTENT_DESSERT = "ask_dessert"

# === New Session Logging Functions ===
def get_current_date_str():
    return datetime.now().strftime("%Y-%m-%d")

def get_session_number(log_file):
    try:
        current_date = get_current_date_str()
        max_session = 0
        
        if not os.path.exists(log_file):
            return 1
            
        with open(log_file, "r", encoding='utf-8') as f:
            for line in f:
                if "===== Customer Session" in line and "START" in line and current_date in line:
                    try:
                        parts = line.split()
                        session_num = int(parts[3])
                        if session_num > max_session:
                            max_session = session_num
                    except (IndexError, ValueError):
                        continue
                        
        return max_session + 1
    except Exception as e:
        print(f"Error getting session number: {e}")
        return 1

def log_session_start(log_file):
    session_num = get_session_number(log_file)
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(f"\n===== Customer Session {session_num} - {get_current_date_str()} START =====\n")

def log_session_end(log_file):
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(f"===== Customer Session END =====\n")

# === Enhanced Utility Functions ===
def classify_intent(text: str) -> str:
    """Classify user intent using Azure OpenAI"""
    system_prompt = """You are an intent classifier for a restaurant assistant. 
    Classify the user's intent into one of these categories:
    - place_order: When user wants to order food items
    - ask_menu: When user asks about menu options
    - check_status: When user asks about order status
    - ask_price: When user asks about item prices
    - ask_description: When user asks about item details
    - greeting: When user greets or starts conversation
    - goodbye: When user says goodbye or wants to end
    - general_question: Any other restaurant-related question
    - ask_starter: When user asks about starters
    - ask_dessert: When user asks about desserts
    
    Return ONLY the intent name, nothing else."""
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=20
        )
        intent = response.choices[0].message.content.strip().lower()
        return intent if intent in {
            INTENT_ORDER, INTENT_MENU, INTENT_STATUS, 
            INTENT_GENERAL, INTENT_GREETING, INTENT_GOODBYE,
            INTENT_PRICE, INTENT_DESCRIPTION, INTENT_STARTER,
            INTENT_DESSERT
        } else INTENT_GENERAL
    except Exception as e:
        print(f"Intent classification error: {e}")
        return INTENT_GENERAL

def is_negative_response(text: str) -> bool:
    """Check if user response indicates they don't want to order"""
    negative_phrases = {
        "don't want", "not interested", "nothing", "no thanks", 
        "no thank you", "not now", "maybe later", "skip", "pass",
        "no", "nope", "not today"
    }
    text = text.lower().strip()
    return any(phrase in text for phrase in negative_phrases)

def is_positive_response(text: str) -> bool:
    """Check if user response indicates agreement or positive intent"""
    positive_phrases = {
        "yes", "please", "sure", "okay", "ok", "yeah", "yep", 
        "i'd like", "i want", "i would like", "show me", "tell me",
        "what do you have", "what are the options"
    }
    text = text.lower().strip()
    return any(phrase in text for phrase in positive_phrases)

def log_conversation(role: str, message: str):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        if f.tell() == 0 or "\n===== Customer Session" not in open(LOG_FILE).read():
            log_session_start(LOG_FILE)
        f.write(f"{timestamp} {role}: {message}\n")

def tts(text: str):
    try:
        ssml = (
            f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-IN'>
                <voice name='{speech_cfg.speech_synthesis_voice_name}'>
                <prosody rate='1.2'>{text}</prosody>
                </voice>
                </speak>"""
        )
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_cfg, audio_config=speaker_audio_cfg
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.SpeechSynthesisCancellationDetails(result)
            error_msg = f"Speech canceled: {cancellation.reason}"
            if cancellation.reason == speechsdk.CancellationReason.Error:
                error_msg += f" (Details: {cancellation.error_details})"
            print(error_msg)
            log_conversation("System", f"Speech Error: {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Speech synthesis failed: {str(e)}"
        print(error_msg)
        log_conversation("System", error_msg)
        return False

def clean_for_speech(text: str) -> str:
    text = re.sub(r"【.*?】", "", text)
    return text

def speak(text: str):
    clean = clean_for_speech(text)
    print(f"🤖: {clean}")
    log_conversation("Assistant", clean)
    if not tts(clean):
        print(f"Failed to speak: {clean}")

def listen() -> str | None:
    recog = speechsdk.SpeechRecognizer(
        speech_config=speech_cfg,
        audio_config=mic_audio_cfg
    )
    print("🎙️ Speak now...")
    res = recog.recognize_once_async().get()
    if res.reason == speechsdk.ResultReason.RecognizedSpeech:
        return res.text
    print("⚠️ Nothing heard")
    return None

# === Menu Functions ===
def load_menu(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_ist_time():
    utc_now = datetime.now(timezone.utc)
    return utc_now.astimezone(timezone(timedelta(hours=5, minutes=30)))

def get_meal_period(menu_data: dict) -> str:
    hour = get_ist_time().hour
    for period, times in menu_data["time_ranges"].items():
        if times["start"] <= hour < times["end"]:
            return period
    return "dinner"

def format_price(p) -> str:
    rupees = int(round(float(p)))
    words = num2words(rupees, to='cardinal')
    return f"{words} rupees"

def speak_items_as_statement(menu_data: dict, category_name: str, meal_period: str, show_price=False, show_description=False):
    items = []
    if category_name in menu_data["menu"][meal_period]:
        items = menu_data["menu"][meal_period][category_name]
    elif category_name in menu_data["menu"]["all_day"]:
        items = menu_data["menu"]["all_day"][category_name]
    else:
        statement = f"Sorry, I couldn't find {category_name}."
        speak(statement)
        return

    names = []
    for item in items:
        name = item['name']
        if show_price:
            price_words = format_price(item["price"])
            name += f" for {price_words}"
        if show_description and 'description' in item:
            name += f" - {item['description']}"
        names.append(name)

    if not names:
        statement = f"We have no items in {category_name} at the moment."
    elif len(names) == 1:
        statement = names[0]
    else:
        all_but_last = ", ".join(names[:-1])
        last = names[-1]
        statement = f"{all_but_last}, and {last}"

    speak(statement)

def get_category_items(menu_data: dict, category_type: str, meal_period: str):
    categories = []
    for category in menu_data["menu"][meal_period].keys():
        if category_type in category.lower():
            categories.append(category)
    for category in menu_data["menu"]["all_day"].keys():
        if category_type in category.lower():
            categories.append(category)
    items = []
    for category in categories:
        if category in menu_data["menu"][meal_period]:
            items.extend(menu_data["menu"][meal_period][category])
        else:
            items.extend(menu_data["menu"]["all_day"][category])
    return items, categories

# === AI Assistant Functions ===
def create_vector_store(files: list[str], store_name="RestaurantStore") -> str:
    file_ids = []
    for path in files:
        with open(path, "rb") as f:
            fid = client.files.create(file=f, purpose="assistants").id
            file_ids.append(fid)
            print(f"✔ Uploaded {path}")
    vs = client.beta.vector_stores.create(name=store_name, file_ids=file_ids)
    print("⏳ Building vector store...")
    while client.beta.vector_stores.retrieve(vs.id).status != "completed":
        time.sleep(2)
    print("✔ Vector store ready")
    return vs.id

def build_assistant(vs_id: str):
    asst = client.beta.assistants.create(
        name="Restaurant Assistant",
        instructions="""You are a helpful waiter at Nisarg Hotel Solapur. Follow these rules:
        1. First check the attached files for restaurant information
        2. For menu questions, determine meal period (breakfast/lunch/snacks/dinner) based on current IST time
        3. When asked about recommendations, consider:
           - For spicy dishes: mention 'Veg Kolhapuri', 'Chana Masala', 'Paneer Butter Masala (medium spicy)'
           - For kids: mention 'Plain Dosa', 'Pasta', 'Pizza', 'French Fries'
           - For groups: recommend 'Thali' options or combo meals
           - For gravy dishes: recommend 'Paneer Butter Masala', 'Dal Makhani', 'Malai Kofta'
           - For healthy options: recommend 'Salads', 'Grilled Vegetables', 'Steamed Rice'
        4. For portion questions: 
           - Thali serves 1, Biryani serves 2-3, Pasta serves 1-2
        5. Keep responses short (1-2 sentences) but friendly
        6. Remember previous recommendations if asked again
        7. Never mention you're an AI
         """,
        model=OPENAI_DEPLOY,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}}
    )
    print("✔ Assistant created")
    return asst

def get_ai_response(thread, assistant, user_input, context=None):
    system_message = {
        "role": "system",
        "content": f"""You are a helpful waiter at Nisarg Hotel Solapur. Follow these rules:
        1. First check the attached files for restaurant information
        2. For menu questions, determine meal period (breakfast/lunch/snacks/dinner) based on current IST time
        3. When asked about recommendations, consider:
           - For spicy dishes: mention 'Veg Kolhapuri', 'Chana Masala', 'Paneer Butter Masala (medium spicy)'
           - For kids: mention 'Plain Dosa', 'Pasta', 'Pizza', 'French Fries'
           - For groups: recommend 'Thali' options or combo meals
           - For gravy dishes: recommend 'Paneer Butter Masala', 'Dal Makhani', 'Malai Kofta'
           - For healthy options: recommend 'Salads', 'Grilled Vegetables', 'Steamed Rice'
        4. For portion questions: 
           - Thali serves 1, Biryani serves 2-3, Pasta serves 1-2
        5. Keep responses short (1-2 sentences) but friendly
        6. Remember previous recommendations if asked again
        7. Never mention you're an AI
        {f"Current context: {context}" if context else ""}"""
    }
    
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    start_time = time.time()
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        if time.time() - start_time > 5:
            print("⚠️ Assistant response timeout")
            speak("Please wait, I'm still thinking...")
            break
        time.sleep(0.5)
    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order="desc",
        limit=1
    )
    if not messages.data:
        print("⚠️ No messages received")
        return "Sorry, I didn't get a response."
    if not messages.data[0].content:
        print("⚠️ Empty message content")
        return "My response came back empty."
    if not hasattr(messages.data[0].content[0], 'text'):
        print("⚠️ Non-text response")
        return "I can't process this type of response."
    try:
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        print(f"⚠️ Error processing response: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

# === Order Processing ===
def extract_structured_order(user_utterance: str) -> list[dict]:
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that extracts dishes and their quantities from customer "
            "order sentences. Return ONLY a JSON array of objects with keys 'name' and 'quantity' "
            "(default quantity=1 if missing)."
        )
    }
    user_message = {
        "role": "user",
        "content": user_utterance
    }
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[system_message, user_message],
            max_tokens=200,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        order_items = json.loads(content)
        if isinstance(order_items, list) and all(isinstance(i, dict) and 'name' in i for i in order_items):
            for item in order_items:
                if 'quantity' not in item or not isinstance(item['quantity'], int) or item['quantity'] <= 0:
                    item['quantity'] = 1
            return order_items
    except Exception as e:
        print(f"Error parsing structured order from LLM: {e}")
    return []

def clean_dish_name(dish_name: str) -> str:
    return dish_name.translate(str.maketrans('', '', string.punctuation)).strip()

def process_main_courses(menu_data: dict, meal_period: str, collected_orders: list, context: list):
    def is_main_course_category(category):
        category_low = category.lower()
        if any(ex in category_low for ex in EXCLUDE_CATEGORY_KEYWORDS):
            return False
        return any(key in category_low for key in MAIN_COURSE_CATEGORY_KEYWORDS)

    all_categories = set(menu_data["menu"][meal_period].keys()) | set(menu_data["menu"]["all_day"].keys())
    main_categories = [cat for cat in all_categories if is_main_course_category(cat)]

    if not main_categories:
        speak("We don't have any main courses available right now.")
        return collected_orders, False, context

    for category in main_categories:
        speak_items_as_statement(menu_data, category, meal_period)

    speak("What would you like to order?")

    while True:
        user_input = listen()
        if not user_input:
            speak("I didn't catch that. Could you please repeat your order?")
            continue
            
        print(f"👤: {user_input}")
        log_conversation("User", user_input)
        context.append(f"User: {user_input}")
        cmd = user_input.lower().strip()

        intent = classify_intent(user_input)
        if intent == INTENT_GENERAL or intent == INTENT_DESCRIPTION or intent == INTENT_PRICE:
            response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
            speak(response)
            context.append(f"Assistant: {response}")
            speak("What would you like to order from our main courses?")
            continue

        if is_negative_response(cmd):
            speak("No problem at all. Let me know if you change your mind.")
            return collected_orders, False, context
            
        if any(word in cmd for word in ["exit", "quit", "bye", "stop"]):
            speak("Thank you, please visit again!")
            log_session_end(LOG_FILE)
            return collected_orders, True, context

        if any(word in cmd for word in ["price", "how much"]):
            for category in main_categories:
                items = menu_data["menu"][meal_period].get(category, []) + menu_data["menu"]["all_day"].get(category, [])
                for item in items:
                    if item['name'].lower() in cmd:
                        speak(f"{item['name']} costs {format_price(item['price'])}")
                        break
            continue

        structured_orders = extract_structured_order(user_input)
        if structured_orders:
            for order_item in structured_orders:
                dish_name_raw = order_item.get("name", "").strip()
                quantity = order_item.get("quantity", 1)
                dish_data = menu_helper.find_dish_fuzzy(dish_name_raw, meal_period)
                if dish_data:
                    collected_orders.append({
                        "name": dish_data["name"],
                        "qty": quantity,
                        "price": dish_data["price"]
                    })
                    speak(f"Added {quantity} {dish_data['name']} to your order.")
                else:
                    speak(f"Sorry, we don't have {dish_name_raw} in our main courses.")
            return collected_orders, False, context
        else:
            speak("I didn't quite get that. Could you tell me the dish name and quantity?")

def process_starters(menu_data: dict, meal_period: str, collected_orders: list, context: list):
    items, categories = get_category_items(menu_data, "starter", meal_period)
    if not items:
        return collected_orders, False, context

    # 1. SHOW MENU IMMEDIATELY (NO SEPARATE CONFIRMATION)
    speak_items = []
    for category in categories:
        items = menu_data["menu"][meal_period].get(category, []) + \
                menu_data["menu"]["all_day"].get(category, [])
        speak_items.extend(item["name"] for item in items)
    
    # Format: "Paneer Tikka, Veg Manchurian, and French Fries"
    menu_text = ", ".join(speak_items[:-1]) + f", and {speak_items[-1]}" if len(speak_items) > 1 else speak_items[0]
    speak(f"We have {menu_text}. Would you like to order any starters?")

    # 2. SINGLE INPUT HANDLING
    while True:
        user_input = listen()
        if not user_input:
            speak("Please say your order or 'no' to skip.")
            continue
            
        print(f"👤: {user_input}")
        log_conversation("User", user_input)
        
        if is_negative_response(user_input):
            speak("No problem, skipping starters.")
            return collected_orders, False, context

        # Process order directly
        structured_orders = extract_structured_order(user_input)
        if structured_orders:
            for item in structured_orders:
                dish_data = menu_helper.find_dish_fuzzy(item["name"], meal_period)
                if dish_data:
                    collected_orders.append({
                        "name": dish_data["name"],
                        "qty": item.get("quantity", 1),
                        "price": dish_data["price"]
                    })
                    speak(f"Added {item.get('quantity', 1)} {dish_data['name']}.")
            return collected_orders, False, context
        else:
            speak("Please specify like: '1 Paneer Tikka' or say 'no' to skip")

def process_desserts(menu_data: dict, meal_period: str, collected_orders: list, context: list):
    items, categories = get_category_items(menu_data, "dessert", meal_period)
    if not items:
        return collected_orders, False, context

    # 1. SHOW MENU AND PROMPT IN ONE QUESTION
    speak_items = []
    for category in categories:
        items = menu_data["menu"][meal_period].get(category, []) + \
                menu_data["menu"]["all_day"].get(category, [])
        speak_items.extend(item["name"] for item in items)
    
    # Format: "Gulab Jamun, Ice Cream, and Chocolate Cake"
    menu_text = ", ".join(speak_items[:-1]) + f", and {speak_items[-1]}" if len(speak_items) > 1 else speak_items[0]
    speak(f"We have {menu_text}. Would you like to order any desserts?")

    # 2. SINGLE INPUT HANDLING
    while True:
        user_input = listen()
        if not user_input:
            speak("Please say your order or 'no' to skip.")
            continue
            
        print(f"👤: {user_input}")
        log_conversation("User", user_input)
        
        if is_negative_response(user_input):
            speak("No problem, skipping desserts.")
            return collected_orders, False, context

        # Process order directly
        structured_orders = extract_structured_order(user_input)
        if structured_orders:
            for item in structured_orders:
                dish_data = menu_helper.find_dish_fuzzy(item["name"], meal_period)
                if dish_data:
                    collected_orders.append({
                        "name": dish_data["name"],
                        "qty": item.get("quantity", 1),
                        "price": dish_data["price"]
                    })
                    speak(f"Added {item.get('quantity', 1)} {dish_data['name']}.")
            return collected_orders, False, context
        else:
            speak("Please specify like: '2 Gulab Jamun' or say 'no' to skip")

def handle_direct_order(user_input: str, meal_period: str, collected_orders: list, context: list):
    if is_negative_response(user_input):
        speak("No problem at all. Please let me know if you'd like anything later.")
        return collected_orders, False, context
        
    structured_orders = extract_structured_order(user_input)
    if not structured_orders:
        intent = classify_intent(user_input)
        if intent == INTENT_ORDER:
            speak("I didn't quite catch what you'd like to order. Could you please say the dish name and quantity?")
        elif intent == INTENT_GENERAL or intent == INTENT_DESCRIPTION or intent == INTENT_PRICE:
            response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
            speak(response)
            context.append(f"Assistant: {response}")
        else:
            speak("I didn't quite understand your order. Could you please say the dish name and quantity?")
        return collected_orders, False, context

    for order_item in structured_orders:
        dish_name_raw = order_item.get("name", "").strip()
        quantity = order_item.get("quantity", 1)
        dish_data = menu_helper.find_dish_fuzzy(dish_name_raw, meal_period)
        if dish_data:
            collected_orders.append({
                "name": dish_data["name"],
                "qty": quantity,
                "price": dish_data["price"]
            })
            speak(f"Added {quantity} {dish_data['name']} to your order.")
        else:
            speak(f"Sorry, we don't have {dish_name_raw} on our menu right now.")
    return collected_orders, False, context

def check_order_status(order_id: int):
    try:
        status = db_helper.get_order_status(order_id)
        if status is None:
            speak(f"Sorry, I couldn't find order number {order_id}. Could you please check the number?")
            return
            
        status_map = {
            "received": "Your order has been received and is being prepared.",
            "preparing": "The kitchen is currently preparing your order.",
            "ready": "Your order is ready for pickup or delivery!",
            "completed": "Your order has been completed. Enjoy your meal!",
            "cancelled": "Your order was cancelled."
        }
        message = status_map.get(status.lower(), f"Your order status is: {status}.")
        speak(message)
    except Exception as e:
        print(f"Error fetching order status: {e}")
        speak("Sorry, I'm having trouble checking your order status. Please try again later.")

def handle_dynamic_ordering_flow(user_input: str, menu_data: dict, meal_period: str, collected_orders: list, context: list):
    intent = classify_intent(user_input)
    
    if intent == INTENT_STARTER:
        collected_orders, exit_flag, context = process_starters(menu_data, meal_period, collected_orders, context)
        if exit_flag:
            return collected_orders, True, context
            
        # Ask about main courses after starters
        speak("Would you like to order any main course dishes now?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like to see our main courses?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_main_courses(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like to see our main courses?")
        
        # Ask about desserts
        speak("Would you like to finish your meal with some desserts?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like any desserts?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_desserts(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like any desserts?")
                
    elif intent == INTENT_DESSERT:
        collected_orders, exit_flag, context = process_desserts(menu_data, meal_period, collected_orders, context)
        if exit_flag:
            return collected_orders, True, context
            
        # Ask about main courses after desserts
        speak("Would you like to order any main course dishes now?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like to see our main courses?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_main_courses(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like to see our main courses?")
        
        # Ask about starters after main courses
        speak("Would you like to start your meal with some starters?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like any starters?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_starters(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like any starters?")
                
    elif intent == INTENT_MENU:
        # First show main courses
        collected_orders, exit_flag, context = process_main_courses(menu_data, meal_period, collected_orders, context)
        if exit_flag:
            return collected_orders, True, context
            
        # Then ask about starters
        speak("Would you like to start your meal with some starters while your main course is being prepared?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like any starters?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_starters(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like any starters?")
        
        # Finally ask about desserts
        speak("Would you like to finish your meal with some desserts?")
        while True:
            user_input = listen()
            if not user_input:
                speak("I didn't catch that. Would you like any desserts?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            if is_positive_response(cmd):
                collected_orders, exit_flag, context = process_desserts(menu_data, meal_period, collected_orders, context)
                if exit_flag:
                    return collected_orders, True, context
                break
            elif is_negative_response(cmd):
                break
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(context[-3:]))
                speak(response)
                context.append(f"Assistant: {response}")
                speak("Would you like any desserts?")
                
    return collected_orders, False, context

# === Main Conversation Flow ===
if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        log_session_start(LOG_FILE)
    else:
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            content = f.read()
            if "===== Customer Session" not in content:
                log_session_start(LOG_FILE)
            elif "===== Customer Session END =====" not in content.splitlines()[-1]:
                log_session_start(LOG_FILE)

    try:
        menu_data = load_menu("restaurant_menu.json")
        menu_helper = RestaurantMenuHelper("restaurant_menu.json")
        db_helper = RestaurantDB(host="localhost", user="root", password="jshaikh@1234", database="restaurant_db")
        
        vs_id = create_vector_store(["restaurant_menu.json", "restaurant_info.txt"])
        assistant = build_assistant(vs_id)
        thread = client.beta.threads.create()

        current_time = get_ist_time()
        meal_period = get_meal_period(menu_data)
        greeting = f"Welcome to Nisarg Hotel! It's currently {current_time.strftime('%I:%M %p')} and we're serving our {meal_period} menu."
        speak(greeting)

        speak("How may I help you today? You can ask about our menu, place an order, or inquire about our restaurant.")

        collected_orders = []
        exit_flag = False
        order_placed = False
        current_order_id = None
        conversation_context = []

        while True:
            if order_placed:
                speak(f"Your order number is {current_order_id}. Please let me know if you need anything else.")
                
                while True:
                    user_input = listen()
                    if not user_input:
                        continue
                        
                    print(f"👤: {user_input}")
                    log_conversation("User", user_input)
                    conversation_context.append(f"User: {user_input}")
                    cmd = user_input.lower().strip()

                    intent = classify_intent(user_input)
                    
                    if intent == INTENT_STATUS:
                        check_order_status(current_order_id)
                        continue
                    elif intent == INTENT_GOODBYE:
                        speak("Thank you for visiting Nisarg Hotel! We hope to see you again soon.")
                        exit_flag = True
                        break
                    else:
                        response = get_ai_response(thread, assistant, user_input, "\n".join(conversation_context[-3:]))
                        speak(response)
                        conversation_context.append(f"Assistant: {response}")
                
                if exit_flag:
                    break
                continue

            user_input = listen()
            if not user_input:
                speak("I didn't hear your request. Could you please repeat that?")
                continue
                
            print(f"👤: {user_input}")
            log_conversation("User", user_input)
            conversation_context.append(f"User: {user_input}")
            cmd = user_input.lower().strip()

            intent = classify_intent(user_input)

            if intent == INTENT_GOODBYE:
                speak("Thank you for visiting Nisarg Hotel! We hope to see you again soon.")
                log_session_end(LOG_FILE)
                break

            if intent == INTENT_GREETING:
                response = "Hello! Welcome to Nisarg Hotel. How may I assist you today?"
                speak(response)
                conversation_context.append(f"Assistant: {response}")
                continue

            if intent == INTENT_GENERAL or intent == INTENT_DESCRIPTION or intent == INTENT_PRICE:
                response = get_ai_response(thread, assistant, user_input, "\n".join(conversation_context[-3:]))
                speak(response)
                conversation_context.append(f"Assistant: {response}")
                continue

            if intent in [INTENT_STARTER, INTENT_DESSERT, INTENT_MENU]:
                collected_orders, exit_flag, conversation_context = handle_dynamic_ordering_flow(
                    user_input, menu_data, meal_period, collected_orders, conversation_context
                )
                if exit_flag:
                    break

                if not collected_orders:
                    continue

                speak("Let me confirm your complete order:")
                for item in collected_orders:
                    speak(f"{item['qty']} {item['name']}")
                total_amount = sum(item["price"] * item["qty"] for item in collected_orders)
                speak(f"The total comes to {format_price(total_amount)}")
                speak("Should I place this order for you?")

                while True:
                    confirm_input = listen()
                    if not confirm_input:
                        speak("I didn't catch that. Should I place this order?")
                        continue

                    print(f"👤: {confirm_input}")
                    log_conversation("User", confirm_input)
                    conversation_context.append(f"User: {confirm_input}")
                    confirm_cmd = confirm_input.lower().strip()

                    if is_negative_response(confirm_cmd):
                        speak("No problem, I've cancelled this order. What would you like instead?")
                        collected_orders = []
                        break

                    if any(word in confirm_cmd for word in ["yes", "please", "confirm"]):
                        table_no = 1
                        try:
                            current_order_id = db_helper.place_order(table_no, collected_orders, total_amount)
                            if current_order_id is None:
                                speak("Apologies, there was an issue placing your order. Let me try that again.")
                                continue
                            bill_id = db_helper.add_billing(current_order_id, total_amount, paid=False)
                            if bill_id is None:
                                speak("There was a small issue with the billing. Let me correct that.")
                                continue

                            response_text = (
                                f"Your order has been placed! Your order number is {current_order_id} "
                                f"and the total is {format_price(total_amount)}. Thank you!"
                            )
                            speak(response_text)
                            collected_orders = []
                            order_placed = True
                            conversation_context.append(f"Assistant: {response_text}")
                            break
                        except Exception as e:
                            print("⚠️ Error processing order:", str(e))
                            speak("My apologies, there was an issue processing your order. Let me try again.")
                    elif any(word in confirm_cmd for word in ["no", "cancel"]):
                        speak("No problem, I've cancelled this order. What would you like instead?")
                        collected_orders = []
                        break
                    else:
                        speak("I didn't quite understand. Should I place this order?")

                if order_placed:
                    continue
            elif intent == INTENT_ORDER:
                collected_orders, exit_flag, conversation_context = handle_direct_order(user_input, meal_period, collected_orders, conversation_context)
                if exit_flag:
                    break

                if collected_orders:
                    speak("Let me confirm your complete order:")
                    for item in collected_orders:
                        speak(f"{item['qty']} {item['name']}")
                    total_amount = sum(item["price"] * item["qty"] for item in collected_orders)
                    speak(f"That will be {format_price(total_amount)} in total")
                    speak("Should I place this order for you?")

                    while True:
                        confirm_input = listen()
                        if not confirm_input:
                            speak("I didn't catch that. Would you like me to place this order?")
                            continue

                        print(f"👤: {confirm_input}")
                        log_conversation("User", confirm_input)
                        conversation_context.append(f"User: {confirm_input}")
                        confirm_cmd = confirm_input.lower().strip()

                        if is_negative_response(confirm_cmd):
                            speak("No problem at all. I've cancelled this order. What would you like instead?")
                            collected_orders = []
                            break

                        if any(word in confirm_cmd for word in ["yes", "please", "confirm"]):
                            table_no = 1
                            try:
                                current_order_id = db_helper.place_order(table_no, collected_orders, total_amount)
                                if current_order_id is None:
                                    speak("Apologies, there was an issue placing your order. Let me try again.")
                                    continue
                                bill_id = db_helper.add_billing(current_order_id, total_amount, paid=False)
                                if bill_id is None:
                                    speak("There was a small billing issue. Let me correct that.")
                                    continue

                                response_text = (
                                    f"Your order is confirmed! Your order number is {current_order_id} "
                                    f"and the total is {format_price(total_amount)}. Thank you!"
                                )
                                speak(response_text)
                                collected_orders = []
                                order_placed = True
                                conversation_context.append(f"Assistant: {response_text}")
                                break
                            except Exception as e:
                                print("⚠️ Error processing order:", str(e))
                                speak("My apologies, there was an error processing your order. Let me try again.")
                        elif any(word in confirm_cmd for word in ["no", "cancel"]):
                            speak("No problem at all. I've cancelled this order. What would you like instead?")
                            collected_orders = []
                            break
                        else:
                            speak("I didn't quite understand. Would you like me to place this order?")

                    if order_placed:
                        continue
            else:
                response = get_ai_response(thread, assistant, user_input, "\n".join(conversation_context[-3:]))
                speak(response)
                conversation_context.append(f"Assistant: {response}")

            if exit_flag:
                break
    finally:
        log_session_end(LOG_FILE)