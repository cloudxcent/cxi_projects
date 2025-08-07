# Restaurant AI Assistant Configuration Setup

## Overview
This project now uses a JSON configuration file instead of environment variables for better management of settings.

## Setup Instructions

### 1. Create Configuration File
Copy `config_sample.json` to `config.json`:
```bash
cp config_sample.json config.json
```

### 2. Update Configuration Values

Edit `config.json` and replace the placeholder values:

#### Azure OpenAI Settings
```json
"azure_openai": {
  "api_key": "your-actual-openai-key",
  "endpoint": "https://your-resource.openai.azure.com/",
  "deployment": "your-deployment-name",
  "version": "2024-05-01-preview"
}
```

#### Azure Speech Settings
```json
"azure_speech": {
  "key": "your-actual-speech-key",
  "region": "your-region",
  "voice_name": "en-IN-NeerjaNeural"
}
```

#### Database Settings
```json
"database": {
  "host": "localhost",
  "user": "your-db-user",
  "password": "your-db-password", 
  "database": "restaurant_db"
}
```

### 3. File Paths
The following files should be in the same directory as the script:
- `restaurant_menu.json` - Menu data
- `restaurant_info.txt` - Restaurant information
- `restaurant_assistant.log` - Will be created automatically

### 4. Customization

#### Menu Categories
You can customize how the system categorizes menu items by modifying:
```json
"menu_categories": {
  "main_course_keywords": ["main", "curry", "rice", ...],
  "exclude_keywords": ["starter", "dessert", ...]
}
```

#### Speech Settings
Adjust speech synthesis rate:
```json
"speech_settings": {
  "prosody_rate": "1.2"
}
```

## Security Notes
- Never commit `config.json` with real credentials to version control
- The sample file `config_sample.json` contains placeholder values
- Consider using different config files for development/production

## Migration from Environment Variables
If you were previously using environment variables, copy those values to the corresponding fields in `config.json`.

## Error Messages
- "Configuration file 'config.json' not found" - Create config.json from the sample
- "Please update azure_openai.api_key" - Replace placeholder values with actual credentials
- "Missing configuration key" - Check that all required fields are present in config.json
