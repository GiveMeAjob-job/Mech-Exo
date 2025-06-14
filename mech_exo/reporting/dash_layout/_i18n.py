"""
Internationalization (i18n) module for Mech-Exo dashboard
Phase P11 Week 3 Day 5: Support for English/Chinese language switching

Features:
- Browser language detection via Accept-Language header
- Language switching dropdown in Dash UI
- JSON-based translation files
- Default fallback to English
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class I18nManager:
    """Manages internationalization for the dashboard"""
    
    def __init__(self):
        self.current_language = 'en'
        self.translations = {}
        self.supported_languages = ['en', 'zh-Hans']
        self.default_language = 'en'
        
        # Load all translation files
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files from i18n directory"""
        i18n_dir = Path(__file__).parent.parent / 'i18n'
        
        for lang in self.supported_languages:
            lang_file = i18n_dir / f'{lang}.json'
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang] = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Failed to load {lang} translations: {e}")
                    if lang == self.default_language:
                        # Fallback minimal translations for English
                        self.translations[lang] = {
                            "app": {"title": "Mech-Exo Risk Control"},
                            "errors": {"loading_failed": "Loading Failed"}
                        }
    
    def set_language(self, language: str):
        """Set the current language"""
        if language in self.supported_languages:
            self.current_language = language
        else:
            print(f"Warning: Language '{language}' not supported, falling back to {self.default_language}")
            self.current_language = self.default_language
    
    def detect_browser_language(self, accept_language_header: Optional[str] = None) -> str:
        """
        Detect language from browser Accept-Language header
        
        Args:
            accept_language_header: HTTP Accept-Language header value
            
        Returns:
            Detected language code or default language
        """
        if not accept_language_header:
            return self.default_language
        
        # Parse Accept-Language header (e.g., "zh-CN,zh;q=0.9,en;q=0.8")
        try:
            languages = []
            for item in accept_language_header.split(','):
                if ';q=' in item:
                    lang, quality = item.split(';q=')
                    languages.append((lang.strip(), float(quality)))
                else:
                    languages.append((item.strip(), 1.0))
            
            # Sort by quality (preference)
            languages.sort(key=lambda x: x[1], reverse=True)
            
            # Check for supported languages
            for lang, _ in languages:
                # Handle Chinese variants
                if lang.startswith('zh'):
                    if 'zh-Hans' in self.supported_languages:
                        return 'zh-Hans'
                # Handle exact matches
                elif lang in self.supported_languages:
                    return lang
                # Handle language prefix (e.g., 'en-US' -> 'en')
                elif lang.split('-')[0] in self.supported_languages:
                    return lang.split('-')[0]
            
        except (ValueError, IndexError):
            pass
        
        return self.default_language
    
    def get_text(self, key_path: str, **kwargs) -> str:
        """
        Get translated text by key path
        
        Args:
            key_path: Dot-separated path to translation key (e.g., 'dashboard.title')
            **kwargs: Format arguments for string interpolation
            
        Returns:
            Translated text or key path if not found
        """
        # Get translations for current language
        translations = self.translations.get(self.current_language, {})
        
        # Navigate through nested keys
        keys = key_path.split('.')
        value = translations
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                # Fallback to English if current language doesn't have the key
                if self.current_language != self.default_language:
                    fallback_translations = self.translations.get(self.default_language, {})
                    fallback_value = fallback_translations
                    for fallback_key in keys:
                        if isinstance(fallback_value, dict) and fallback_key in fallback_value:
                            fallback_value = fallback_value[fallback_key]
                        else:
                            return key_path  # Return key path if not found anywhere
                    value = fallback_value
                else:
                    return key_path  # Return key path if not found
                break
        
        # Apply string formatting if provided
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                pass
        
        return str(value) if value is not None else key_path
    
    def get_language_options(self) -> list:
        """Get available language options for dropdown"""
        language_names = {
            'en': 'English',
            'zh-Hans': '简体中文'
        }
        
        return [
            {'label': language_names.get(lang, lang), 'value': lang}
            for lang in self.supported_languages
        ]
    
    def get_current_language(self) -> str:
        """Get current language code"""
        return self.current_language
    
    def is_rtl(self) -> bool:
        """Check if current language is right-to-left"""
        rtl_languages = ['ar', 'he', 'fa']  # Arabic, Hebrew, Persian
        return self.current_language in rtl_languages


# Global i18n manager instance
i18n = I18nManager()


def t(key_path: str, **kwargs) -> str:
    """
    Convenience function for translation
    
    Args:
        key_path: Dot-separated path to translation key
        **kwargs: Format arguments
        
    Returns:
        Translated text
    """
    return i18n.get_text(key_path, **kwargs)


def set_language_from_request(request_headers: Dict[str, Any] = None):
    """
    Set language based on HTTP request headers
    
    Args:
        request_headers: Dictionary of HTTP headers
    """
    if request_headers and 'Accept-Language' in request_headers:
        detected_lang = i18n.detect_browser_language(request_headers['Accept-Language'])
        i18n.set_language(detected_lang)


def create_language_selector():
    """
    Create language selector dropdown component for Dash
    
    Returns:
        Dash dropdown component configuration
    """
    try:
        import dash_core_components as dcc
    except ImportError:
        try:
            from dash import dcc
        except ImportError:
            # Return basic config if Dash not available
            return {
                'type': 'dropdown',
                'options': i18n.get_language_options(),
                'value': i18n.get_current_language()
            }
    
    return dcc.Dropdown(
        id='language-selector',
        options=i18n.get_language_options(),
        value=i18n.get_current_language(),
        clearable=False,
        style={
            'width': '150px',
            'font-size': '14px'
        },
        placeholder=t('settings.language')
    )


# Example usage in Dash callbacks
def language_callback_example():
    """
    Example callback for handling language changes in Dash
    This should be adapted to your specific dashboard implementation
    """
    pass
    # from dash.dependencies import Input, Output, State
    # from dash import callback_context
    # 
    # @app.callback(
    #     Output('page-content', 'children'),
    #     [Input('language-selector', 'value')],
    #     [State('url', 'pathname')]
    # )
    # def update_language(selected_language, pathname):
    #     i18n.set_language(selected_language)
    #     # Refresh page content with new language
    #     return render_page_content(pathname)


if __name__ == "__main__":
    # Test the i18n system
    print("Testing i18n system...")
    
    # Test English (default)
    print(f"English title: {t('app.title')}")
    print(f"English dashboard: {t('dashboard.title')}")
    
    # Test Chinese
    i18n.set_language('zh-Hans')
    print(f"Chinese title: {t('app.title')}")
    print(f"Chinese dashboard: {t('dashboard.title')}")
    
    # Test browser language detection
    test_headers = [
        "zh-CN,zh;q=0.9,en;q=0.8",
        "en-US,en;q=0.9",
        "zh-Hans-CN,zh-Hans;q=0.9,zh;q=0.8,en;q=0.7"
    ]
    
    for header in test_headers:
        detected = i18n.detect_browser_language(header)
        print(f"Header '{header}' -> Detected: {detected}")
    
    # Test language options
    print(f"Available languages: {i18n.get_language_options()}")