import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
import io

# Download NLTK stopwords
nltk.download('stopwords')

# Function to read CSV file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to detect the language of the text
def detect_language(text):
    try:
        translator = Translator()
        detection = translator.detect(text)
        return detection.lang
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return None

# Function to translate text to a target language
def translate_text(text, target_lang='en'):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return text

# Function to fetch the Google Fonts CSS and extract the font URL
def fetch_font_url_from_google_fonts(language_code):
    # Define font URLs for different languages
    font_urls = {
        'en': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'fr': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'de': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'it': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'es': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'ru': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap',
        'pt': 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap'
        # Add more languages and their corresponding font URLs as needed
    }
    
    # Return the font URL based on the language code
    if language_code in font_urls:
        return font_urls[language_code]
    else:
        raise ValueError(f"Font URL not found for language '{language_code}'.")

# Function to fetch Google Font online
def fetch_google_font(font_url):
    response = requests.get(font_url)
    response.raise_for_status()
    return response.content  # Return font data directly as bytes

# Function to remove stopwords from the text
def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Function to generate word cloud from CSV
def generate_wordcloud_from_csv(df, text_column, target_lang=None, css_url=None):
    try:
        # Check if the specified column exists in the DataFrame
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in the CSV file.")
        
        # Concatenate all text data from the specified column
        all_text = " ".join(df[text_column].astype(str).tolist())
        
        # Detect language
        detected_language = detect_language(all_text)
        if detected_language:
            st.write(f"Detected language: {detected_language}")
        else:
            st.write("Could not detect language. Skipping translation.")
        
        # Translate text to English
        st.write("Translating text to English...")
        all_text = translate_text(all_text, 'en')
        
        # Remove stopwords
        all_text = remove_stopwords(all_text, 'english')
        
        # Translate text to target language if specified
        if target_lang and target_lang != 'en':
            st.write(f"Translating text to {target_lang}...")
            all_text = translate_text(all_text, target_lang)
        
        # Count word frequencies
        word_counts = Counter(all_text.split())
        
        # Fetch the Google Font URL from the provided CSS URL
        if css_url:
            font_url = fetch_font_url_from_google_fonts(target_lang)
            font_data = fetch_google_font(font_url)
        else:
            font_data = None
        
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=None).generate_from_frequencies(word_counts)
        
        # Display the word cloud
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

        # Save word cloud to a BytesIO object
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer

    except FileNotFoundError:
        st.error(f"File not found. Please check the file path.")
    except KeyError as e:
        st.error(e)
    except requests.RequestException as e:
        st.error(f"Error fetching font: {e}")
    except TypeError as e:
        st.error(f"Type error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app
def main():
    # CSS to inject contained in a string
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 16px;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Word Cloud Generator")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
        
        text_column = st.selectbox("Select the column containing text data", df.columns)
        
        languages = {
            "English": "en",
            "French": "fr",
            "Italian": "it",
            "German": "de",
            "Russian": "ru",
            "Spanish": "es",
            "Portuguese": "pt"
        }
        
        target_lang = st.selectbox("Select the language for the word cloud", list(languages.keys()), index=0)
        target_lang_code = languages[target_lang]
        
        if st.button("Generate Word Cloud"):
            css_url = 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap'
            img_buffer = generate_wordcloud_from_csv(df, text_column, target_lang_code, css_url)
            if img_buffer:
                st.download_button(label="Download Word Cloud", data=img_buffer, file_name="wordcloud.png", mime="image/png")
    
    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app was created by Nikhil Saxena for the LMRI team use. It can create word clouds in different languages from the provided dataset.
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Automatically detects the input data language.
            - Provides functionality to create word clouds in different languages (default is English)
            ---
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
