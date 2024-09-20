import os
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import requests


def validate_Google_api_key(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    request = requests.get(url)
    print(request)
    if request.status_code == 200:
        return True
    else:
        return False


def validate_Groq_api_key(api_key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    request = requests.get(url, headers=headers)
    if request.status_code == 200:
        return True
    else:
        return False


st.set_page_config(page_title="NoteCraft AI - Get Access", page_icon="🔑")
cookies = EncryptedCookieManager(
    prefix="AL-Sayed1/NOTECRAFT_AI_WEB",
    password=os.environ.get("COOKIES_PASSWORD", "COOKIES_PASSWORD"),
)

if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.stop()


if "GOOGLE_API_KEY" in cookies and "GROQ_API_KEY" in cookies:
    st.success(
        "You have already set both API keys, you can change them if they don't work."
    )


GOOGLE_API_KEY = st.text_input(
    "GOOGLE API KEY:",
    type="password",
    value=cookies["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in cookies else "",
)
GROQ_API_KEY = st.text_input(
    "GROQ API KEY:",
    type="password",
    value=cookies["GROQ_API_KEY"] if "GROQ_API_KEY" in cookies else "",
)


if st.button("SAVE") and GOOGLE_API_KEY and GROQ_API_KEY:
    cookies["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    cookies["GROQ_API_KEY"] = GROQ_API_KEY
    cookies.save()  # Force saving the cookies now, without a rerun
    if (
        cookies["GOOGLE_API_KEY"] == GOOGLE_API_KEY
        and cookies["GROQ_API_KEY"] == GROQ_API_KEY
    ):
        if validate_Google_api_key(GOOGLE_API_KEY) and validate_Groq_api_key(
            GROQ_API_KEY
        ):
            st.success(
                "Access has been granted successfully! You can now use NoteCraft AI!"
            )
        elif validate_Google_api_key(GOOGLE_API_KEY):
            st.error("Groq API key seems invalid.")
        elif validate_Groq_api_key(GROQ_API_KEY):
            st.error("Google API key seems invalid.")
        else:
            st.error("API keys seem invalid.")
    else:
        st.error("There was an error while saving the API keys :(")

st.caption(
    f"Get the api keys from the [Google AI studio](https://aistudio.google.com/) and [Groq](https://console.groq.com/keys) websites."
)
