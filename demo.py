import streamlit as st
from email_validator import validate_email, EmailNotValidError
import mysql.connector
from mysql.connector import Error
import bcrypt
import base64
from audiorecorder import audiorecorder
# from pydub import AudioSegment
import os
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr



with open("styles.css") as f:
    try:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.rerun()

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_data = get_image_base64("images/68184dfa4e4cb332c4a54662-removebg-preview.png")
st.set_page_config(
    page_title="Himanshu Pethe",  # ✅ This changes the browser tab title
    page_icon=get_image_base64("images/68184dfa4e4cb332c4a54662-removebg-preview.png"),                             # Optional: Adds a favicon (emoji or path to image)
    layout="wide",                              # Optional: wide or centered layout
    initial_sidebar_state="expanded"            # Optional: collapsed/expanded
)
# --- Password Utils ---
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# --- MySQL Connection ---
def create_connection():
    return mysql.connector.connect(
        host='localhost',
        port=3306,
        user='root',
        password='Him@nshu2003',
        database='User',
        auth_plugin='mysql_native_password'
    )

# --- Session Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'Login'

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# --- Auth Pages ---
def register_user(name, email, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        hashed_pwd = hash_password(password)
        cursor.execute("INSERT INTO userdata (user_name, email, password) VALUES (%s, %s, %s)", 
                       (name, email, hashed_pwd))
        conn.commit()
        st.success("✅ Registered successfully! Please login.")
        st.session_state.page = "Login"
        st.rerun()  # <-- so it shows the login form after success
    except mysql.connector.IntegrityError:
        st.error("❌ Email already registered.")
    finally:
        cursor.close()
        conn.close()


def login_user(email, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, password FROM userdata WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        name, hashed_pwd = result
        if check_password(password, hashed_pwd):
            st.session_state.authenticated = True
            st.session_state.user_name = name
            st.session_state.page = "Intro"
            st.success(f"✅ Welcome back, {name}!")
            st.rerun()  # <-- force page change to reflect
        else:
            st.error("❌ Incorrect password.")
    else:
        st.error("❌ Email not found.")


# --- Pages ---
def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login_user(email, password)

def register_page():
    st.title("Register")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Submit Registration"):
        try:
            validate_email(email)
            register_user(name, email, password)
        except EmailNotValidError as e:
            st.error(f"❌ Invalid email: {e}")

def intro_page():
    st.title(f"Welcome, {st.session_state.user_name}!")
    st.write("You are now logged in. Use the sidebar to explore more pages.")

def Linear_Regression():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    data=pd.read_csv("Salary_Data.csv")
    st.title("Salary Prediction")
    st.text("""Welcome to our Salary Prediction platform! This project demonstrates how Linear Regression, a foundational machine learning algorithm, can be used to predict salaries based on experience. By training on historical salary data, our model identifies patterns and builds a relationship between years of experience and annual salary. Whether you're a beginner exploring data science or an HR analyst estimating compensation, this tool showcases how data-driven insights can enhance decision-making. Simply visualize, predict, and understand how experience impacts earnings—powered by real-world data and machine learning.""")
    
    st.write(data)
    st.text("""


""")
    
    st.write("### Step 1: Scatter Plot")
    st.write("""To begin our salary prediction journey, we first visualize the data using a scatter plot. This plot helps us understand the pattern or correlation between:
             
X-axis: Years of Experience       
Y-axis: Salary     
              
Each point on the graph represents one individual’s data. If there's a clear trend — like a rising pattern — it indicates that as experience increases, salary tends to increase, which is ideal for applying Linear Regression.""")
    st.scatter_chart(data=data, x="YearsExperience", y="Salary", x_label="Experience (in Year)", y_label="Salary")
    
    st.text("""


""")
    
    st.write("### Step 2: Spliting Dataset")
    st.write("""After visualizing the data, the next step is to prepare it for training and testing. We divide the dataset into two parts:

Training Set: Used to teach the model the relationship between experience and salary.

Testing Set: Used to evaluate how well the model performs on new, unseen data.

📌 Typically, we use:

80% of the data for training

20% of the data for testing

This step is crucial because it prevents overfitting and ensures that the model can generalize well to future predictions.""")
    
    y = data.iloc[:,1].values
    x = data.iloc[:,0].values

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)
    


def record_audio(filename="output.wav", duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio_data)
    return filename

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "❗ Could not understand the audio."
    except sr.RequestError:
        return "❗ Could not request results from the speech recognition service."

def LTM():
    st.title("🎙️ Basic Voice Recorder & Translator")
    st.write("Click the button below to record your voice and transcribe it to text.")

    duration = st.slider("Select Duration (in seconds)", 5, 1000, 5)

    if st.button("🔴 Start Recording"):
        filename = record_audio(duration=duration)
        st.success("✅ Recording Complete!")
        
        st.audio(filename, format="audio/wav")

        if st.button("🗣️ Transcribe to Text"):
            with st.spinner("Transcribing..."):
                result = transcribe_audio(filename)
            st.subheader("📝 Transcription:")
            st.write(result)

# --- Main App Routing ---

st.sidebar.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{img_data}" width="100"/>
        <h1>Himanshu Pethe</h1>
    </div>
""", unsafe_allow_html=True)

if st.session_state.authenticated:
    # Show logout + page switch
    st.sidebar.markdown("""
        <div class="animated-gradient-bar"></div>
        """, unsafe_allow_html=True)

    st.sidebar.text(f"{st.session_state.user_name}")


    if st.sidebar.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.session_state.user_name = ""
        st.session_state.page = "Login"
        st.rerun()

    # Page Selection
    page_options = {
        "Intro": intro_page,
        "Linear Regression": Linear_Regression,
        "Language Translation": LTM
    }
    selected_page = st.sidebar.selectbox("Select Page", list(page_options.keys()), index=list(page_options.keys()).index(st.session_state.page))
    st.session_state.page = selected_page

    # Show selected page
    page_options[st.session_state.page]()

else:
    # Show login/register options
    if st.sidebar.button("Login"):
        st.session_state.page = "Login"
    if st.sidebar.button("Register"):
        st.session_state.page = "Register"

    # Show auth page based on state
    if st.session_state.page == "Register":
        register_page()
    else:
        login_page()
