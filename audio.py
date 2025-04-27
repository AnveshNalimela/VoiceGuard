# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from io import BytesIO

# # Cache the model loading for performance
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model("D:\MiniprojectII\deepfake_audio_detector.h5", compile=False)
#     return model

# model = load_model()

# # Function to load audio from uploaded file
# def load_audio(uploaded_file):
#     try:
#         audio_bytes = uploaded_file.getvalue()
#         audio, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
#         return audio, sr
#     except Exception as e:
#         st.error(f"Error loading audio: {e}")
#         return None, None

# # Function to extract fixed-size MFCC features
# def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512, fixed_length=200):
#     try:
#         # Extract MFCCs
#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

#         # Ensure fixed shape (13, 200) using padding or truncation
#         if mfcc.shape[1] < fixed_length:
#             pad_width = fixed_length - mfcc.shape[1]
#             mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
#         else:
#             mfcc = mfcc[:, :fixed_length]

#         return mfcc
#     except Exception as e:
#         st.error(f"Error extracting MFCC features: {e}")
#         return None

# # Function to plot MFCC features
# def plot_mfcc(mfcc, sr, hop_length=512):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, cmap="viridis")
#     ax.set_title("MFCC Features")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("MFCC Coefficients")
#     plt.colorbar()
#     return fig

# # Function to predict deepfake audio
# def predict_audio(mfcc):
#     try:
#         # Reshape MFCCs to match model input
#         mfcc = mfcc[..., np.newaxis]  # Shape: (13, 200, 1)
#         mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
#         prediction = model.predict(mfcc)
#         label = "real audio" if prediction[0][0] > 0.5 else "fake audio"
#         return label, prediction[0][0]
#     except Exception as e:
#         st.error(f"Error making prediction: {e}")
#         return None, None

# # Streamlit app interface
# st.title("Deepfake Audio Detector")
# st.write("Upload an audio file to determine if it's real or a deepfake using a pre-trained model.")

# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "flac", "mp3"])

# if uploaded_file is not None:
#     audio, sr = load_audio(uploaded_file)
#     if audio is not None:
#         mfcc = extract_mfcc(audio, sr)
#         if mfcc is not None:
#             # Display MFCC plot
#             fig = plot_mfcc(mfcc, sr)
#             st.pyplot(fig)

#             # Make and display prediction
#             label, prob = predict_audio(mfcc)
#             if label is not None:
#                 st.write(f"**Prediction:** {label} (Probability of being real: {prob:.4f})")

import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration and theme
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading for performance
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("D:\MiniprojectII\deepfake_audio_detector.h5", compile=False)
    return model

# Function to load audio from uploaded file
def load_audio(uploaded_file):
    try:
        audio_bytes = uploaded_file.getvalue()
        audio, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

# Function to extract fixed-size MFCC features
def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512, fixed_length=200):
    try:
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Ensure fixed shape (13, 200) using padding or truncation
        if mfcc.shape[1] < fixed_length:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :fixed_length]

        return mfcc
    except Exception as e:
        st.error(f"Error extracting MFCC features: {e}")
        return None

# Function to plot MFCC features
def plot_mfcc(mfcc, sr, hop_length=512):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, cmap="viridis")
    ax.set_title("MFCC Features", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("MFCC Coefficients", fontsize=12)
    plt.colorbar(format='%+2.0f dB')
    fig.tight_layout()
    return fig

# Function to predict deepfake audio
def predict_audio(mfcc):
    try:
        # Reshape MFCCs to match model input
        mfcc = mfcc[..., np.newaxis]  # Shape: (13, 200, 1)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        prediction = model.predict(mfcc)
        label = "Real Audio" if prediction[0][0] > 0.5 else "Fake Audio"
        return label, prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Add custom CSS for better styling
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 20px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        margin: 20px 0px;
    }
    .real-result {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    .fake-result {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #0066cc;
    }
    .logo-text {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(45deg, #0066cc, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .header {
        display: flex;
        align-items: center;
        background-color: #f8f9fa;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-icon {
        font-size: 32px;
        margin-right: 15px;
    }
    .header-title {
        color: #0066cc;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .header-nav {
        margin-left: auto;
    }
    .header-nav a {
        color: #0066cc;
        text-decoration: none;
        margin-left: 20px;
        font-weight: 500;
    }
    .header-nav a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to display the header
def display_header():
    header_html = """
    <div class="header">
        <div class="header-icon">🎵</div>
        <h1 class="header-title">AudioGuard</h1>
        <div class="header-nav">
            <a href="javascript:void(0);" onclick="goToHome()">Home</a>
            <a href="javascript:void(0);" onclick="goToDetect()">Detect Audio</a>
        </div>
    </div>
    
    <script>
    function goToHome() {
        // This uses Streamlit's event handling to change page
        const streamlitDoc = window.parent.document;
        const buttons = Array.from(streamlitDoc.querySelectorAll('button'));
        const homeButton = buttons.find(el => el.innerText === 'Return to Home');
        if (homeButton) homeButton.click();
    }
    
    function goToDetect() {
        // This uses Streamlit's event handling to change page
        const streamlitDoc = window.parent.document;
        const buttons = Array.from(streamlitDoc.querySelectorAll('button'));
        const detectButton = buttons.find(el => el.innerText === 'Start Detection Process');
        if (detectButton) detectButton.click();
    }
    </script>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def home_page():
    display_header()
    
    st.markdown("<h1 class='logo-text'>Deepfake Audio Detector</h1>", unsafe_allow_html=True)
    
    st.markdown("### Welcome to the Deepfake Audio Detection System")
    st.markdown("""
    This application uses artificial intelligence to analyze audio files and determine 
    if they are genuine recordings or AI-generated deepfakes.
    """)
    
    # About Us Section
    st.markdown("## About Us")
    st.markdown("""
    Our system utilizes advanced machine learning techniques to detect subtle patterns in audio that 
    indicate artificial generation. With the rise of AI-generated voice cloning and deepfakes, 
    our tool provides a reliable way to verify audio authenticity.
    
    The model has been trained on thousands of real and synthetic audio samples to recognize the 
    telltale signs of manipulation that may not be audible to the human ear.
    """)
    
    # Features Section
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        # st.markdown("### 🔍 Advanced Detection")
        # st.markdown("""
        # Our system analyzes MFCC (Mel-frequency cepstral coefficients) features 
        # to identify patterns characteristic of synthetic speech.
        # """)
        # st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Visual Analysis")
        st.markdown("""
        See the audio's spectral representation and understand why the system 
        made its determination.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Results")
        st.markdown("""
        Get instant verification with clear indicators showing 
        whether audio is likely real or fake.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        # st.markdown("### 🛡️ Professional Tool")
        # st.markdown("""
        # Designed for journalists, content moderators, and security professionals 
        # who need to verify audio authenticity.
        # """)
        # st.markdown("</div>", unsafe_allow_html=True)
    
    # Navigation button to upload page
    if st.button("Start Detection Process"):
        st.session_state.page = "upload"
        st.rerun()

def upload_page():
    display_header()
    
    st.markdown("## Upload Audio for Analysis")
    st.markdown("""
    Upload an audio file to determine if it's authentic or AI-generated. 
    Supported formats: WAV, FLAC, and MP3.
    """)
    
    # Load the model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "flac", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        with st.spinner('Processing audio...'):
            audio, sr = load_audio(uploaded_file)
            
            if audio is not None:
                mfcc = extract_mfcc(audio, sr)
                
                if mfcc is not None:
                    # Create two columns for analysis display
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("### MFCC Feature Analysis")
                        fig = plot_mfcc(mfcc, sr)
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### Detection Results")
                        label, prob = predict_audio(mfcc)
                        
                        if label is not None:
                            # Display result with appropriate styling
                            if "Real" in label:
                                st.markdown(f"<div class='real-result'>{label}</div>", unsafe_allow_html=True)
                                st.progress(float(prob))
                                st.markdown(f"<p style='text-align:center;'>Confidence: {prob:.2f}</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='fake-result'>{label}</div>", unsafe_allow_html=True)
                                st.progress(float(1 - prob))
                                st.markdown(f"<p style='text-align:center;'>Confidence: {1-prob:.2f}</p>", unsafe_allow_html=True)
                            
                            # Additional explanation
                            st.markdown("### Interpretation")
                            if "Real" in label:
                                st.markdown("""
                                This audio shows characteristics consistent with natural human speech. 
                                The spectral patterns and acoustic features align with those typically 
                                found in genuine recordings.
                                """)
                            else:
                                st.markdown("""
                                This audio shows characteristics consistent with AI-generated speech.
                                The model has detected subtle patterns that are typical of synthetic
                                audio generation techniques.
                                """)
    
    # Button to return to home page - hidden but used by JavaScript navigation
    if st.button("Return to Home", key="return_home"):
        st.session_state.page = "home"
        st.rerun()

# Main app logic
def main():
    add_custom_css()
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Display the appropriate page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "upload":
        upload_page()

if __name__ == "__main__":
    main()
