# Voice Guard:
This project, VoiceGuard, aims to address this growing threat by developing a robust 
detection framework that can distinguish between real and deepfake audio. Leveraging 
a hybrid deep learning model combining Convolutional Neural Networks (CNNs) and 
Long Short-Term Memory (LSTM) units, the system is designed to capture both spatial 
and temporal patterns in audio data. Key features such as Mel Frequency Cepstral 
Coefficients (MFCCs) and additional spectral metrics provide the foundation for 
accurate classification. The ASVspoof 2019 dataset, which includes both genuine and 
manipulated audio samples, serves as the primary benchmark for training and 
evaluation.
Implemnatation of the Project:
![WhatsApp Image 2025-04-25 at 2 03 39 PM](https://github.com/user-attachments/assets/214e1237-642f-4b93-94c6-b16280151415)

### Home Page and System Features Overview
![WhatsApp Image 2025-04-25 at 10 14 03 AM](https://github.com/user-attachments/assets/414b4dff-0146-408b-80b7-4e1b740e93d9)
Fig indicates homepage of VoiceGuard highlights the system's purpose and key 
features. It outlines components like advanced detection, visual MFCC analysis, quick 
results, and its professional utility for verifying audio authenticity. 

###  Upload Interface of VoiceGuard   
![WhatsApp Image 2025-04-25 at 10 14 02 AM](https://github.com/user-attachments/assets/a12bf738-69f1-450c-a845-0098c7795e7a)
Fig illustrates this screen allows users to upload audio files in WAV, FLAC, or MP3 
format for analysis. The drag-and-drop interface supports files up to 200MB and directs 
the uploaded file to the detection pipeline. 


### Results
![WhatsApp Image 2025-04-25 at 12 23 32 AM](https://github.com/user-attachments/assets/735c2cc9-b8c8-443d-b6e4-83186a066a5a)
![WhatsApp Image 2025-04-25 at 10 14 00 AM](https://github.com/user-attachments/assets/ccf6bbd6-d60d-469d-b6d6-a0af46b52062)

## Steps to Run:

1.Clone the Repo
2.Install all the required libaries and Packages

```code
pip install -r requirements.txt
```

3 Run the Application

```
streamlit run audio.py
```




