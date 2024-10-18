import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    .stSelectbox div {{
        color: white;
    }}
    .stSubheader {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("image.jpg")  # Path to your uploaded image
set_background(image_base64)



# Load the model
with open(r"C:\Users\91939\Desktop\AI&DS\Data science projects\PulsarStarClassification\final_model.pkl", 'rb') as f:
    model = pickle.load(f)


st.title("Pulsar Star Classification")

st.write(
    "This project aims to classify pulsar stars using machine learning models based on various statistical features. "
    "The dataset used contains multiple features such as mean, standard deviation, skewness, and kurtosis of the "
    "pulsar signal's intensity, which are critical for distinguishing between pulsar stars and other astronomical "
    "bodies. The machine learning models trained in this project analyze these features to accurately predict whether "
    "a given observation is a pulsar star or not."
)

# Load the dataset (if needed)
df = pd.read_csv(r"C:\Users\91939\Desktop\AI&DS\Data science projects\PulsarStarClassification\Pulsar.csv")

st.sidebar.title("About the Project")
st.sidebar.info(
    """
    - This app classifies whether a star is a pulsar or not.
    - Pulsars are neutron stars that emit beams of electromagnetic radiation.
    - The model was trained using data from astronomical observations.
    """
)




col1,col2 = st.columns(2)
with col1:
    st.subheader("Pulsar Star image")
    st.image(r"C:\Users\91939\Desktop\AI&DS\Data science projects\PulsarStarClassification\image1.jpeg", caption="An artistic impression of a pulsar.")
with col2:
    st.info("Information")
    st.write("Pulsars are rapidly spinning neutron stars formed from the collapse of massive stars (4-8 times the mass of the Sun) after a supernova explosion."
             "As the core collapses, the star spins faster " 
             "Some millisecond pulsars spin even faster by drawing material from a companion star in a binary system.")

st.subheader("Input Features")

# Input features from user
Mean_Integrated = st.number_input("Mean of the integrated profile", value=0.0)
SD = st.number_input("Standard deviation of the integrated profile", value=0.0)
EK = st.number_input("Excess kurtosis of the integrated profile", value=0.0)
Skewness = st.number_input("Skewness of the integrated profile", value=0.0)
Mean_DMSNR_Curve = st.number_input("Mean of the DM-SNR curve", value=0.0)
SD_DMSNR_Curve = st.number_input("Standard deviation of the DM-SNR curve", value=0.0)
EK_DMSNR_Curve = st.number_input("Excess kurtosis of the DM-SNR curve", value=0.0)
Skewness_DMSNR_Curve = st.number_input("Skewness of the DM-SNR curve", value=0.0)

# Collect input features into a list or numpy array
input_features = np.array([[Mean_Integrated, SD, EK, Skewness, Mean_DMSNR_Curve,
                            SD_DMSNR_Curve, EK_DMSNR_Curve, Skewness_DMSNR_Curve]])

# Predict button
if st.button('Predict'):
    # Make the prediction
    prediction = model.predict(input_features)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The star is likely a pulsar.")
    else:
        st.warning("The star is likely not a pulsar.")
        
col_names = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']      
df.columns = col_names
# Feature Distribution Visualization
st.subheader("Feature Distributions")
selected_feature = st.selectbox("Select a feature to visualize:", df.columns)

# Histogram
plt.figure(figsize=(10, 4))
sns.histplot(df[selected_feature], bins=30, kde=True)
plt.title(f'Distribution of {selected_feature}')
plt.xlabel(selected_feature)
plt.ylabel('Frequency')
st.pyplot(plt)

# Boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x=df[selected_feature])
plt.title(f'Boxplot of {selected_feature}')
st.pyplot(plt)