
import streamlit as st
import tensorflow as tf
import numpy as np

# Function to load model and make predictions
@st.cache(allow_output_mutation=True)  # Cache the model to avoid loading it multiple times
def load_model():
    return tf.keras.models.load_model("F:\project\model_avg_20_inception.h5")


# Function for model prediction with resized input images
def model_prediction(model, test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))  # Resize input image
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Main function to run Streamlit app
def main():
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Medicinal Herbs Recognition"])

    if app_mode == "Home":
        st.header("MEDICINAL HERBS RECOGNITION SYSTEM")
        image_path = "homepage.jpg"
        st.image(image_path, width=500)
        st.markdown("""
    Welcome to the Medicinal Herbs Recognition System! 🌿🔍
    
    Our mission is to help in identifying Medicinal Herbs efficiently. Upload an image of a Herb, and our system will analyze it to detect its name.
     Together, let's utilize more and more natural medicines!

    ### How It Works
    1. **Upload Image:** Go to the **Medicinal Herbs Recognition** page and upload an image of a plant.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify the plant.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate herbs detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Medicinal Herbs Recoginition** page in the sidebar to upload an image and experience the power of our Plant Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

    elif app_mode == "About":
        st.header("About")
        st.title("NOIDA INSTITUTE OF ENGINEERING AND TECHNOLOGY\nCSE-C\n2ND YEAR")
        st.markdown("""
                    
    ### GROUP MEMBERS:
        Mohsin Raza    2201330100158
        Anurag         2201330100058
        Lakshay Kumar  2201330100142
    """)        

    elif app_mode == "Medicinal Herbs Recognition":
        st.header("HERBAL EYES")
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=128, use_column_width=True, caption="Uploaded Image")
            if st.button("Predict"):
                model = load_model()  # Load the model
                result_index = model_prediction(model, test_image)
                # Assuming your classes are ["Aloevera", "Betel", "Neem"]
                class_names = ["Aloevera", "Betel", "Neem"]
                st.success(f"Model is predicting it's a {class_names[result_index]}.")

if __name__ == "__main__":
    main()
