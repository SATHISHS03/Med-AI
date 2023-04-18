import json
import time
import base64
import requests
import streamlit as st
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
# Define CSS styles
PAGE_TRANSITIONS = """
<style>
.page-enter {
    opacity: 0;
    transform: translateY(50px);
    transition: all 0.5s ease;
}
.page-enter-active {
    opacity: 1;
    transform: translateY(0);
    transition: all 0.5s ease;
}
.page-exit {
    opacity: 1;
    transform: translateY(0);
    transition: all 0.5s ease;
}
.page-exit-active {
    opacity: 0;
    transform: translateY(-50px);
    transition: all 0.5s ease;
}
sidebar .sidebar-content {
        background-color: #F5F5F5;
        font-family: 'Open Sans', sans-serif;
        padding: 25px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content .sidebar-section {
        margin-bottom: 25px;
    }
    .sidebar .sidebar-content .sidebar-section h2 {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li {
        list-style: none;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li a {
        color: #555555;
        text-decoration: none;
        display: flex;
        align-items: center;
    }
    .sidebar .sidebar-content .sidebar-section ul li a svg {
        margin-right: 10px;
    }
    .sidebar .sidebar-content .sidebar-section ul li a:hover {
        color: #0072C6;
    }
</style>
"""

# Display CSS styles
st.markdown(PAGE_TRANSITIONS, unsafe_allow_html=True)

# Define page content
def home_page(image_file):
    st.image("logo.png")
    st.image("sub.png")
    # st.subheader('welcome to the Med AI web! please select a domain form the sidebar')

    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
     # Set the background image of the body element
        # Set the background image of the body element
    st.markdown(
        f"""
        <style>
        .stApp  {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-size: 100% 90%;
        }}
        </style>
        """,
        
        unsafe_allow_html=True
    )
    
    
def chest_page(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
     # Set the background image of the body element
        # Set the background image of the body element
    st.markdown(
        f"""
        <style>
        .stApp  {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-size: 100% 90%;
        }}
        </style>
        """,
        
        unsafe_allow_html=True
    )
    st.title('Pulmonology disease detection(X-RAY)')
    st.write("For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.")
    st.write("The normal chest X-ray  depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia  typically exhibits a focal lobar consolidation, in this case in the right upper lobe , whereas viral pneumonia  manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.")
    uploaded_file = st.file_uploader('Upload a chest X-ray image [ jpg , jpeg , png ]🔽', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.write('Chest X-ray image uploaded.')
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="chest_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Load image
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 224, 224, 3)
        img_array = img_array.astype('float32') / 255.0

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Invoke interpreter
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        # comparing the label with the output tensor

        labels =["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
        st.image(uploaded_file)
        # Find the predicted label
        predicted_label = labels[output.argmax()]
        if predicted_label == "COVID19":
            st.error("You have been affected by COIVD-19")
            
            st.warning("""Treatment plan:Mild cases: managed at home with rest, hydration, and over-the-counter medications for fever and pain.Severe cases: require hospitalization and may include oxygen therapy, corticosteroids, anticoagulants, and other medications.
            Preventive measures:
            Get vaccinated
            Wear masks
            Practice physical distancing
            Wash hands frequently
            Avoid large gatherings
            """)
        elif predicted_label == "NORMAL":
            st.success("You dont have any problem in breathing")
            st.balloons()
        elif predicted_label == "PNEUMONIA":
            st.error("you have been diagosed by PNEUMONIA ")
            st.warning("""Treatment plan - Bacterial pneumonia: treated with antibiotics Viral pneumonia: managed with antiviral medications Oxygen therapy may also be necessary for severe cases.
            Preventive measures:
            Get vaccinated
            Practice good hygiene
            Avoid smoking and exposure to secondhand smoke
            Manage chronic health conditions.""")

        elif predicted_label == "TURBERCULOSIS":
            st.error("You have been diagonsed by TURBERCULOSIS")
            st.warning("""Treatment plan -Combination of antibiotics taken for several months.Treatment can be more complex if the infection is drug-resistant.
            Preventive measures:
            Get vaccinated
            Practice good hygiene
            Avoid close contact with people who have active TB
            Test and treat latent TB infection to prevent progression to active disease """)
def brain_page():
    
    # Set page title
    st.title("Brain Tumor Classification (MRI)")

    # Add some text
    st.write("Brain tumors are complex and varied in size and location, making them difficult to fully understand. Professional neurosurgeons are required for MRI analysis, but in developing countries, the lack of skilled doctors and knowledge about tumors can make it challenging to generate reports. An automated cloud system could solve this problem. Brain tumors can be classified as benign, malignant, pituitary, and others. Accurate diagnostics and proper treatment planning are crucial to improve patient life expectancy, and MRI is the best technique for detection.   ")

    # Add file upload widget
    uploaded_file = st.file_uploader("Upload a MRI image[ jpg , jpeg , png ]🔽  ", type=["jpg", "jpeg", "png"])

    # Check if file is uploaded
    if uploaded_file is not None:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="brain_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Load the input image and preprocess it
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 224, 224, 3)
        img_array = img_array.astype('float32') / 255.0


        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Invoke interpreter
        interpreter.invoke()

                # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        print(output)
        output = output[0]

        # Find the predicted label
        labels =["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        predicted_label = labels[output.argmax()]
        st.write("Predicted label:", predicted_label)
        st.image(uploaded_file)

        # Display the prediction
        if predicted_label =='no_tumor' :
            st.success("This brain does not have a tumor.")
            st.balloons()
        elif predicted_label == "pituitary_tumor":
            st.error("This brain has a tumor.")
            st.write("A pituitary tumor is a non-cancerous growth that develops in the pituitary gland, a small gland located at the base of the brain that plays a crucial role in regulating various hormones in the body. These tumors can cause an overproduction or underproduction of hormones, depending on their size and location.")
            st.write("""general treatment for pituitary treatment includes:
            Observation
            Medication
            Surgery
            Radiation therapy""")
        elif predicted_label == "meningioma_tumor":
            st.error("This brain has a tumor.")
            st.write("A meningioma is a type of tumor that grows from the meninges, which are the layers of tissue that cover the brain and spinal cord. Meningiomas are usually slow-growing and often benign, meaning they do not spread to other parts of the body. However, they can still cause problems by pressing on the brain or spinal cord. ")
            st.write("""The general treatment options for meningioma include :
            Observation
            Surgery
            Radiation therapy
            Chemotherapy
            Hormonal therapy
            """)
        elif predicted_label == "glioma_tumor":
            st.error("This brain has a tumor.")
            st.write("A glioma is a type of tumor that originates from the glial cells in the brain or spinal cord. Glial cells are non-neuronal cells that provide support and nourishment to the neurons in the brain.")
            st.write("""Treatment for gliomas depends on the type and grade of the tumor and may include: 
            Surgery
            radiation therapy
            chemotherapy or a combination of these treatments.""")

                     

def eye_page():
    st.title('Ocular diseases recongition')
    st.write('Ocular diseases refer to a group of medical conditions that affect the eyes and vision. There are various types of ocular diseases, and each has its own set of symptoms and causes. Various diseases include bulging eyes, cataracts, crossed eyes, glaucoma and uveitis. Detection of these diseases is crucial to their treatment. An automated system helps in this detection.')
    uploaded_file = st.file_uploader('Upload an eye fundus image [ jpg , jpeg , png ]🔽', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="brain_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load image
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 224, 224, 3)
        img_array = img_array.astype('float32') / 255.0
        print(img_array.shape)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Invoke interpreter
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        # comparing the label with the output tensor

        labels =["Cataract", "Diabetic retinopathy", "Glaucoma", "Normal"]
        print(output)
        st.image(uploaded_file)

        # Find the predicted label
        predicted_label = labels[output.argmax()]
        if predicted_label == "Normal":
            st.success("Congulations your eye sight is normal")
            st.balloons()
        elif predicted_label == "Cataract":
            st.error("You have been diagonesd with oculur cataract")
            st.warning("In the early stages, cataracts may be managed through changes in eyeglass prescriptions or brighter lighting.However, if vision loss becomes more significant, cataract surgery may be necessary. During this procedure, the cloudy lens is removed and replaced with a clear artificial lens")
            st.warning("Treatment: Surgery is the most common treatment for cataracts. During the procedure, the cloudy lens is removed and replaced with a clear, artificial lens.Prevention: Protect your eyes from the sun by wearing sunglasses and a hat, maintain a healthy diet, and avoid smoking.")
        elif predicted_label == "Diabetic retinopathy":
            st.error("You have been diagonesd with oculur Diabetic retinopathy")
            st.warning("Treatment for diabetic retinopathy may include controlling blood sugar levels through diet, exercise, and medication.Other treatments may include laser surgery to seal leaking blood vessels or reduce swelling in the retina. In more advanced cases, surgery may be necessary to remove blood or scar tissue from the eye.")
            st.warning("Treatment: Depending on the stage and severity of the disease, treatment may include laser surgery, injections of medication into the eye, or vitrectomy surgery.Prevention: Control your blood sugar levels, blood pressure, and cholesterol levels, maintain a healthy diet, exercise regularly, and have regular eye exams.")
        elif predicted_label == "Glaucoma":
            st.error("You have been diagonesd with oculur Glaucoma")
            st.warning("Treatment for glaucoma may include eye drops, oral medication, or surgery. Eye drops and oral medication may be used to reduce eye pressure and prevent further damage to the optic nerve.In some cases, surgery may be necessary to improve the drainage of fluid in the eye and reduce eye pressure")
            st.warning("Treatment: Treatment may include eye drops, oral medication, laser therapy, or surgery.Prevention: Have regular eye exams, maintain a healthy diet and exercise regularly, protect your eyes from the sun, and avoid smoking.")
def skin_page():
    st.title('Skin diseases recongition')
    st.write('Dermatology-related diseases encompass a wide range of conditions that affect the skin, hair, and nails. Some examples include actinic keratosis, carcinoma, melanoma, nevus etc.AI can play a significant role in assisting dermatologists in several ways. For instance, AI-powered tools can aid in the early detection and diagnosis of skin cancer by analyzing images of moles and lesions for signs of malignancy. AI algorithms can also help identify patterns and predict disease outcomes based on patient data, leading to more personalized treatment plans.')
    uploaded_file = st.file_uploader('Upload an skin fundus image [ jpg , jpeg , png ]🔽', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="brain_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load image
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 224, 224, 3)
        img_array = img_array.astype('float32') / 255.0
        print(img_array.shape)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Invoke interpreter
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        # comparing the label with the output tensor

        labels =["actinic keratosis", "basal cell carcinoma", "dermatofibroma", "melanoma","nevus","pigmented benign keratosis","seborrheic keratosis","squamous cell carcinoma","vascular lesion"]

        # Find the predicted label
        predicted_label = labels[output.argmax()]
        st.image(uploaded_file)

        if predicted_label == "actinic keratosis":
            st.error("You have been diagonesd with actinic keratosis")
            st.warning("Treatment: Treatment may include cryotherapy, topical creams or gels, photodynamic therapy, or curettage and electrodesiccation.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "basal cell carcinoma":
            st.error("You have been diagonesd with basal cell carcinoma")
            st.warning("Treatment: Treatment may include excisional surgery, Mohs micrographic surgery, cryosurgery, or radiation therapy.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "dermatofibroma":
            st.error("You have been diagonesd with dermatofibroma")
            st.warning("Treatment: Treatment may include surgical removal of the growth or cryotherapy.")
            st.warning("Prevention: There are no specific preventive measures for dermatofibroma, but protecting your skin from the sun may help reduce the risk of developing skin growths.")

        elif predicted_label == "melanoma":
            st.error("You have been diagonesd with melanoma")
            st.warning("Treatment: Treatment may include surgical removal of the cancerous growth, chemotherapy, radiation therapy, or immunotherapy.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "nevus":
            st.error("You have been diagonesd with nevus")
            st.warning("Treatment: Treatment may include surgical removal of the growth or cryotherapy.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "pigmented benign keratosis":
            st.error("You have been diagonesd with pigmented benign keratosis")
            st.warning("Treatment: Treatment may include cryotherapy, electrosurgery, or laser therapy.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "seborrheic keratosis":
            st.error("You have been diagonesd with seborrheic keratosis")
            st.warning("Treatment: Treatment may include cryotherapy, electrosurgery, or laser therapy.")
            st.warning("Prevention: There are no specific preventive measures for seborrheic keratosis, but protecting your skin from the sun may help reduce the risk of developing skin growths.")

        elif predicted_label == "Squamous cell carcinoma":
            st.error("You have been diagonesd with Squamous cell carcinoma")
            st.warning("Treatment: Treatment may include surgical removal of the cancerous growth, chemotherapy, radiation therapy, or immunotherapy.")
            st.warning("Prevention: Protect your skin from the sun by wearing protective clothing and sunscreen, avoid tanning beds, and have regular skin exams.")

        elif predicted_label == "vascular lesion":
            st.error("You have been diagonesd with vascular lesion")
            st.warning("Treatment: Treatment may include laser therapy or surgical removal.")
            st.warning("Prevention: There are no specific preventive measures for vascular lesions, but protecting your skin from the sun may help reduce the risk of developing skin growths.")
# Define page transitions
def page_transition(next_page):
    st.markdown(f'<div class="page-exit page-exit-active">{st.session_state.current_page}</div>', unsafe_allow_html=True)
    st.session_state.current_page = next_page
    st.markdown(f'<div class="page-enter page-enter-active">{next_page}</div>', unsafe_allow_html=True)

# Initialize app
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
def sidebar():
    
    # st.sidebar.markdown('Precise medical diagnosis from expert machine learning algorithms and medical imaging data.')

    st.sidebar.markdown("your health is our priority - let's take the first step together")
    selection = st.sidebar.radio('🔽Domain selection', ['🔘Home','1️⃣ Pulmonology','2️⃣ Neurology ', '3️⃣ Ophthalmology','4️⃣ Dermatology'])
    return selection
# Define layout
st.sidebar.image('Screenshot 2023-04-18 054027.png')
st.sidebar.title('MED AI')
page = sidebar()
# menu_items = {'Home': 'home', 'Chest': 'chest', 'Brain': 'brain', 'Eye': 'eye'}
# menu_selection = st.sidebar.radio('Select a page', list(menu_items.keys()))
# Display the selected page
if page == '1️⃣ Pulmonology':
    chest_page("bffff.png")
elif page == '🔘Home':    
    home_page('doctor_computer.jpg')
elif page == '2️⃣ Neurology ':
    brain_page()
elif page == '3️⃣ Ophthalmology':
    eye_page()
elif page == '4️⃣ Dermatology':
    skin_page()



