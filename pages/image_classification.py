import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import io
import math
import torch
from sqlalchemy import text
from transformers import pipeline
from PIL import Image
from datetime import datetime

st.title("Image Classification 📸")
# TASK 1 - Provide the user with a description of this page
st.markdown(
    """This image classification page allows users to classify various images with state-of-the-art models from Hugging Face.""")
st.divider()

# TASK 2 - Ensure the analyzed_images history is in the session state
if "analyzed_images" not in st.session_state or not st.session_state.analyzed_images:
    # add it to the session state
    st.session_state.analyzed_images = []

# TASK 3 - Ensure the image classification model is available
if "classifier" not in st.session_state or not st.session_state.classifier:
    st.session_state.classifier = None

# TASK 4 - Ensure that model_loaded is in the session_state
if "model_loaded" not in st.session_state or not st.session_state.model_loaded:
    st.session_state.model_loaded = False


# Gather latest database id for classification_count
def gather_classification_id():
    try:
        if st.session_state.db_conn:
            result = st.session_state.db_conn.query("""SELECT id FROM image_storage;""")
            # If there is no content in the database return 0
            if not result.empty:
                return int(result.to_numpy()[-1])

            return 0
        st.warning("No database connection available. Unable to store ID")
        return
    except Exception as e:
        st.error(f"Error in accessing table ID: {str(e)}")


# TASK 6 - Create the image table
def create_image_table(conn):
    """Creating the image_storage table"""
    # Create a new table 'image_storage' and insert dummy data into the query
    try:
        conn.session.execute(text("""CREATE TABLE IF NOT EXISTS image_storage (
     id INTEGER PRIMARY KEY,
     label VARCHAR(50) NOT NULL,
     image_name VARCHAR(100) NOT NULL,
     CONSTRAINT label_ck CHECK(LENGTH(label) > 3),
     CONSTRAINT image_name_ck CHECK(LENGTH(image_name) > 3)
     );"""))
        st.success("Successfully created table image_storage")
    except Exception as e:
        st.error(f"Unable to create table: {str(e)}")


# TODO TASK 5 - Initiate the database to save user predictions
@st.cache_resource(ttl="300")
def connect_to_database():
    """Connect to sqlite database and return connection"""
    try:
        with st.spinner("Connecting to database..."):
            db_conn = st.connection("app_database", url="sqlite:///data/app_db", type="sql", ttl="1h23s")
            st.success("Successfully connected to database")
            # Then call create table
            create_image_table(db_conn)
            return db_conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")

    return None


# Establish connection
if "db_conn" not in st.session_state or not st.session_state.db_conn:
    st.session_state.db_conn = connect_to_database()

# Add the image classification ids
if "classification_count" not in st.session_state or not st.session_state.classification_count:
    st.session_state.classification_count = gather_classification_id()


# TASK 7 - INSERT DUMMY DATA
def insert_data_database(conn, data_dict: list[dict]):
    """Insert data into database and session state"""
    try:
        # Insert all data into database
        with st.spinner("Inserting content...", show_time=True):
            saved_data = {}
            with conn.session as session:
                session.execute(text("""
            INSERT OR IGNORE INTO image_storage(id, label, image_name)
              VALUES(:id, :label, :image_name)
            """), params=data_dict)
                st.success("Data inserted into image_storage")
                session.commit()
                session.close()

    except Exception as e:
        st.error(f"Failed to insert data into database: {str(e)}")


# TASK 8 - Select everything from image_storage table
def select_data_from_table(conn, limit_by: int = 3):
    """Select all data from image_storage table"""
    try:
        content = conn.query(f"""SELECT * FROM image_storage""")
        return content
    except Exception as e:
        st.error(f"Failed to select data: {str(e)}")


# TASK 10 - Check if all images are the same mode
def upload_and_process_image(image):
    """Check if all images are the same mode"""
    if image.mode != 'RGB':
        # if not convert it
        image = image.convert('RGB')
        # After converting the image, return it
        return image

    return image


# TASK 11 - Load the custom image classifier
@st.cache_resource()
def load_custom_image_model(model_name):
    """Load custom image classifier model"""
    try:
        with st.spinner("Loading model...", show_time=True):
            model = pipeline(
                task='image-classification',
                model=model_name,
            )
            # Add the model to the session_state if not already available
            if st.session_state.classifier:
                st.warning("A model has already been loaded")
                return
            # if a model has not yet loaded, do so and let the user know
            st.session_state.classifier = model
            st.session_state.model_loaded = True
            st.success("Model loaded successfully ✅")
    except Exception as e:
        st.error(f"Failed to load image classifier: {str(e)}")


def classify_image(image, model, top_k=3):
    """Classify user image with classifier model"""
    # Classify user image
    try:
        predictions = model(image, top_k=top_k)
        # return the label and score
        return predictions
    except Exception as e:
        st.error(f"Failed to classify image: {str(e)}")


# TASK 12 - Add the tabs for the image dashboard section
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Image",
    "Image History",
    "Data & Analytics",
    "Load Model",
    "Database"
])

with tab1:
    st.header("Single Image Classification")
    selected_file = st.file_uploader("Upload Image", type=["jpeg", "jpg", "png"],
                                     help="Upload an image for image classification", key="img_uploader")
    # if image is available then proceed to preprocess it
    if selected_file:
        # if file is available convert to PIL Image object
        image = Image.open(selected_file)
        preprocessed_image = upload_and_process_image(image)
        # Allow the user to classify the image if possible
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(preprocessed_image, caption='Uploaded Image', use_container_width=True)
        with col2:
            # TODO - Update the image classification section
            # Remove classification threshold (for model parameter creation)
            if st.button("Classify Image", type="primary"):
                # Proceed to classify the image with the image model
                if st.session_state.model_loaded:
                    predictions = classify_image(preprocessed_image, st.session_state.classifier,
                                                 st.session_state.top_k)
                    st.success("Image classified successfully ✅")
                    # Since there might be multiple classifications due to the model's top_k result (via the pipeline) use for loop to access them all
                    for idx, pred in enumerate(predictions):
                        st.write(
                            f"""**Image prediction:** {pred['label'] + '🟢' if pred['score'] > st.session_state.confidence_threshold else '🔴'}
              **Score:** {pred['score']:.2%}""")
                    # Save the classification results in history
                    # image_id = np.random.randint(low=1, high=1_000_000)
                    # gather the table ids and convert it into a Series object to gather the most recent id for the next object
                    # st.session_state.classification_count+=1
                    st.session_state.analyzed_images.append({
                        "ID": idx + 1,
                        "File Name": selected_file.name,
                        "Predictions": predictions,
                        "Prediction Time": datetime.now(),
                        "Image": preprocessed_image
                    })
                else:
                    st.warning("⚠️ No image classifier model loaded")
with tab2:
    st.header("Image Classification History 🏛️")
    # Include an expander to view classification history
    if st.session_state.analyzed_images:
        with st.expander("Image History"):
            # Iterate through history of images if available
            # if images are available, display the history
            for idx, images in enumerate(st.session_state.analyzed_images):
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Display the image on the left-hand side
                    st.image(images["Image"], caption=f"Analyzed Image {idx + 1}")
                with col2:
                    # Display the content of the prediction
                    st.write(f"""
              **Image ID:** {idx + 1}\n
              **Prediction:** {images['Predictions'][0]['label']}\n
              **Score:** {images['Predictions'][0]['score']:.2%}
              """)
    else:
        st.info("No images classified")


# Turn image metrics code into a function for organization
def analyzed_image_metrics(analyzed_images):
    col1, col2, col3 = st.columns(3)
    if not analyzed_images:
        st.info("No images classified. Please classify a few to get started")
        return

    all_predictions = []  # ID LABEL IMAGE_NAME
    # Go through all analyzed images and turn the data into a dict
    for pred in analyzed_images:
        all_predictions.append({
            "Image ID": pred['ID'],
            "Prediction": pred['Predictions'][0]['label'],
            "Score": pred['Predictions'][0]['score'],
            "Timestamp": pred['Prediction Time']
        })
    # Convert to pandas dataframe
    images_df = pd.DataFrame(all_predictions)
    with col1:
        st.metric(
            "Total Images",
            value=len(images_df),
            border=True,
            help="The total number of images classified"
        )
    with col2:
        average_score = images_df['Score'].mean()
        st.metric(
            "Mean Confidence Score",
            value=f'{average_score:.2%}',
            border=True,
            help="The mean/average confidence score per image"
        )
    with col3:
        st.metric(
            "Top Class",
            value=images_df['Prediction'].mode().iloc[0] if not images_df.empty else "Empty",
            border=True,
            help="Most frequent label"
        )
    # Add a new section for showcasing visualization
    with st.expander("View Visual Analytics"):
        fig1, fig2 = st.columns(2)
        with fig1:
            fig, axis = plt.subplots(figsize=(8, 6))
            # Plot the number of labels (label frequency with a horizontal bar graph)
            label_counts = images_df['Prediction'].value_counts().head(10)
            axis.barh(label_counts.index, label_counts.values, align='center', label='Frequent Class Predictions')
            axis.set_title('Labels v Frequency')
            # Add images to streamlit
            axis.legend()
            axis.set_xlabel('Frequency')
            axis.set_ylabel('Class Label')

            plt.tight_layout()
            st.pyplot(fig)

        with fig2:
            fig, axis = plt.subplots(figsize=(8, 6))
            axis.set_title('Confidence Scores v Frequency')
            axis.hist(images_df['Score'], bins=20, alpha=0.7)
            axis.grid(visible=True, which='both')
            axis.legend()
            axis.set_xlabel('Confidence Score')
            axis.set_ylabel('Frequency')

            plt.tight_layout()
            st.pyplot(fig)

    if not images_df.empty:
        with st.expander("View Tabular Summary"):
            # Convert dataframe into a streamlit table
            st.table(images_df)
            # Creating the load data to database
            if not images_df.empty and st.session_state.db_conn:
                if st.button("Store Inside Database", type="primary"):
                    # Store the results inside the database
                    storage_content = []
                    for image in st.session_state.analyzed_images:
                        storage_content.append({
                            "id": st.session_state.classification_count + 1,
                            "label": image['Predictions'][0]['label'],
                            "image_name": image['File Name']
                        })
                    # Insert the items into the database in bulk
                    insert_data_database(st.session_state.db_conn, storage_content)


with tab3:
    st.header("Classification Metrics 📊")
    analyzed_image_metrics(st.session_state.analyzed_images)
    # Proceed with
with tab4:
    st.header("Load Image Model")
    selected_model = st.selectbox("Select model", [
        "google/vit-base-patch16-224",
        "microsoft/resnet-50"
    ], help="Select an image classification model")
    # Allow the user to select a classification threshold
    st.subheader("Classification Options")
    top_k = st.slider("Choose top k results", min_value=1, max_value=10, value=5, step=1,
                      help="Select the top k likely labels for the model to return")
    confidence_threshold = st.slider("Select confidence threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.1,
                                     help="Select a classification threshold for the model")
    # Add the model confidence threshold to the session state in case the order of the file changes
    if "confidence_threshold" not in st.session_state or not st.session_state.confidence_threshold:
        st.session_state.confidence_threshold = confidence_threshold
    else:
        st.session_state.confidence_threshold = confidence_threshold
    if "top_k" not in st.session_state or not st.session_state.top_k:
        st.session_state.top_k = top_k
    else:
        st.session_state.top_k = top_k
    # Allow the user to load
    if st.button("Load Model", type="primary"):
        # Ensure a model has been selected
        load_custom_image_model(selected_model)


def download_table_content(conn):
    try:
        result = conn.query("""SELECT * FROM image_storage""")
        if result.empty:
            st.info("No content available")
            return
        # if there is data available save it as an xlsx file
        string_buffer = io.StringIO()
        result.to_csv(string_buffer, index=False)
        st.download_button(
            label="Download Content",
            data=string_buffer.getvalue(),
            file_name=f"image_classification_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")

    except Exception as e:
        st.error(f"Error loading database content: {str(e)}")


@st.dialog("Delete Table?")
def table_deletion_dialog(conn):
    st.write("Are you sure you wish to delete table contents?")
    choice1, choice2 = st.columns(2)
    try:
        with choice1:
            if st.button("Yes", type="secondary"):
                # Then delete the table content
                conn.session.execute(text("""DELETE FROM image_storage;"""))
                conn.session.commit()
                # st.rerun()
                return True
        with choice2:
            if st.button("No", type="secondary"):
                return False
    except Exception as e:
        st.error(f"Failed to delete table: {str(e)}")


def purge_table(conn):
    """Delete table content if the user desires"""
    if st.button("Delete Content", type="secondary"):
        # Add a dialog ensuring the user wishes to delete the content
        # result = table_deletion_dialog(conn)
        result = conn.session.execute(text("""DELETE FROM image_storage;"""))
        conn.session.commit()
        st.info(f"Number of rows deleted: {result.rowcount}")
        # if "delete_dialog" not in st.session_state or not st.session_state.delete_dialog:
        #   result = table_deletion_dialog(conn)
        # if result: st.success("Table image_storage content deleted")
        st.rerun()


def select_all_table_content(conn):
    """Store content in session state (for immediate updates) and database for cached maintainance"""
    try:
        # Show table preview content
        return conn.query("""SELECT * FROM image_storage LIMIT 10;""")
    except Exception as e:
        st.error(f"Failed to retrieve content: {str(e)}")


# Function to perform database operations
def database_operations():
    # Allow the user to load the database if the db_conn exists
    if st.session_state.db_conn:
        st.write("**Database Content Preview**")
        result = select_all_table_content(st.session_state.db_conn)
        if not result.empty:
            st.dataframe(result)
        else:
            st.info("No content to preview")
        # Add a rerun button to reload the database table query
        col1, col2 = st.columns([1, 1])
        with col1:
            download_table_content(st.session_state.db_conn)
        with col2:
            purge_table(st.session_state.db_conn)
    # Else display no content
    else:
        st.info("No content within image_storage table")


with tab5:
    database_operations()


