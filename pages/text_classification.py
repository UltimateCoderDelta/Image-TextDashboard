import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import string
import re
import time

st.header("Text Classification ✏️")
st.markdown("""
**Description:** Use the following classifier to classify text
into specific classes.
""")
# Establish the session states
if "text_classifier" not in st.session_state or not st.session_state.text_classifier:
    st.session_state.text_classifier = {"model": None, "model_name": ''}

if "classification_history" not in st.session_state or not st.session_state.classification_history:
    st.session_state.classification_history = []

if "preprocessed_text" not in st.session_state or not st.session_state.preprocessed_text:
    st.session_state.preprocessed_text = ""

if "prediction" not in st.session_state or not st.session_state.prediction:
    st.session_state.prediction = None

if "params_toggled" not in st.session_state:
    st.session_state.params_toggled = False

if "num_classification" not in st.session_state or not st.session_state.num_classification:
    st.session_state.num_classification = 0

# if "classification_params" not in st.session_state or not st.session_state.classification_params:
#    st.session_state.classification_params = {}

DEFAULT_CLASSIFIER = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TASK 1 - function to load the text models
@st.cache_resource
def load_default_text_classifier():
    """Load default text classifier"""
    try:
        classification_model = pipeline(
            task='text-classification',
            model=DEFAULT_CLASSIFIER,
            device=DEVICE
        )
        st.session_state.text_classifier["model"] = classification_model
        st.session_state.text_classifier["model_name"] = DEFAULT_CLASSIFIER
        st.info("Text classifier loaded successfully!")
        st.balloons()
    except Exception as e:
        st.error(f"Failed to load text classifier: {str(e)}")

    # Function to load custom text classifier


def load_custom_text_classifier(model_name: str):
    """Load a custom text classifier"""
    try:
        custom_classifier = pipeline(
            task='text-classification',
            model=model_name,
            device=DEVICE
        )
        # if the custom model was loaded successfully, let the user know
        st.session_state.text_classifier["model"] = custom_classifier
        st.session_state.text_classifier["model_name"] = model_name
        st.success("Custom text classifier loaded successfully")
        st.balloons()
    except Exception as e:
        st.error(f"Failed to load text classifier: {str(e)}")


# Create a function to calculate metrics
def process_text_metrics(input_text: str):
    """Calculate standard text metrics"""
    total_chars = len(input_text)
    num_tokens = len(input_text.split())
    num_unique_tokens = len(set(input_text.split()))
    return total_chars, num_tokens, num_unique_tokens


# Create function to preprocess input text
def preprocess_text(input_text: str,
                    return_tokens: bool = False,
                    to_lowercase: bool = False) -> [str, None | list[str]]:
    """Preprocess user input text"""
    if len(input_text) == 0:
        st.warning("No text entered. Please enter valid data")
        return

    input_text_tokens = None
    # If the input text is valid, proceed to preprocessing
    input_text = input_text.lower() if to_lowercase else input_text
    # replace all punctuation
    input_text = re.sub(r'[{text}]'.format(text=string.punctuation), ' ', input_text)
    # ensure the text has the same spacing in case with .join
    input_text = ' '.join(input_text.split()).strip()
    if return_tokens:
        input_text_tokens = input_text.split()
        tokens_count = len(input_text_tokens)
        st.info(f"Total number of tokens: {tokens_count}")
        return input_text, input_text_tokens
    # Else if return_tokens is false only return the text
    return input_text, input_text_tokens


# TASK 1 - Adding the dropdown/dialog to select the classification hyperparameters
@st.dialog("Adjust classification parameters")
def adjust_classification_params():
    # Add checkbox if they wish to use lowercase
    use_lower: bool = st.checkbox("Use lowercase", help="Classify a text document in lowercase format")
    return_tokens: bool = st.checkbox("Return tokens", help="Return the document tokens after preprocessing")
    # submit the contents of the dialog
    if st.button("Adjust Parameters", type="primary"):
        st.session_state.classification_params = {"use_lower": use_lower, "return_tokens": return_tokens}
        st.rerun()


# TASK 3 - Adding the tabs for data classification analytics
# TASK 3 - Adding analytics tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Text 📝",
    "Classification History 🏛️",
    "Results & Analytics 📊",
    "Load Model ➕"
])

# TASK 2 - Allow the user to enter their input text for classification
with tab1:
    st.markdown("<h3><b>Single Text Classification</b></h3>",
                unsafe_allow_html=True)
    st.divider()
    user_text = st.text_area("Enter input text for classification", max_chars=2000,
                             help="Enter your text for classification; do not exceed 2000 characters")
    # if the classify text button is pressed, check th
    classify_btn_col, adjust_params_toggle = st.columns([1, 1])
    with classify_btn_col:
        if st.button("Classify Text", type="primary"):
            if len(user_text) == 0:
                st.warning("No input text to classify")
            if not st.session_state.text_classifier["model"]:
                st.warning("No model selected")
            # if the processing parameter button is toggled
            if st.session_state.params_toggled:
                to_lowercase, return_tokens = st.session_state.classification_params.values()
                # if true, call the processing function with the saved adjustments (return_tokens, to_lowercase)
                preprocessed_text, tokens = preprocess_text(user_text, return_tokens, to_lowercase)
                st.session_state.preprocessed_text = preprocessed_text
                # Classify preprocessed text
                try:
                    st.session_state.prediction = st.session_state.text_classifier["model"](preprocessed_text)
                    # st.session_state.prediction = "DUMMY POSITIVE"
                    clear = st.empty()
                    with clear:
                        st.success("Classification successful")
                        time.sleep(2)
                    clear.empty()
                    # Display the result of the prediction
                    st.markdown(
                        f"**Prediction**: {st.session_state.prediction[0]['label']} | Num Tokens: {len(tokens) if tokens else 0}")
                    # Add prediction to prediction history
                    st.session_state.classification_history.append({
                        "Input Text": user_text,
                        "Processed Text": preprocessed_text,
                        "Prediction": st.session_state.prediction[0]['label'].upper(),
                        "Score": round(st.session_state.prediction[0]['score'], 4)
                    })
                    st.session_state.num_classification += 1
                    # reset the toggle state
                    # st.session_state.params_toggled = False
                except Exception as e:
                    st.error(f"Failed to classify input text: {str(e)}")
            else:
                # if it has not been toggled the classification_params don't exist and must be defined
                # Use the default params
                # to_lowercase, return_tokens = st.session_state.classification_params.values()
                preprocessed_text, tokens = preprocess_text(user_text)
                st.session_state.preprocessed_text = preprocessed_text
                try:
                    st.session_state.prediction = st.session_state.text_classifier["model"](preprocessed_text)
                    # st.session_state.prediction = "DUMMY NEGATIVE"
                    clear = st.empty()
                    with clear:
                        st.success("Classification successful")
                        time.sleep(2)
                    clear.empty()
                    # Display the result of the prediction
                    st.markdown(
                        f"**Prediction**: {st.session_state.prediction[0]['label']} | Num Tokens: {len(tokens) if tokens else 0}")
                    # Add prediction to prediction history
                    st.session_state.classification_history.append({
                        "Input Text": user_text,
                        "Processed Text": preprocessed_text,
                        "Prediction": st.session_state.prediction[0]['label'].upper(),
                        "Score": round(st.session_state.prediction[0]['score'], 4)
                    })
                    st.session_state.num_classification += 1
                except Exception as e:
                    st.error(f"Failed to classify input text: {str(e)}")

    with adjust_params_toggle:
        # Update preprocessing params
        # if toggled, call the dialoge
        params_toggle: bool = st.toggle("Adjust preprocessing parameters",
                                        help="Adjust the text preprocessing parameters")
        if "classification_params" not in st.session_state:
            if params_toggle:
                # if toggled, adjust the custom parameters
                adjust_classification_params()
                st.info(f"Status of toggle: {params_toggle}")
        else:
            if not params_toggle:
                st.session_state.pop("classification_params")
        st.session_state.params_toggled = params_toggle
    #  else:
    #     #Allow for a new button which allows for the preprocessing params to be updated
    #     if st.checkbox("Change Parameters", disabled=True if st.session_state.classification_params else False):
    #        del st.session_state["classification_params"]
    #        if st.session_state.params_toggled:
    #           adjust_classification_params()
with tab2:
    st.write("<h3><b>Classification History</b></h3>",
             unsafe_allow_html=True)
    st.divider()
    # Display all the previous classifications if available
    if not st.session_state.classification_history:
        st.info("No texts classified")
    else:
        # if classification_history is not empty display the previous results
        for i, result in enumerate(st.session_state.classification_history):
            # for each classification, display the text, it's prediction
            classification_color = "🟢" if st.session_state.classification_history[i][
                                              'Prediction'].upper() == 'POSITIVE' else "🔴"
            with st.expander(f"**Text ID:** {i + 1}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"""{st.session_state.classification_history[i]['Input Text']}""")
                with col2:
                    # Include next to the prediction a red or green circle; red if the result is NEGATIVE and green if POSITIVE
                    st.write(
                        f"**Prediction:** {st.session_state.classification_history[i]['Prediction']}{classification_color}")
with tab3:
    st.write("<h3><b>Current Text Metrics</b></h3>",
             unsafe_allow_html=True)
    if not st.session_state.classification_history:
        st.info("No classifications for analytics")
    else:
        # add the metrics for the classification texts
        col1, col2, col3, col4 = st.columns(4)
        total_chars, num_tokens, num_unique_tokens = process_text_metrics(user_text)
        with col1:
            st.metric(
                label="Total Characters",
                value=total_chars,
                border=True
            )
        with col2:
            st.metric(
                label="Number of Tokens",
                value=num_tokens,
                border=True
            )
        with col3:
            st.metric(
                label="Unique Tokens",
                value=num_unique_tokens,
                border=True
            )
        with col4:
            st.metric(
                label="All Predictions",
                value=st.session_state.num_classification,
                border=True
            )
        # Add the plot showing the
        with st.expander("View Classification Graph"):
            positives = [pred for pred in st.session_state.classification_history if pred['Prediction'] == 'POSITIVE']
            negatives = [pred for pred in st.session_state.classification_history if pred['Prediction'] == 'NEGATIVE']
            neutrals = [pred for pred in st.session_state.classification_history if pred['Prediction'] == 'NEUTRAL']

            fig = plt.figure()
            plt.barh(['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                     [len(positives), len(negatives), len(neutrals)], label='Classification Results')
            plt.title('Classification Labels v Count')
            plt.xlabel('Prediction')
            plt.ylabel('Count')
            plt.grid(visible=True, which='both')
            plt.yticks(rotation=45)
            plt.legend()
            plt.show()
            st.pyplot(fig)

        with st.expander("Classification With Parameters"):
            st.write("**Description:** here you can view the different model results when parameters are changed")
            st.divider()
            # Allow the user to see the score of their results
            with st.container():
                st.subheader("Alter Input Text")
                st.write("**Current Classified Text**")
                st.write(user_text)
                to_lowercase, return_tokens = st.checkbox("Convert to lowercase"), st.checkbox("View tokens")
                current_text = user_text
                if st.button("Classify", type="primary", key="second_classify_btn"):
                    # Showcase the original text and altered text
                    col1, col2, col3 = st.columns([2, 2, 1])
                    # Showcase the scores
                    with col1:
                        st.write("**Original Text**")
                        st.write(user_text)
                    with col2:
                        st.write("**Preprocessed Text**")
                        if to_lowercase:
                            current_text = preprocess_text(user_text, to_lowercase=to_lowercase)[0]
                            st.write(current_text)
                        else:
                            st.write(current_text)
                    with col3:
                        try:
                            prediction = st.session_state.text_classifier["model"](current_text)
                            score = round(prediction[0]['score'], 4)
                            st.write(f"**Prediction Score:** {score:.2%}")
                        except Exception as e:
                            st.error(f"Failed to classify text: {str(e)}")
        with st.expander("View All Classifications"):
            # Go through all text classifications, and display the top 5 (predictions) in a dataframe
            all_predictions = []
            # move through all prediction values if the history is available
            if not st.session_state.classification_history:
                st.info("No classification history to analyze")
            else:
                for idx, prediction in enumerate(st.session_state.classification_history):
                    # append the unique results to the all_predictions list
                    all_predictions.append({
                        "Text ID": idx + 1,
                        "Sample Text": user_text[:100] if len(user_text) > 100 else user_text,
                        "Prediction": prediction["Prediction"],
                        "Score": round(prediction["Score"], 4),
                        "Score (Percent)": f'{prediction["Score"]:.2%}'
                    })
                dataframe_texts = pd.DataFrame(all_predictions)
                st.dataframe(dataframe_texts)

with tab4:
    st.write("<h3>Load Classification Model</h3>",
             unsafe_allow_html=True)
    # Allow the user to select their text classification model
    st.write("Select a text classification model")
    user_choice = st.selectbox("Available Models", [
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ])
    # Allow the user to load the selected model
    if st.button("Load Model", type='primary'):
        if user_choice:
            with st.spinner("Loading custom model..."):
                # check if a model is already available before loading again, and if the model is the same type
                if st.session_state.text_classifier["model"] and st.session_state.text_classifier[
                    "model_name"] == user_choice:
                    st.warning("A text classification model has already been loaded")
                else:
                    load_custom_text_classifier(user_choice)
        else:
            # else load default model
            load_default_text_classifier()



