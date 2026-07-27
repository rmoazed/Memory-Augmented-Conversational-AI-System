import streamlit as st
from baseline_memory_chatbot import BaselineMemoryChatbot
from conflict_memory_chatbot import ConflictMemoryChatbot
from config import client, index
from config import OPENAI_API_KEY, PINECONE_API_KEY
from prompt import system_prompt
import joblib
import numpy as np
import pandas as pd
import ast #convert strings back into real python lists
from evaluation_datasets import evaluation_dataset, retrieval_dataset
import plotly.express as px
from pathlib import Path
import pandas as pd
from evaluation_functions import (
    initialize_bot,
    evaluate_retrieval,
)
#---

BASE_DIR = Path(__file__).resolve().parent #had to do this to regenerate retrieval_results.csv because the way it was originally saved it didn't have the responses saved, which meant
#the sceanrio roleplay app wouldn't run correctly, but doing it in the notebook would have involved re-initializing so many variables and running so many cells and would have been so 
#inefficient. So decided to bypass this and regenrate the results through the app itsef!

RETRIEVAL_RESULTS_PATH = (
    BASE_DIR / "/Users/roshanmoazed/Downloads/CS6120 STREAMLIT/evaluation/retrieval_results.csv"
)
#---

#load datasets
train_df = pd.read_csv("data/memory_dataset_train.csv")
train_embeddings = np.load("data/X_train_embeddings.npy")

#load ML models
svm_model = joblib.load("models/svm_memory_classifier.joblib")
#random_forest_model = joblib.load("models/random_forest.joblib") -> not going to use this model; no need to have multiple ML models running; we just chose the best one, which was
#SVM. In eval section of app we compare XGboost model


#eval data
memory_summary = pd.read_csv('evaluation/memory_summary.csv')
memory_results = pd.read_csv('evaluation/memory_results.csv')
retrieval_summary = pd.read_csv('evaluation/retrieval_summary.csv')
retrieval_results = pd.read_csv('evaluation/retrieval_results.csv')
svm_report = pd.read_csv('evaluation/svm_test_classification_report.csv')
xgb_report = pd.read_csv('evaluation/xgb_test_classification_report-Copy1.csv')
#---

def generate_and_save_retrieval_responses( #was written as part of regenerating the retrieval_results.csv
    existing_results,
    retrieval_dataset,
    output_path
    ):
    """
    Regenerate chatbot responses for every retrieval scenario and
    configuration.

    Existing Boolean result columns are preserved. Actual chatbot response
    columns are added or corrected.

    Results are saved after every completed configuration so progress is
    not lost if generation is interrupted.
    """

    updated_results = existing_results.copy()

    configurations = [
        ("Baseline", "semantic"),
        ("Baseline", "svm"),
        ("Baseline", "llm"),
        ("Conflict", "semantic"),
        ("Conflict", "svm"),
        ("Conflict", "llm")
    ]

    total_evaluations = (
        len(retrieval_dataset) * len(configurations)
    )

    completed_evaluations = 0

    progress_bar = st.progress(0)

    status_placeholder = st.empty()

    #making sure the output folder exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for test in retrieval_dataset:

        scenario_name = test["name"]

        scenario_mask = (
            updated_results["Scenario"] == scenario_name
        )

        if not scenario_mask.any():
            st.warning(
                f"Scenario '{scenario_name}' was not found "
                "in retrieval_results. Skipping it."
            )

            completed_evaluations += len(configurations)

            progress_bar.progress(
                completed_evaluations / total_evaluations
            )

            continue

        for architecture, extraction in configurations:
            display_name = (
                extraction.upper()
                if extraction in {"svm","llm"}
                else extraction.title()
            )
            column_prefix = (
                f"{architecture} {extraction}"
            )

            response_column = (
                f"{column_prefix} Response"
            )

            #add the response column if it does not exist.
            if response_column not in updated_results.columns:
                updated_results[response_column] = None

            updated_results[response_column] = (
            updated_results[response_column]
            .astype(object)
            )

            current_value = updated_results.loc[
                scenario_mask,
                response_column
            ].iloc[0]

            #valid saved response should be nonempty text and 
            #not simply be True or False.
            already_has_valid_response = (
                pd.notna(current_value)
                and str(current_value).strip() != ""
                and str(current_value).strip().lower()
                not in {"true", "false"}
            )

            if already_has_valid_response:

                status_placeholder.info(
                    f"Skipping existing response: "
                    f"{scenario_name} — "
                    f"{architecture} + {extraction}"
                )

                completed_evaluations += 1

                progress_bar.progress(
                    completed_evaluations / total_evaluations
                )

                continue

            status_placeholder.info(
                f"Generating: {scenario_name} — "
                f"{architecture} + {extraction}"
            )

            try:
                # Create a fresh chatbot for this individual test.
                chatbot = initialize_bot(
                    architecture,
                    extraction
                )

                result = evaluate_retrieval(
                    chatbot,
                    test,
                    architecture.lower()
                )

                # Save the actual natural-language response.
                updated_results.loc[
                    scenario_mask,
                    response_column
                ] = result["response"]

                # Update the existing pass/fail column too.
                if column_prefix in updated_results.columns:
                    updated_results.loc[
                        scenario_mask,
                        column_prefix
                    ] = result["passed"]

                # Save immediately after every successful evaluation.
                updated_results.to_csv(
                    output_path,
                    index=False
                )

            except Exception as error:

                # Save all progress completed before the error.
                updated_results.to_csv(
                    output_path,
                    index=False
                )

                st.error(
                    f"Error while generating {scenario_name}, "
                    f"{architecture} + {extraction}: {error}"
                )

            completed_evaluations += 1

            progress_bar.progress(
                completed_evaluations / total_evaluations
            )

    #final save after all evaluations.
    updated_results.to_csv(
        output_path,
        index=False
    )

    progress_bar.progress(1.0)

    status_placeholder.success(
        "Retrieval-response generation complete!"
    )

    return updated_results
#---

page = st.sidebar.radio( #setting up navigation sidebar
    "Navigation",
    [
        "Chatbot Demo", #will be the 3 pages that user can navigate to
        "Evaluation Dashboard",
        "Model Performance"
    ]
)

#---
def parse_list_value(value):
    """
    Convert a CSV string representation of a list back into a Python list.
    Returns an empty list when the value is missing or cannot be parsed.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

        return [str(parsed)]

    except (ValueError, SyntaxError):
        return [str(value)]

#---
#for scenario roleplay (where you can see a sample conversation with the bot)
def render_retrieval_replay(
    retrieval_results,
    retrieval_dataset,
    scenario_name,
    architecture,
    extraction
):
    """
    Display one completed retrieval-evaluation scenario.

    The conversation, question, required keywords, and forbidden keywords
    come from retrieval_dataset.

    The saved chatbot response comes from retrieval_results.
    """

    #find the selected scenario in retrieval_dataset

    scenario = next(
        (
            item
            for item in retrieval_dataset
            if item["name"] == scenario_name
        ),
        None
    )

    if scenario is None:
        st.error(
            f"Scenario '{scenario_name}' was not found in retrieval_dataset."
        )
        return

    #find selected scenario's row in retrieval_results

    matching_rows = retrieval_results[
        retrieval_results["Scenario"] == scenario_name
    ]

    if matching_rows.empty:
        st.error(
            f"No saved retrieval results were found for '{scenario_name}'."
        )
        return

    row = matching_rows.iloc[0]

    #determine whether to use baseline or conflict criteria

    mode = architecture.lower()

    if mode not in ["baseline", "conflict"]:
        st.error(
            "Architecture must be either 'Baseline' or 'Conflict'."
        )
        return

    required_key = f"{mode}_keywords" #just as in functions in notebook
    forbidden_key = f"{mode}_forbidden"

    if required_key not in scenario:
        st.error(
            f"The scenario does not contain '{required_key}'." #if failed, because didn't have required keywords in response
        )
        return

    if forbidden_key not in scenario:
        st.error(
            f"The scenario does not contain '{forbidden_key}'." #good, forbidden key not supposed to be in response!
        )
        return

    required_keywords = scenario[required_key]
    forbidden_keywords = scenario[forbidden_key]

    #find the saved chatbot-response column
    #supports either column naming order:
    #"Baseline Semantic Response"
    #or
    #"Semantic Baseline Response"

    possible_response_columns = [
        f"{architecture} {extraction} Response", #possible formatting of response columns so function works with both
        f"{extraction} {architecture} Response"
    ]

    response_column = next(
        (
            column
            for column in possible_response_columns
            if column in retrieval_results.columns
        ),
        None
    )

    if response_column is None:
        st.error(
            "The response column for this configuration could not be found."
        )

        st.write(
            "Tried these column names:",
            possible_response_columns
        )

        st.write(
            "Available columns:",
            retrieval_results.columns.tolist()
        )

        return

    response = row[response_column]

    #handle missing response values.
    if pd.isna(response):
        response = ""

    response = str(response)
    normalized_response = response.lower()

    # Recalculate pass/fail using the same logic as
    #evaluate_retrieval()

    required_checks = {
        keyword: keyword.lower() in normalized_response
        for keyword in required_keywords
    }

    forbidden_checks = {
        keyword: keyword.lower() in normalized_response
        for keyword in forbidden_keywords
    }

    contains_expected = all(required_checks.values())
    contains_forbidden = any(forbidden_checks.values())

    passed = contains_expected and not contains_forbidden #passes test if has required keywords and no forbidden keywords

#---
    
    #configuration summary; this breaks down everything in the scenario roleplay for users including the conversation between bot and user, whether the response passed the test,
    #why, etc.

    st.subheader("Selected Configuration")

    config_col1, config_col2, config_col3 = st.columns(3) #formatting

    with config_col1:
        st.metric(
            "Scenario",
            scenario_name.replace("_", " ").title()
        )

    with config_col2:
        st.metric(
            "Architecture",
            architecture
        )

    with config_col3:
        st.metric(
            "Extraction Method",
            extraction
        )

    st.divider()

    #original conversation

    st.subheader("Conversation")

    for message in scenario["conversation"]:
        with st.chat_message("user"):
            st.write(message)

    st.divider()

    #retrieval question and chatbot response

    st.subheader("Retrieval Test")

    with st.chat_message("user"):
        st.write(scenario["question"])

    with st.chat_message("assistant"):
        if response.strip():
            st.write(response)
        else:
            st.write("No response was recorded.")

    st.divider()

    #keyword evaluation

    st.subheader("Keyword Evaluation")

    required_col, forbidden_col = st.columns(
        2,
        gap="large"
    )

    with required_col:
        with st.container(border=True):

            st.markdown("#### Required Information")

            st.caption(
                "Every required keyword must appear in the response."
            )

            if not required_keywords:
                st.info(
                    "This scenario has no required keywords."
                )

            for keyword, was_found in required_checks.items():

                if was_found:
                    st.success(
                        f"Found: **{keyword}**"
                    )
                else:
                    st.error(
                        f"Missing: **{keyword}**"
                    )

    with forbidden_col:
        with st.container(border=True):

            st.markdown("#### Forbidden Information")

            st.caption(
                "None of the forbidden keywords should appear."
            )

            if not forbidden_keywords:
                st.info(
                    "This scenario has no forbidden keywords."
                )

            for keyword, was_found in forbidden_checks.items():

                if was_found:
                    st.error(
                        f"Incorrectly included: **{keyword}**"
                    )
                else:
                    st.success(
                        f"Correctly excluded: **{keyword}**"
                    )

    st.divider()

    #overall evaluation result

    if passed:
        st.success( #text for if passed or failed
            """
            ### Overall Result: PASS 

            The response contained all required information and excluded
            all forbidden information.
            """
        )
    else:
        st.error(
            """
            ### Overall Result: FAIL

            The response was missing required information, included forbidden
            information, or both.
            """
        )


#---
def get_memory_scenario(scenario_name): #part of scenario roleplay pipeline
    """
    Find the original conversation dictionary for a selected scenario.
    """
    for scenario in evaluation_dataset:
        if scenario["name"] == scenario_name:
            return scenario

    return None


def display_memory_list(memories, box_type="info"): #for chatbot page/chatting with bot and viewing metadata during chat
    """
    Display each memory in a separate Streamlit message box.
    """
    if not memories:
        st.write("No memories were stored.")
        return

    for memory in memories:
        if box_type == "success":
            st.success(memory)
        elif box_type == "error":
            st.error(memory)
        else:
            st.info(memory)

#---

def display_memory_details(result): #adding so that we can display memory details after every chatbot response
    memory_result = result.get("memory_result")
    retrieved_memories = result.get("retrieved_memories", [])

    with st.expander("View memory processing details"):
        if memory_result is None:
            st.write("No memory-processing result was returned.")
            return

        extraction = memory_result.get("extraction", memory_result)

        col1, col2 = st.columns(2)

        with col1: #show importance score
            st.metric(
                "Importance",
                extraction.get("importance", "N/A")
            )

        with col2: #show if qualified for memory storage
            stored = extraction.get("store", False)
            st.metric(
                "Qualified for memory",
                "Yes" if stored else "No"
            )

        st.write("**Extraction method:**", result.get("extraction_method", "N/A")) 

        relationship = memory_result.get("relationship")

        if relationship is not None: #if memory gets stored, will be relevant for conflict memory bot
            st.divider()
            st.write("### Memory relationship")

            st.write(
                "**Relationship:**",
                relationship.get("relationship", "N/A")
            )
            st.write(
                "**Action:**",
                relationship.get("action", "N/A")
            )

            existing_text = relationship.get("existing_memory_text", "")
            if existing_text:
                st.write("**Existing memory:**")
                st.info(existing_text)

            memory_to_store = relationship.get("memory_to_store", "")
            if memory_to_store:
                st.write("**Memory to store:**")
                st.success(memory_to_store)

        st.divider()
        st.write("### Retrieved memories")

        if retrieved_memories: #showing top 5 most similar memories in train dataset!
            for i, memory in enumerate(retrieved_memories, start=1):
                st.write(f"{i}. {memory}")
        else:
            st.write("No relevant memories were retrieved.") #if no memory stored

        top_k_examples = extraction.get("top_k_examples")

        if top_k_examples is not None:
            st.divider()
            st.write("### Semantic nearest neighbors")
            st.dataframe(top_k_examples)


#---
if page == "Chatbot Demo": #1st page
    st.title("Memory-Augmented Chatbot")
    st.write( #page text
        "Compare different memory extraction and memory management strategies."
    )

    with st.sidebar: #making the sidebar
        st.header("Chatbot Configuration") #sidebar header

        chatbot_type = st.radio(
            "Memory architecture",
            ["Baseline", "Conflict-aware"] #pick if you want to use the baseline model or the conflict memory chatbot atchitecture for 
        #chatbot interaction
        )

        extraction_method = st.selectbox( #choose which type of extraction method you want to use for the chatbot you're using
            "Memory extraction method",
            ["semantic", "svm", "llm"] #will add other ml models later
        )

#---


        architecture_key = ( #creating unique session keys (streamlit reruns whole script when user interacts w/ bot, so this keeps
            #conversation history and objects available across reruns
            "baseline"
            if chatbot_type == "Baseline"
            else "conflict"
        )

        memory_namespace = ( #displaying memory namespace
                f"{extraction_method}_{architecture_key}_app"
            )

        st.divider()

        st.caption("Current Pinecone Namespace")

        st.code(memory_namespace)

    bot_key = f"bot_{architecture_key}_{extraction_method}"
    messages_key = f"messages_{architecture_key}_{extraction_method}"

#---

    #loading bot into session state
    if bot_key not in st.session_state:
        st.session_state[bot_key] = initialize_bot(
            chatbot_type,
            extraction_method
        )

    #initialize display history
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    bot = st.session_state[bot_key]

#---

    #render exisiting messages
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "details" in message
            ):
                display_memory_details(message["details"])

#---

    user_message = st.chat_input("Type a message") #accept one user message

    if user_message:
        st.session_state[messages_key].append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = bot.respond_once(user_message)

            st.markdown(result["response"])
            display_memory_details(result)

        st.session_state[messages_key].append(
            {
                "role": "assistant",
                "content": result["response"],
                "details": result
            
            }
        )

#---
elif page == "Evaluation Dashboard": #2nd page/evaluation dashboard

    st.title("Evaluation Dashboard")

    st.write(
        """
        Compare memory-state accuracy and retrieval performance across
        extraction methods and chatbot architectures.
        """
    )

    st.info( #llm accuracies show up as 0% even though semantic meaning is perfectly preserved, so want to add note about this at top (have one at bottom under graph, too)
    """
    **Evaluation Note**

    LLM-generated memories were evaluated using **exact string matching**. 
    Although many responses were semantically correct, paraphrased memories
    (e.g., *"The user lives in Boston"* vs. *"I live in Boston"*) were
    counted as incorrect. Consequently, the reported LLM memory-state
    accuracy is an underestimate of its true semantic performance, and
    should be regarded as such.
    """
)

    #headline metrics

    best_memory_row = memory_summary.loc[
        memory_summary["Accuracy"].idxmax()
    ]

    best_retrieval_row = retrieval_summary.loc[
        retrieval_summary["Accuracy"].idxmax()
    ]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4) #formatting

    with metric_col1:
        st.metric(
            "Best Memory Accuracy",
            f"{best_memory_row['Accuracy']:.1%}"
        )

    with metric_col2:
        st.metric(
            "Best Retrieval Accuracy",
            f"{best_retrieval_row['Accuracy']:.1%}"
        )

    with metric_col3:
        st.metric(
            "Memory Scenarios",
            memory_results["Scenario"].nunique()
        )

    with metric_col4:
        st.metric(
            "Retrieval Scenarios",
            retrieval_results["Scenario"].nunique()
        )

    st.divider()

    #tabs

    memory_tab, retrieval_tab, replay_tab = st.tabs(
        [
            "Memory-State Evaluation", #the 3 tabs in eval dashboard
            "Retrieval Evaluation",
            "Scenario Replay"
        ]
    )

    #memory-state tab

    with memory_tab:
        st.header("Memory-State Evaluation")

        st.write( #displayed text explaining memory state
            """
            This evaluation measures whether the chatbot's final stored
            memories match the expected memory state after processing a
            conversation.
            """
        )


        fig = px.bar( #bar chart for memory state accuracy
            memory_summary,
            x="Model",
            y="Accuracy",
            title="Memory-State Accuracy",
            color="Model",
            color_discrete_map={ #color map. might change in the future because the colors don't look great together in my opinion
                "Semantic": "#4C78A8",   # blue
                "SVM": "#54A24B",        # green
                "LLM": "#B279A2"         # purple
            },
            text="Accuracy"
            )

        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
            )

        fig.update_layout(
            yaxis_title="Accuracy",
            xaxis_title="Model",
            yaxis_tickformat=".0%",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption( #the caption at the bottom clarifyinf again about the llm 0% accuracy. in future interations of this project would be good to design a separate evaluation method
            #for llms
            """
            LLM memory outputs are evaluated using exact matching.
            Semantically equivalent paraphrases may therefore be marked
            as failures.
            """
        )

    #retrieval tab

    with retrieval_tab:
        st.header("Retrieval Evaluation")

        st.write( #displayed text explaining retrieval
            """
            This evaluation measures whether the chatbot response contains
            the expected information and avoids outdated information.
            """
        )

        fig = px.bar( #bar chart for retrieval
            retrieval_summary,
            x="Model",
            y="Accuracy",
            color="Model",
            color_discrete_map={ #again, will probably change color scheme in future
                "Semantic": "#4C78A8",   # blue
                "SVM": "#54A24B",        # green
                "LLM": "#B279A2"         # purple
            },
            title="Retrieval Evaluation",
            text="Accuracy"
            )

        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
            )

        fig.update_layout(
            yaxis_title="Accuracy",
            xaxis_title="Model",
            yaxis_tickformat=".0%",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
    
    #replay tab

    with replay_tab: #the scenario roleplay tab

        st.header("Interactive Retrieval Demo")

        st.write(
        """
        Select a completed evaluation scenario and chatbot configuration.
        The demo shows the original conversation, retrieval question,
        chatbot response, and keyword-based evaluation.
        """
        )

        st.info(
        """
        **How the replay is evaluated**

        Baseline and conflict-aware architectures have different expected
        and forbidden information. The replay automatically uses the keyword
        lists associated with the selected architecture.
        """
        )

    #selection controls/what user can control during replay

        control_col1, control_col2, control_col3 = st.columns( #formatting
            3,
            gap="medium"
        )

        with control_col1: #can select scenario from a drop down list. There are 5 of them. In the full blown project, Anupreeta has likely designed an eval metric that uses more than
            #five cases, but for the purpose of the demo app this works fine. 
            selected_scenario = st.selectbox(
                "Scenario",
                options=sorted(
                    retrieval_results["Scenario"].unique()
                ),
                format_func=lambda scenario_name: (
                    scenario_name
                    .replace("_", " ")
                    .title()
                ),
                key="retrieval_replay_scenario"
            )

        with control_col2: #can select baseline or conflict-aware bot
            selected_architecture = st.selectbox(
                "Architecture",
                options=[
                    "Baseline",
                    "Conflict"
                ],
                key="retrieval_replay_architecture"
            )

        with control_col3: #can select semantic, svm, or llm
            selected_extraction = st.selectbox(
                "Extraction Method",
                options=[
                    "Semantic",
                    "SVM",
                    "LLM"
                ],
                key="retrieval_replay_extraction"
            )

    #keep replay visible after the button is pressed

        if "show_retrieval_replay" not in st.session_state:
            st.session_state.show_retrieval_replay = False

        if st.button(
            "Show Retrieval Demo",
            type="primary",
            use_container_width=True
        ):
            st.session_state.show_retrieval_replay = True

    #display the replay

        if st.session_state.show_retrieval_replay:

            st.divider()

            render_retrieval_replay(
                retrieval_results,
                retrieval_dataset,
                selected_scenario,
                selected_architecture,
                selected_extraction
            )

#---
    #this is exclusively for regenreating retrival_results.csv with the actual responses to be used in the playback/scenario roleplay. Right now it is a super innocuous collapsed 
    #button at the end of the eval dashboard, and for now I may leave it in there just because this code is so long and rather delicate and I don't want to accidentally delete the
    #wrong thing. And even if a user pressed the button, nothing bad would happen. It would just replace the csv in my directory. But in the future I will take it out. 
    with st.expander(
        "Regenerate Retrieval Responses",
        expanded=False
    ):

        st.warning(
            """
            This process runs every retrieval scenario through all six chatbot
            configurations. It may take a long time.

            Progress is saved after every completed evaluation, so the process
            can be resumed if it is interrupted.
            """
        )

        st.write(
            "The file will be saved to:"
        )

        st.code(
            str(RETRIEVAL_RESULTS_PATH)
        )

        if st.button(
            "Generate and Save Responses",
            type="primary",
            key="generate_retrieval_responses"
        ):

            retrieval_results = (
                generate_and_save_retrieval_responses(
                    existing_results=retrieval_results,
                    retrieval_dataset=retrieval_dataset,
                    output_path=RETRIEVAL_RESULTS_PATH
                )
            )

            st.session_state[
                "updated_retrieval_results"
            ] = retrieval_results

            st.success(
                f"Results saved successfully to:\n\n"
                f"{RETRIEVAL_RESULTS_PATH}"
            )

#---
elif page == "Model Performance": #last page about ML model performance (comparing SVM and XGBoost--the two models that Anupreeta trained)

    st.title("Machine Learning Model Performance")

    st.write( #displayed text)
        """
        Comparison of the supervised models used to classify the memory
        importance of user message input.
        """
    )

    #keep the model-specific content separated and consistently formatted.
    svm_tab, xgb_tab, comparison_tab = st.tabs(
        [
            "SVM",
            "XGBoost",
            "Model Comparison"
        ]
    )

    #SVM tab

    with svm_tab:

        st.header("Support Vector Machine")

        st.write(
            """
            The SVM achieved strong overall performance and made only a small
            number of errors on the test set.
            """
        )

        st.divider()

        #centered confusion matrix. confusion matrix was made in notebook and saved as png. In furture maybe would be nicer to have in rendered by the actual app so that it looks a 
        #little more crisp, but it certainly works for now

        st.subheader("Confusion Matrix")

        image_left, image_center, image_right = st.columns(
            [1, 6, 1]
        )

        with image_center:
            st.image(
                "images/svm_test_confusion_matrix.png",
                caption="SVM test-set confusion matrix",
                use_container_width=True
            )

        st.divider()

        #interpretation cards. explain how to interpret the results of the confusion matrix (test confusion matrix) and how it pertains to model performance

        st.subheader("Performance Summary")

        svm_card_1, svm_card_2, svm_card_3 = st.columns( #formatting for 3 interpretation cards
            3,
            gap="medium"
        )

        with svm_card_1:
            with st.container(border=True):
                st.markdown("#### Overall Result")
                st.metric(
                    label="Test Accuracy",
                    value="97.0%"
                )
                st.write(
                    "The SVM produced the strongest overall test-set "
                    "performance of the two supervised models."
                )

        with svm_card_2:
            with st.container(border=True):
                st.markdown("#### Strongest Categories")
                st.success(
                    "Categories 1 (do not store), 2 (temporary context), and 3 (recurrent context) were classified correctly "
                    "for every test example."
                )

        with svm_card_3:
            with st.container(border=True):
                st.markdown("#### Classification Errors")
                st.info(
                    """
                    Two category 4 (stable preference/goal) examples were classified as category 3.

                    One category 5 (critical constraint) example was classified as category 3.
                    """
                )

        #separate conclusion from the three cards so the cards stay aligned (formatting. at first the formatting was all over the place so had to play around with it a lot)
        with st.container(border=True):
            st.markdown("#### Practical Interpretation")
            st.write(
                """
                The SVM's three errors lowered the predicted importance of
                certain messages. However, categories 3, 4, and 5 all meet
                the threshold for long-term storage, so these particular
                errors would not prevent the affected memories from being
                stored and later accessed during user-chatbot interaction.
                """
            )

        st.divider()

        #classification report--> these results were generated by sklearn classification report

        st.subheader("Classification Report")

        st.caption(
            """
            Precision measures the reliability of each category prediction,
            recall measures how many examples in each category were identified,
            and F1-score balances precision and recall.
            """
        )

        st.dataframe( #making table to be displayed on page
            svm_report,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn(
                    "Category",
                    width="large"
                ),
                "precision": st.column_config.NumberColumn(
                    "Precision",
                    format="%.3f"
                ),
                "recall": st.column_config.NumberColumn(
                    "Recall",
                    format="%.3f"
                ),
                "f1-score": st.column_config.NumberColumn(
                    "F1 Score",
                    format="%.3f"
                ),
                "support": st.column_config.NumberColumn(
                    "Support",
                    format="%.0f"
                )
            }
        )

    #XGBOOST tab

    with xgb_tab:

        st.header("XGBoost")

        st.write(
            """
            XGBoost performed well overall, but made more errors than the SVM,
            including one error with potentially important consequences.
            """
        )

        st.divider()

        #centered confusion matrix (also png)

        st.subheader("Confusion Matrix")

        image_left, image_center, image_right = st.columns(
            [1, 6, 1]
        )

        with image_center:
            st.image(
                "images/xgb_test_confusion_matrix.png",
                caption="XGBoost test-set confusion matrix",
                use_container_width=True
            )

        st.divider()

        #interpretation cards (same idea as before)

        st.subheader("Performance Summary")

        xgb_card_1, xgb_card_2, xgb_card_3 = st.columns(
            3,
            gap="medium"
        )

        with xgb_card_1:
            with st.container(border=True):
                st.markdown("#### Overall Result")
                st.metric(
                    label="Test Accuracy",
                    value="92.1%"
                )
                st.write(
                    "XGBoost achieved lower test accuracy and made more "
                    "classification errors than the SVM."
                )

        with xgb_card_2:
            with st.container(border=True):
                st.markdown("#### Category 5 Errors")
                st.warning(
                    """
                    Category 5 (critical constraints) were mistaken for categories 1 
                    (do not store), 3 (recurrent context), and 4 (stable preference/goal).
                    """
                )

        with xgb_card_3:
            with st.container(border=True):
                st.markdown("#### Primary Risk")
                st.error(
                    """
                    One critical constraint was classified as category 1:
                    information that should not be stored.
                    """
                )

        with st.container(border=True):
            st.markdown("#### Practical Interpretation")
            st.write(
                """
                Classifying a "critical constraint" as "do not store" could prevent
                important information from entering the chatbot's long-term
                memory. For example, a severe allergy or another safety-related
                constraint could be treated as irrelevant and discarded, 
                which is troubling.
                """
            )

        st.divider()

        #classification report for XGBoost

        st.subheader("Classification Report")

        st.caption(
            """
            Precision measures the reliability of each category prediction,
            recall measures how many examples in each category were identified,
            and F1-score balances precision and recall.
            """
        )

        st.dataframe( #making table for XGBoost
            xgb_report,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn(
                    "Category",
                    width="large"
                ),
                "precision": st.column_config.NumberColumn(
                    "Precision",
                    format="%.3f"
                ),
                "recall": st.column_config.NumberColumn(
                    "Recall",
                    format="%.3f"
                ),
                "f1-score": st.column_config.NumberColumn(
                    "F1 Score",
                    format="%.3f"
                ),
                "support": st.column_config.NumberColumn(
                    "Support",
                    format="%.0f"
                )
            }
        )

    #comparison tab (compares SVM and XGBoost directly)

    with comparison_tab:

        st.header("Model Comparison")

        st.write(
            """
            Side-by-side comparison of the two supervised memory-importance
            classifiers.
            """
        )

        st.divider()

        #matching model cards

        svm_comparison_col, xgb_comparison_col = st.columns(
            2,
            gap="large"
        )

        with svm_comparison_col:
            with st.container(border=True):

                st.subheader("SVM")

                metric_col_1, metric_col_2 = st.columns(2)

                with metric_col_1:
                    st.metric(
                        "Test Accuracy",
                        "97.0%"
                    )

                with metric_col_2:
                    st.metric(
                        "Errors",
                        "3"
                    )

                st.success("Selected model")

                st.write(
                    """
                    The SVM achieved higher accuracy, made fewer errors,
                    and did not classify any critical constraint as
                    information that should be discarded.
                    """
                )

        with xgb_comparison_col:
            with st.container(border=True):

                st.subheader("XGBoost")

                metric_col_1, metric_col_2 = st.columns(2)

                with metric_col_1:
                    st.metric(
                        "Test Accuracy",
                        "92.1%"
                    )

                with metric_col_2:
                    st.metric(
                        "Errors",
                        "8"
                    )

                st.warning("Not selected")

                st.write(
                    """
                    XGBoost made more errors and classified one critical
                    constraint as category 1, creating a greater risk of
                    discarding important information.
                    """
                )

        st.divider()

        #comparison table

        comparison_df = pd.DataFrame(
            {
                "Model": ["SVM", "XGBoost"],
                "Test Accuracy": [0.970, 0.921],
                "Incorrect Predictions": [3, 8],
                "Selected": ["Yes", "No"]
            }
        )

        st.subheader("Summary")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn(
                    "Model",
                    width="medium"
                ),
                "Test Accuracy": st.column_config.ProgressColumn(
                    "Test Accuracy",
                    min_value=0,
                    max_value=1,
                    format="%percent"
                ),
                "Incorrect Predictions":
                    st.column_config.NumberColumn(
                        "Incorrect Predictions",
                        format="%d"
                    ),
                "Selected": st.column_config.TextColumn(
                    "Selected",
                    width="small"
                )
            }
        )

        st.success(
            """
            **Final model selection: SVM**

            The SVM was selected because it produced higher test accuracy,
            fewer total errors, and a safer pattern of misclassification for
            the chatbot's memory-storage decision.
            """
        )
        
    
    

    
    
