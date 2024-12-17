import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from config import Group
from utils import (
    get_ai_prompt,
    get_case_description,
    get_case_index,
    get_chat_completion,
    get_group,
    get_hypotheses,
    get_json_schema,
    get_latest_message_content,
    page_setup,
    parse_case_description,
    parse_message,
    save_widget,
)

#######################################
# SETUP
#######################################


page_setup("Case")

if "case_index" not in st.session_state:
    st.session_state["case_index"] = 0

if "citations" not in st.session_state:
    st.session_state["citations"] = None

if f"case_{get_case_index()}_start_time" not in st.session_state["results"]:
    st.session_state["results"][
        f"case_{get_case_index()}_start_time"
    ] = datetime.now().time()

if f"case_{get_case_index()}_hypotheses" not in st.session_state["results"]:
    st.session_state["results"][f"case_{get_case_index()}_hypotheses"] = []


#######################################
# HELPER FUNCTIONS
#######################################


def save_all_hypotheses(hypotheses_table: Dict):
    case_key = f"case_{get_case_index()}"
    for row in hypotheses_table.get("added_rows", []):
        hypothesis = row["hypothesis"]
        if hypothesis not in st.session_state["results"][f"{case_key}_hypotheses"]:
            st.session_state["results"][f"{case_key}_hypotheses"].append(hypothesis)


def validate_hypotheses(group: Group, hypotheses: List[str]) -> bool:
    if group is Group.HYPOTHESIS_DRIVEN:
        if len(hypotheses) == 0:
            st.status(
                label="Please select one hypothesis.", expanded=False, state="error"
            )
            return False
        if len(hypotheses) > 1:
            st.status(
                label="Please select only one hypothesis.",
                expanded=False,
                state="error",
            )
            return False
    elif group is Group.RECOMMENDATIONS_DRIVEN and len(hypotheses) == 0:
        st.status(
            label="Please add at least one hypothesis.", expanded=False, state="error"
        )
        return False
    return True


#######################################
# DISPLAYS
#######################################


def display_case_description():
    st.write(
        parse_case_description(
            get_case_description(get_case_index()), st.session_state["citations"]
        )
    )


def display_hypothesis_input(group: Group, key: str):

    columns = ["hypothesis"]
    column_config = {
        "hypothesis": st.column_config.TextColumn(
            "Your hypotheses:", help="Enter your hypotheses here.", required=True
        )
    }

    if group is Group.HYPOTHESIS_DRIVEN:
        columns.append("selected")
        column_config["selected"] = st.column_config.CheckboxColumn(
            "Select?",
            help="Select the hypotheses you want to investigate further.",
            default=False,
            required=True,
        )

    st.data_editor(
        pd.DataFrame(columns=columns),
        column_config=column_config,
        use_container_width=True,
        num_rows="dynamic",
        on_change=on_hypotheses_change,
        key=key,
    )


def on_hypotheses_change():
    # Get the current hypotheses table from session state
    hypotheses_table = st.session_state["hypotheses_table"]
    
    # Sort the added_rows if they exist
    if "added_rows" in hypotheses_table and hypotheses_table["added_rows"]:
        sorted_rows = sorted(hypotheses_table["added_rows"], key=lambda x: x["hypothesis"].lower())
        hypotheses_table["added_rows"] = sorted_rows

    #st.toast("Hypotheses updated!")
    st.toast("Hypotheses updated and alphabetically sorted!")


def display_ai_help(group: Group, case_description: str, hypotheses_table: dict):

    if group is Group.CONTROL:
        st.write("You are in the control group. You will not receive any AI help.")
        return

    hypotheses, selected_hypotheses = get_hypotheses(hypotheses_table)

    save_all_hypotheses(hypotheses_table)

    if not validate_hypotheses(group, selected_hypotheses):
        return

    chat_completion = get_chat_completion(
        get_ai_prompt(group, case_description, hypotheses),
        get_json_schema(group, hypotheses),
    )

    with st.status(label="", expanded=True, state="complete"):
        raw_message = get_latest_message_content(chat_completion)
        if raw_message is None:
            st.write("No AI message received.")
        else:
            citations, parsed_message = parse_message(
                raw_message, hypotheses, selected_hypotheses, get_group()
            )
            st.write(parsed_message)
            display_citations(citations)
            st.session_state["parsed_message"] = parsed_message
            st.session_state["citations"] = citations
            st.session_state["results"][
                f"case_{get_case_index()}_ai_help_{chat_completion.id}"
            ] = {
                "hypotheses": hypotheses,
                "selected_hypotheses": selected_hypotheses,
                "raw_message": raw_message,
                "parsed_message": parsed_message,
                "citations": citations,
            }


def display_citations(citations: list[str]):
    """
    Display the citations.

    :param citations: The list of citations to display.
    """
    citations_string = (
        "**Citations:**\n\n" if len(citations) > 0 else "No citations found."
    )
    for index, citation in enumerate(citations, start=1):
        citations_string += f"{index}. {citation}\n\n"
    st.caption(citations_string)


@st.experimental_dialog("Are you sure you want to move on to the next case?")  # type: ignore
def dialog_case_done():
    if st.button("Yes", type="primary"):

        st.session_state["results"][
            f"case_{get_case_index()}_end_time"
        ] = datetime.now().time()
        save_widget("hypotheses_table", new_name=f"case_{get_case_index()}_hypotheses")

        st.switch_page("pages/03_case_questionnaire.py")


#######################################
# MAIN
#######################################


st.title(f"Case {get_case_index() + 1}")

# Check if there is an image in the data folder and display it.
if os.path.exists(f"data/case_{get_case_index()}.jpg"):
    text_col, image_col = st.columns([0.8, 0.2])
    with text_col:
        case_description_container = st.container(height=400)
    with image_col:
        st.image(f"data/case_{get_case_index()}.jpg", use_column_width=True)
# If there is no image, just display the case description.
else:
    case_description_container = st.container(height=400)

col1, col2 = st.columns([0.3, 0.7])
with col1:
    hypotheses_df = display_hypothesis_input(get_group(), key="hypotheses_table")
with col2:
    if get_group() is Group.RECOMMENDATIONS_DRIVEN:
        if st.button("See AI Recommendations"): # only show recommendations when the button is pressed
            display_ai_help(
                get_group(),
                get_case_description(get_case_index()),
                st.session_state["hypotheses_table"],
            )
    else:
        display_ai_help(
            get_group(),
            get_case_description(get_case_index()),
            st.session_state["hypotheses_table"],
        )

# Because the case description depends on the AI message - because of
# citations - we need to compute it after the AI message.
with case_description_container:
    display_case_description()

st.divider()

st.checkbox(
    "Please verbally provide a lead diagnosis and a rationale as if you were "
    "presenting your conclusions to a colleague with whom you have discussed the case.",
    key="concluded",
)

if st.button("Next", disabled=not st.session_state["concluded"]):
    dialog_case_done()

with st.sidebar:
    st.header("Debug")
    st.write(st.session_state)
