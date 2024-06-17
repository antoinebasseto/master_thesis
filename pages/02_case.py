import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from config import Group
from utils import (
    get_ai_prompt,
    get_case_description,
    get_case_index,
    get_group,
    get_hypotheses,
    get_latest_message_content,
    get_run_and_thread_id,
    get_run_status,
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
    st.toast("Hypotheses updated!")


def display_ai_help(group: Group, case_description: str, hypotheses_table: dict):

    if group is Group.CONTROL:
        st.write("You are in the control group. You will not receive any AI help.")
        return

    hypotheses = get_hypotheses(group, hypotheses_table)

    # Save all hypotheses ever entered.
    for ar in hypotheses_table["added_rows"]:
        h = ar["hypothesis"]
        if h not in st.session_state["results"][f"case_{get_case_index()}_hypotheses"]:
            st.session_state["results"][f"case_{get_case_index()}_hypotheses"].append(h)

    # Check and ask for the right number of hypotheses.
    if group is Group.HYPOTHESIS_DRIVEN:
        if len(hypotheses) == 0:
            st.status(
                label="Please select one hypothesis.", expanded=False, state="error"
            )
            return
        if len(hypotheses) > 1:
            st.status(
                label="Please select only one hypothesis.",
                expanded=False,
                state="error",
            )
            return
    if group is Group.RECOMMENDATIONS_DRIVEN:
        if len(hypotheses) == 0:
            st.status(
                label="Please add at least one hypothesis.",
                expanded=False,
                state="error",
            )
            return

    # Add message to a new thread and run it.
    # Because function is cached, this will only run once per prompt and
    # otherwise just return the corresponding ids.
    thread_id, run_id = get_run_and_thread_id(
        get_ai_prompt(group, case_description, hypotheses)
    )

    # Wait for the AI response to be generated.
    status_container = None
    if get_run_status(thread_id, run_id) in ["queued", "in_progress"]:
        status_container = st.status(label="", expanded=False, state="running")
        while get_run_status(thread_id, run_id) in ["queued", "in_progress"]:
            time.sleep(0.1)

    # Parse, save and display the AI's response.
    if get_run_status(thread_id, run_id) == "completed":
        if status_container:
            status_container.update(label="", expanded=True, state="complete")
        else:
            status_container = st.status(label="", expanded=True, state="complete")
        with status_container:
            raw_message = get_latest_message_content(thread_id)
            citations, parsed_message = parse_message(
                raw_message, hypotheses, get_group()
            )
            st.write(parsed_message)
            display_citations(citations)
            st.session_state["parsed_message"] = parsed_message
            st.session_state["citations"] = citations
            st.session_state["results"][
                f"case_{get_case_index()}_ai_help_{thread_id}"
            ] = {
                "considered_hypotheses": hypotheses,
                "raw_message": raw_message,
                "parsed_message": parsed_message,
                "citations": citations,
            }

    # Handle the run's failure to complete
    else:
        st.status(
            label="Failed to generate AI help. Please try again.",
            expanded=False,
            state="error",
        )


def display_citations(citations: list[str]):
    """
    Display the citations.

    :param citations: The list of citations to display.
    """
    citations_string = (
        "**Citations:**\n\n" if len(citations) > 0 else "No citations found."
    )
    for citation in citations:
        citations_string += f"{citations.index(citation) + 1}. {citation}\n\n"
    st.caption(citations_string)


@st.experimental_dialog("Are you done with the case?")  # type: ignore
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
    "Please verbally provide a lead suspect and a rationale as if you were presenting "
    "your conclusions to a colleague with whom you have discussed the case.",
    key="concluded",
)

if st.button("Next", disabled=not st.session_state["concluded"]):
    dialog_case_done()

with st.sidebar:
    st.header("Debug")
    st.write(st.session_state)
