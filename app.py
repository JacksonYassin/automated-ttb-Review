import streamlit as st
import pandas as pd
import json
import os
import easyocr
from processing import process_label, evaluate_label_results


##############
#
# Web application via streamlit
#
##############


DATA_FILE = "data.json"
LABEL_DIRS = ["test_labels"]
FAILURE_LABELS = [
    "brand name",
    "class",
    "fanciful name",
    "bottler name",
    "bottler address",
    "alcohol content",
    "net content",
    "government warning",
]


# load in expensive easyOCR model weights once
@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(["en"])


# get .json application information
def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


# write to application information
def save_data(data:list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


# determine whether the image file exists
def find_label_image(application_num:str):
    for directory in LABEL_DIRS:
        path = os.path.join(directory, f"{application_num}.png")
        if os.path.exists(path):
            return path
        
    return None


# get information from each application
def build_app_info(record:dict):
    return [
        record.get("brand_name"),
        record.get("class"),
        record.get("fanciful_name"),
        record.get("bottler_name"),
        record.get("bottler_address"),
    ]


def run_processing(record:dict, reader:easyocr.Reader):
    """
    Driver function for processing and validating a label
    """
    app_num = record["application_num"]
    image_path = find_label_image(app_num)

    if image_path is None:
        return {"status": "error", "failures": ["Label image not found"]}

    app_info = build_app_info(record)

    # handles occasional streamlit error where data cannot be loaded in
    try:
        final_output = process_label(image_path, reader, app_info)
        result = evaluate_label_results(final_output, FAILURE_LABELS)
        
        if result == 0:
            return {"status": "passed", "failures": []}
        else:
            _, failed_elements = result
            return {"status": "failed", "failures": failed_elements}
        
    except Exception as e:
        return {"status": "error", "failures": [str(e)]}


def get_status_display(record:dict):
    """
    Used in streamlit table to determine if a label passes or fails
    If it fails, reasons for failure are returned 
    """
    if "processing_result" not in record:
        return ""
    res = record["processing_result"]
    if res["status"] == "passed":
        return "Passed"
    elif res["status"] == "failed":
        reasons = ", ".join(res["failures"])
        return f"Failed: {reasons}"
    elif res["status"] == "error":
        reasons = ", ".join(res["failures"])
        return f"Error: {reasons}"
    
    return ""


def build_main_dataframe(data:list[dict]):
    rows = []
    for record in data:
        rows.append({
            "Application #": record["application_num"],
            "Brand Name": record.get("brand_name", ""),
            "Class": record.get("class", ""),
            "Brewer": record.get("bottler_name", ""),
            "Status": get_status_display(record),
        })

    return pd.DataFrame(rows)


def build_preview_dataframe(data:list[dict], query:str):
    """
    Creates a table to view information on search results for the label preview
    """
    q = query.strip().lower() if query else ""
    filtered = []
    for record in data:
        fanciful = record.get("fanciful_name") or record.get("fancifcul_name") or ""
        searchable = " ".join([
            str(record.get("application_num", "")),
            str(record.get("brand_name", "")),
            str(record.get("class", "")),
            str(fanciful),
            str(record.get("bottler_name", "")),
            str(record.get("bottler_address", "")),
        ]).lower()
        if q == "" or q in searchable:
            filtered.append({
                "Application #": record["application_num"],
                "Brand Name": record.get("brand_name", ""),
                "Class": record.get("class", ""),
                "Brewer": record.get("bottler_name", ""),
                "Address": record.get("bottler_address", ""),
            })

    df = pd.DataFrame(filtered)

    return df.head(5) if len(df) > 5 else df # limit to five rows for clarity


def build_results_csv(data:list[dict]):
    """
    Creates dataframe to later be used to create a csv of processed application information
    """
    rows = []
    for record in data:
        status = ""
        if "processing_result" in record:
            res = record["processing_result"]
            if res["status"] == "passed":
                status = "Passed"
            elif res["status"] == "failed":
                status = "Failed: " + ", ".join(res["failures"])
            elif res["status"] == "error":
                status = "Error: " + ", ".join(res["failures"])

        fanciful = record.get("fanciful_name") or record.get("fancifcul_name") or ""
        rows.append({
            "Application #": record["application_num"],
            "Brand Name": record.get("brand_name", ""),
            "Class": record.get("class", ""),
            "Fanciful Name": fanciful,
            "Brewer": record.get("bottler_name", ""),
            "Address": record.get("bottler_address", ""),
            "Status": status,
        })

    return pd.DataFrame(rows)


def main():
    """
    Driver function for streamlit app
    """
    st.set_page_config(page_title="Automated TTB Compliance Processing", layout="wide")
    st.title("Automated TTB Compliance Processing")

    # load data into session state
    if "data" not in st.session_state:
        st.session_state.data = load_data()
    if "processed_once" not in st.session_state:
        st.session_state.processed_once = False
    if "show_preview" not in st.session_state:
        st.session_state.show_preview = False
    if "preview_indices" not in st.session_state:
        st.session_state.preview_indices = []

    data = st.session_state.data

    # make label preview section
    st.subheader("Preview Labels")
    st.text("Search for and view a label")

    preview_search = st.text_input(
        "Search",
        placeholder="Search by application #, brand, class, brewer, or address",
        label_visibility="collapsed",
    )

    preview_df = build_preview_dataframe(data, preview_search)

    preview_selection = st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="preview_table",
    )

    selected_rows = preview_selection.selection.rows if preview_selection.selection else []

    # place function buttons
    # one to load a label for preview, the other to clear it
    btn_preview_col, btn_clear_col, _ = st.columns([1, 1, 4])

    # load in a label
    with btn_preview_col:
        if st.button("Preview Label"):
            if selected_rows:
                st.session_state.show_preview = True
                st.session_state.preview_indices = selected_rows
            else:
                st.warning("Select at least one row from the table above.")

    # clear loaded in labels
    with btn_clear_col:
        if st.button("Clear previews"):
            st.session_state.show_preview = False
            st.session_state.preview_indices = []
            st.rerun()

    # display label for preview
    if st.session_state.show_preview and st.session_state.preview_indices:
        for row_idx in st.session_state.preview_indices:
            if row_idx < len(preview_df):
                app_num = preview_df.iloc[row_idx]["Application #"]
                image_path = find_label_image(app_num)
                col_img, _ = st.columns(2)
                with col_img:
                    if image_path:
                        st.image(
                            image_path,
                            caption=f"Label for {app_num}",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"No label image found for {app_num}")

    # make visual break between elements
    st.divider()

    # load/process/validate labels
    st.subheader("Process labels")
    st.text("Select labels for processing from the table below using the checkboxes next to each row")

    # buttons to process labels, download results csv, and reset processed data
    col_process, col_download, col_reset, _ = st.columns([1, 1, 1, 4])

    with col_process:
        if st.button("Process Selected", type="primary"):
            main_selection = st.session_state.get("main_table", None)
            main_selected_rows = (
                main_selection.selection.rows
                if main_selection and main_selection.selection
                else []
            )

            if not main_selected_rows:
                st.warning("Select labels from the table below first.")
            else:
                reader = load_easyocr_reader()
                progress_bar = st.progress(0)
                status_text = st.empty()

                for count, row_idx in enumerate(main_selected_rows):
                    record = data[row_idx]
                    app_num = record["application_num"]
                    status_text.text(f"Processing {app_num}...")
                    result = run_processing(record, reader)
                    data[row_idx]["processing_result"] = result
                    progress_bar.progress((count + 1) / len(main_selected_rows))

                st.session_state.data = data
                st.session_state.processed_once = True
                save_data(data)
                status_text.text("Processing complete!")
                st.rerun()

    with col_download:
        if st.session_state.processed_once:
            csv_df = build_results_csv(data)
            csv_data = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv_data,
                file_name="label_results.csv",
                mime="text/csv",
            )
    

    with col_reset:
        if st.button("Reset Results"):
            for record in data:
                if "processing_result" in record:
                    del record["processing_result"]
            save_data(data)
            st.session_state.data = data
            st.session_state.processed_once = False
            st.rerun()


    # create table to veiw and select applications for processing
    st.caption("select the checkbox in the top-left corner to select or deselect all labels")
    main_df = build_main_dataframe(data)

    st.dataframe(
        main_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="main_table",
        column_config={
            "Application #": st.column_config.TextColumn("Application #", width="medium"),
            "Brand Name": st.column_config.TextColumn("Brand Name", width="medium"),
            "Class": st.column_config.TextColumn("Class", width="medium"),
            "Brewer": st.column_config.TextColumn("Brewer", width="large"),
            "Status": st.column_config.TextColumn("Status", width="large"),
        },
    )


if __name__ == "__main__":
    main()