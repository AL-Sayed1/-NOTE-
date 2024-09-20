import streamlit as st
import os
import pandas as pd
import io
import pdf_handler
from llm_worker import worker
from streamlit_cookies_manager import EncryptedCookieManager

st.set_page_config(page_title="NoteCraft AI - Flashcards Generator", page_icon="📝")
cookies = EncryptedCookieManager(
    prefix="AL-Sayed1/NOTECRAFT_AI_WEB",
    password=os.environ.get("COOKIES_PASSWORD", "COOKIES_PASSWORD"),
)


def main():
    st.header("NoteCraft AI - Flashcards Generator")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("gemini-pro", "llama-3.1-70b-versatile"))

        flashcard_type = st.radio(
            "Flashcard Type", ["Term --> Definition", "Question --> Answer"]
        )
        flashcard_range = st.slider(
            "Select how many flashcards do you want",
            value=(5, 20),
            step=5,
            min_value=5,
            max_value=200,
        )
        flashcard_range = " to ".join(map(str, flashcard_range))

        st.subheader("Your Document")
        pdf = st.file_uploader(
            "upload your PDF", accept_multiple_files=False, type="pdf"
        )
        prossess = st.button("Process")

    if pdf: 
        max_pages = pdf_handler.page_count(pdf)
        with st.sidebar:
            pages = st.slider("Select the pages to generate notes from: ", value=(1, max_pages), min_value=1, max_value=max_pages)
        if prossess:
            with st.spinner("Processing"):
                try:
                    llm_worker = worker(selected_llm, cookies, flashcard_type)
                    chain = llm_worker.get_chain()
                except KeyError:
                    st.error(
                        f"You do not have access to {selected_llm}, please [get access](/get_access) first and try again."
                    )
                    st.stop()
                st.session_state.raw_text = pdf_handler.get_pdf_text(pdf, page_range=pages)
                output = chain.invoke(
                    {
                        "transcript": st.session_state.raw_text,
                        "flashcard_range": flashcard_range,
                    }
                )
                if selected_llm == "llama-3.1-70b-versatile":
                    output = output.content
                st.session_state["output"] = output

                st.success(
                    'FLASHCARDS Crafted! You can download the CSV file below and upload it to anki! make sure to choose "Tab" as the Field separator.'
                )
    if "output" in st.session_state:
        output_io = io.StringIO(st.session_state["output"])
        st.write(pd.read_csv(output_io, sep="\t"))
        pdf_name = os.path.splitext(pdf.name)[0]
        with st.sidebar:
            st.download_button(
                label=f"Download flashcards as .csv",
                data=st.session_state["output"],
                file_name=f"{pdf_name}.csv",
                mime="text/csv",
            )
        usr_suggestion = st.chat_input("Suggest an edit")
        if usr_suggestion:
            editor = worker(selected_llm, cookies, "edit_flashcard")
            editor_chain = editor.get_chain()
            output = editor_chain.invoke(
                {"request": usr_suggestion, "flashcards": st.session_state["output"]}
            )
            st.session_state["output"] = output
            st.rerun()


if __name__ == "__main__":
    main()
