import streamlit as st
st.set_page_config(page_title="NoteCraft AI", page_icon="📝")
import os
import pdf_handler
from llm_worker import worker
from streamlit_cookies_manager import EncryptedCookieManager


cookies = EncryptedCookieManager(
    prefix="AL-Sayed1/NOTECRAFT_AI_WEB",
    password=os.environ.get("COOKIES_PASSWORD", "COOKIES_PASSWORD"),
)


def main():
    st.header("NoteCraft AI")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("gemini-pro", "llama-3.1-70b-versatile"))
        word_range = st.slider(
            "Select the word range",
            value=(200, 300),
            step=50,
            min_value=50,
            max_value=1000,
        )
        word_range = " to ".join(map(str, word_range))
        st.subheader("Your Documents")
        pdf = st.file_uploader(
            "upload your PDF", accept_multiple_files=False, type="pdf"
        )

        process = st.button("Process")
    if pdf: 
        max_pages = pdf_handler.page_count(pdf)
        if max_pages != 1:
            with st.sidebar:
                pages = st.slider("Select the pages to generate notes from: ", value=(1, max_pages), min_value=1, max_value=max_pages)
        else:
            pages = (1, 1)
            with st.sidebar:
                st.write("Only one page in the document")
        if process:
            with st.spinner("Processing"):
                try:
                    llm_worker = worker(selected_llm, cookies)
                    chain = llm_worker.get_chain()
                except KeyError:
                    st.error(
                        f"You do not have access to {selected_llm}, please [get access](/get_access) first and try again."
                    )
                    st.stop()
                raw_text = pdf_handler.get_pdf_text(pdf, page_range=pages)
                output = chain.invoke({"transcript": raw_text, "word_range": word_range})
                if selected_llm == "llama-3.1-70b-versatile":
                    output = output.content
                st.session_state["output"] = output
                st.session_state["pdf_name"] = (
                    os.path.splitext(pdf.name)[0] if pdf else "note"
                )
                st.success("Note Crafted!")

    if "output" in st.session_state:
        st.markdown(st.session_state["output"])
        with st.sidebar:
            st.download_button(
                label="Download Note as .md",
                data=st.session_state["output"],
                file_name=f"{st.session_state['pdf_name']}.md",
                mime="text/markdown",
            )

        usr_suggestion = st.chat_input("Suggest an edit")
        if usr_suggestion:
            editor = worker(selected_llm, cookies, "edit_note")
            editor_chain = editor.get_chain()
            output = editor_chain.invoke(
                {"request": usr_suggestion, "note": st.session_state["output"]}
            )
            st.session_state["output"] = output
            st.rerun()


if __name__ == "__main__":
    main()
