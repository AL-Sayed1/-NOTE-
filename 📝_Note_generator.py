import streamlit as st
from dotenv import load_dotenv, get_key
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os


def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        # Extract text directly from the page
        page_text = page.extract_text() or ""
        
        # Check for images and extract text from them
        images = convert_from_bytes(
            pdf.getvalue(), first_page=page_num, last_page=page_num
        )
        for image in images:
            image_text = pytesseract.image_to_string(image)
            if image_text.strip():  # Only add if there's text
                page_text += f" {image_text}"
        
        text += f"PAGE {page_num}: {page_text}\n"
    
    text += "\n\n\n"
    return text


def get_llm(selected_llm):
    if selected_llm == "llama3-70b-8192":
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.3,
        )
    elif selected_llm == "gemini-pro":
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=get_key(dotenv_path=".env", key_to_get="GOOGLE_API_KEY"),
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create a note from this transcript, include all the main ideas in bullets, with supporting details in sub-bullets. Make sections headers using given page numbers and other important information. Output in markdown formatting. Do it in {word_range} words.",
            ),
            ("user", "{transcript}"),
        ]
    )
    chain = prompt | llm
    return chain


def main():
    load_dotenv()
    st.set_page_config(page_title="NoteCraft AI", page_icon="📝")

    st.header("NoteCraft AI")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("gemini-pro", "llama3-70b-8192"))
        word_range = st.slider(
            "Select the word range",
            value=(200, 300),
            step=50,
            min_value=50,
            max_value=1000,
        )
        word_range = " to ".join(map(str, word_range))
        st.subheader("Your Documents")
        pdf = st.file_uploader("upload your PDF", accept_multiple_files=False, type="pdf")
        process = st.button("Process")

        

    if process and pdf:
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf)
            chain = get_llm(selected_llm)
            output = chain.invoke({"transcript": raw_text, "word_range": word_range})
            if selected_llm == "llama3-70b-8192":
                output = output.content
            st.session_state['output'] = output
            st.session_state['pdf_name'] = os.path.splitext(pdf.name)[0] if pdf else "note"
            st.success("Note Crafted!")

    if 'output' in st.session_state:
        st.markdown(st.session_state['output'])
        with st.sidebar:
            st.download_button(
                label="Download Note as .md",
                data=st.session_state['output'],
                file_name=f"{st.session_state['pdf_name']}.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()
