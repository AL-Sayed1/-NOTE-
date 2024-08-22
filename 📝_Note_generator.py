import streamlit as st
from dotenv import load_dotenv, get_key
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


def get_pdf_text(pdfs):
    text = ""
    pdf_number = 1
    for pdf in pdfs:
        text += f"PDF number {pdf_number}:\n"
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            images = convert_from_bytes(
                pdf.getvalue(), first_page=page_num, last_page=page_num
            )
            for image in images:
                page_text = pytesseract.image_to_string(image)
                text += f"PAGE {page_num}: {page_text}\n"
        text += "\n\n\n"
        pdf_number += 1
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
    st.set_page_config(page_title="<NOTE • V1>", page_icon="📝")

    st.header("<NOTE • V1>")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("gemini-pro", "llama3-70b-8192"))
        word_range = st.slider(
            "Select the word range",
            value=(200, 300),
            step=50,
            min_value=50,
            max_value=2000,
        )
        word_range = " to ".join(map(str, word_range))
        st.subheader("Your Documents")
        pdfs = st.file_uploader("upload your PDFs", accept_multiple_files=True)
        prossess = st.button("Process")
        st.write("To save the note as PDF, Hide the sidebar and press Ctrl + P")
    if prossess:
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdfs)
            chain = get_llm(selected_llm)
            output = chain.invoke({"transcript": raw_text, "word_range": word_range})
            if selected_llm == "llama3-70b-8192":
                output = output.content
            st.markdown(output)

            st.success("NOTE CREATED USING <NOTE • V1>")


if __name__ == "__main__":
    main()
