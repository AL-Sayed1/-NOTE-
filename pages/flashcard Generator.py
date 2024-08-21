import streamlit as st
from dotenv import load_dotenv, get_key
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
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
    # llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":0.3, "max_length":512})
    if selected_llm == "groq":
        llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            temperature=0.3,
        )
    elif selected_llm == "gemini-pro":
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=get_key(dotenv_path=".env", key_to_get="GOOGLE_API_KEY"),
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are tasked with creating flashcards that will help students learn the important terms, proper nouns and concepts in this note. Only make flashcards directly related to the main idea of the note, include as much detail as possible in each flashcard, returning it in a CSV formate like thiswith | as the seperator like this: Term | Definition. make exactly between {flashcard_range} flashcards. only return the csv file without any other information.",
            ),
            ("user", "{transcript}"),
        ]
    )
    chain = prompt | llm
    return chain


def main():
    load_dotenv()
    st.set_page_config(
        page_title="<NOTE • V1> - Flashcards Generator", page_icon=":robot_face:"
    )

    st.header("<NOTE • V1>")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("groq", "gemini-pro"))

        flashcard_range = st.slider(
            "Select how many flashcards do you want",
            value=(5, 20),
            step=1,
            min_value=1,
            max_value=200,
        )
        flashcard_range = " to ".join(map(str, flashcard_range))
        st.write(f"You will get {flashcard_range} flashcards")

        st.subheader("Your Documents")
        pdfs = st.file_uploader("upload your PDFs", accept_multiple_files=True)
        prossess = st.button("Process")
    if prossess:
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdfs)
            chain = get_llm(selected_llm)
            output = chain.invoke(
                {"transcript": raw_text, "flashcard_range": flashcard_range}
            )
            if selected_llm == "groq":
                output = output.content
            # Save output to CSV file
            for pdf in pdfs:
                pdf_name = os.path.splitext(pdf.name)[0]
                csv_file_name = f"{pdf_name}.csv"
                with open(csv_file_name, "w") as f:
                    f.write(output)

                # Provide download button
                st.download_button(
                    label="Download CSV",
                    data=output,
                    file_name=csv_file_name,
                    mime="text/csv",
                )

            st.success(
                'FLASHCARDS GENERATED! You can download the CSV file below and upload it to anki! make sure to choose "Pipe" as the seperator.'
            )


if __name__ == "__main__":
    main()
