import streamlit as st
from dotenv import load_dotenv, get_key
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd
import io

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


def get_llm(selected_llm, flashcard_type):
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

    if flashcard_type == "Term --> Definition":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are tasked with creating flashcards that will help students learn the important terms, proper nouns and concepts in this note. Only make flashcards directly related to the main idea of the note, include as much detail as possible in each flashcard, returning it in a CSV formate with | as the seperator like this: Term | Definition. make exactly from {flashcard_range} flashcards. only return the csv data without any other information.",
                ),
                ("user", "{transcript}"),
            ]
        )
    elif flashcard_type == "Question --> Answer":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are tasked with creating a mock test that will help students learn and understand concepts in this note. Only make questions directly related to the main idea of the note, You should include all these question types: fill in the blank, essay questions and True or False. return the questions and answers in a CSV formate with | as the seperator like this: This is a question | This is the answer. make exactly from {flashcard_range} Questions, Make sure to not generate less or more than the given amount or you will be punished. only return the csv data without any other information.",
                ),
                ("user", "{transcript}"),
            ]
        )
    chain = prompt | llm
    return chain


def main():
    load_dotenv()
    st.set_page_config(
        page_title="NoteCraft AI - Flashcards Generator", page_icon="📝"
    )

    st.header("NoteCraft AI - Flashcards Generator")

    with st.sidebar:
        selected_llm = st.selectbox("Choose LLM", ("gemini-pro", "llama3-70b-8192"))

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
        pdf = st.file_uploader("upload your PDF", accept_multiple_files=False, type="pdf")
        prossess = st.button("Process")
    if prossess and pdf:
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf)
            chain = get_llm(selected_llm, flashcard_type)
            output = chain.invoke(
                {"transcript": raw_text, "flashcard_range": flashcard_range}
            )
            if selected_llm == "llama3-70b-8192":
                output = output.content
            st.session_state["output"] = output

            st.success(
                'FLASHCARDS Crafted! You can download the CSV file below and upload it to anki! make sure to choose "Pipe" as the seperator.'
            )
    if "output" in st.session_state:
        output_io = io.StringIO(st.session_state["output"])
        st.write(pd.read_csv(output_io, sep="|")) 
        pdf_name = os.path.splitext(pdf.name)[0]
        with st.sidebar:
            st.download_button(
                label=f"Download flashcards as .csv",
                data=st.session_state["output"],
                file_name=f"{pdf_name}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
