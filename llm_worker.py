from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


class worker:
    def __init__(self, selected_llm, cookies, generate=None):
        self.cookies = cookies
        self.selected_llm = selected_llm
        self.llm = self._initialize_llm()
        self.generate = generate

    def _initialize_llm(self):
        if self.selected_llm == "llama-3.1-70b-versatile":
            llm = ChatGroq(
                model_name="llama-3.1-70b-versatile",
                temperature=0.3,
                api_key=self.cookies["GROQ_API_KEY"],
            )
        elif self.selected_llm == "gemini-pro":
            llm = GoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=self.cookies["GOOGLE_API_KEY"],
            )
        return llm

    def _create_prompt(self):
        if not self.generate:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a student writing notes from this transcript, Make sections headers, include all the main ideas in bullets and sub-bullets or in tables. Do not include unimportant information such as page numbers, teacher name, etc... Add information that is not in the provided transcript that will help the student better understand the subject. Try to make it clear and easy to understand as possible. Output in markdown formatting. Do it in {word_range} words.",
                    ),
                    ("user", "{transcript}"),
                ]
            )
        elif self.generate == "edit_note":
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """ You are tasked to make an edit to this note:
                        {note}.

                        output in markdown formatting.""",
                    ),
                    ("user", "{request}"),
                ]
            )
        elif self.generate == "edit_flashcard":
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """ You are tasked to make an edit to these flashcards:
                        {flashcards}.
                        
                        output in the same csv formate with '\t' as the seperator like this: This is a question \t This is the answer. only return the csv data without any other information.""",
                    ),
                    ("user", "{request}"),
                ]
            )
        elif self.generate == "Term --> Definition":
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are tasked with creating flashcards that will help students learn the important terms, proper nouns and concepts in this note. Only make flashcards directly related to the main idea of the note, include as much detail as possible in each flashcard, returning it in a CSV formate with '\t' as the seperator flashcards should be like this example: Term \t Definition. make exactly from {flashcard_range} flashcards. only return the csv data without any other information.",
                    ),
                    ("user", "{transcript}"),
                ]
            )
        elif self.generate == "Question --> Answer":
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are tasked with creating a mock test that will help students learn and understand concepts in this note. Only make questions directly related to the main idea of the note, You should include all these question types: fill in the blank, essay questions, short answer questions and True or False. return the questions and answers in a CSV formate with '\t' as the seperator flashcards should be like this example: This is a question \t This is the answer. make exactly from {flashcard_range} Questions, Make sure to not generate less or more than the given amount or you will be punished. only return the csv data without any other information.",
                    ),
                    ("user", "{transcript}"),
                ]
            )
        return prompt

    def get_chain(self):
        prompt = self._create_prompt()
        chain = prompt | self.llm
        return chain
