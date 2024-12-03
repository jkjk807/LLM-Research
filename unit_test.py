import unittest
from crewai import Agent, Task, Crew, LLM
import json


class TestEmailClassifier(unittest.TestCase):
    def setUp(self):
        """
        Set up the classifier and responder agents before each test.
        """
        self.classifier = Agent(
            role="email classifier",
            goal="accurately classify email based on their importance. give every email strictly one of these ratings: important, casual or spam. No explanation is ever needed for your answer",
            backstory="You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails bad ratings if they are not important. Your job is to help the user manage their inbox.",
            verbose=False,
            allow_delegation=False,
            llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
        )
        self.responder = Agent(
            role="email responder",
            goal="Based on the importance of the email, write a concise and simple response. If the email is rated 'important' write a formal response, if the email is rated 'casual' write a casual response, and if the email is rated 'spam' ignore the email. Be very concise.",
            backstory="You are an AI assistant whose only job is to write short responses to emails based on their importance. The importance will be provided to you by the 'classifier' agent.",
            verbose=False,
            allow_delegation=False,
            llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
        )

    def test_email_classification_and_response(self):
        """
        Test various email scenarios for correct classification and response.
        """
        test_cases = [
            {
                "email": """
                Dear Team,
                The submission deadline for the Q4 Financial Report has been extended to December 18, 2024.
                Please ensure all updates and approvals are completed by this new date. Let me know if there are any issues meeting this deadline.

                Best regards,
                Sarah Cooper
                CFO, Finance Department
                """,
                "expected_classification": "important",
                "expected_response": "Understood. I will review and provide feedback by EOD.",
            },
            {
                "email": """
                Hi John,
                Attached is the latest version of the partnership agreement. Please review the terms highlighted on page 5 and provide your feedback by EOD tomorrow. Let me know if you have any questions.
                Best,
                Jonathan Marks
                Legal Counsel
                """,
                "expected_classification": "important",
                "expected_response": "Understood. I will review and provide feedback by EOD.",
            },
            {
                "email": """
                Hey John,
                Are you free for lunch tomorrow? Let’s catch up! I was thinking about trying that new Thai place near the office. Let me know what works for you.
                Cheers,
                Emily
                """,
                "expected_classification": "casual",
                "expected_response": "Sounds good! Let's plan for 1 PM.",
            },
            {
                "email": """
                Hi John,
                Hope you’re doing well! Any fun plans for the weekend? I was thinking about organizing a hiking trip to Blue Ridge Park if the weather holds up. Would love to have you join! Let me know if you’re interested.
                Best,
                Mike
                """,
                "expected_classification": "casual",
                "expected_response": "Sounds good! Let's plan for 1 PM.",
            },
            {
                "email": """
                Congratulations!
                You've been selected for a chance to receive a brand-new La Vie Est Belle Eau De Perfum! To claim your reward,
                simply answer a few quick questions about your experience with us.

                Limited time offer for our customers!


                Celebrate Life's Beautiful Moments!
                Your La Vie Est Belle Eau De Perfum Awaits

                THE FRAGRANCE OF HAPPINESS. Life is what you make of it so make it beautiful with La vie est belle — the iconic women's fragrance by Lancôme.
                A floral gourmand bouquet encapsulated in a crystal smile. Blending the most noble ingredients,
                La vie est belle is an unforgettable fragrance. Precious Iris accord with vibrant earthy notes of patchouli,
                sensual warm vanilla and spun sugar are combined in this scent exclusively created by two of the most renowned French perfumers —
                Anne Flipo and Dominique Ropion. The art of French perfumery manifested in a unique bottle shape that symbolizes a smile. Each spray will unlock a beautiful day.
                """,
                "expected_classification": "spam",
                "expected_response": "",
            },
            {
                "email": """
                Hi John,
                Congratulations! You’ve been selected to receive a FREE $500 Amazon Gift Card. To claim your reward, simply click the link below and complete the quick survey.
                Click here to claim your gift.
                Don’t miss out—this offer is only valid for the next 24 hours!
                Regards,
                The Rewards Team
                """,
                "expected_classification": "spam",
                "expected_response": "",
            },
        ]

        for index, case in enumerate(test_cases, start=1):
            email = case["email"]
            expected_classification = case["expected_classification"]
            expected_response = case["expected_response"]

            classify_task = Task(
                description=f"Classify the following email: '{email}'",
                agent=self.classifier,
                expected_output="One of these three options: 'important', 'casual', or 'spam'.",
            )

            respond_task = Task(
                description=f"Respond to the email: '{email}' based on the importance provided by the 'classifier' agent",
                agent=self.responder,
                expected_output="A very concise response to the email based on the importance provided by the 'classifier' agent.",
            )

            crew = Crew(
                agents=[self.classifier],
                tasks=[classify_task],
                verbose=False,
            )

            result = crew.kickoff()

            print(f"\nTest {index} of {len(test_cases)}:")
            print(f"Email: {email}")
            print(f"Expected Classification: {expected_classification}")
            print(f"Actual Classification: {result.raw}")

            self.assertEqual(
                result.raw.lower(),
                expected_classification.lower(),
                f"Test {index} failed: Classification mismatch.",
            )


class TestEmailParser(unittest.TestCase):
    def setUp(self):
        """
        Set up the parser agent before each test.
        """
        self.parser = Agent(
            role="email data parser",
            goal=(
                """Extract structured information from the unstructured email text. Specifically, extract the following fields:
        - Sender Name: The name of the person sending the email.
        - House Address: The full house address provided in the email.
        - Years of Work Experience: Total number of years of work experience mentioned.
        - Highest Education Qualification: The highest educational qualification provided.
        - Age: The age of the sender.
        Return the extracted data in the following JSON format:
        {"Sender Name": "<extracted name>", "House Address": "<extracted address>", "Years of Work Experience": <extracted years>, "Highest Education Qualification": "<extracted qualification>", "Age": <extracted age>}
        Ensure all fields are filled, and provide 'N/A' for any missing data. Ensure strict JSON formatting without newlines or unnecessary spaces."""
            ),
            backstory="You are an expert AI email parser, designed to extract and organize specific information accurately.",
            verbose=False,
            allow_delegation=False,
            llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
        )

    def test_email_parsing(self):
        """
        Test the email parser with various unstructured emails.
        """
        test_cases = [
            {
                "email": """
                Dear HR Manager,
                I am submitting my details as requested:
                - Years of Work Experience: 10
                - Address: 456 Oak Street, Denver, CO, 80201
                - Qualification: PhD in Data Science
                - Age: 45
                
                Best, John Doe
                """,
                "expected_output": {
                    "Sender Name": "John Doe",
                    "Years of Work Experience": 10,
                    "House Address": "456 Oak Street, Denver, CO, 80201",
                    "Highest Education Qualification": "PhD in Data Science",
                    "Age": 45,
                },
            },
            {
                "email": """
                Hello Team,
                Here’s my information:
                Work Experience: 5 years
                Address: 789 Pine Lane, Apt 3A, Seattle, WA, 98101
                Education: Master’s in Mechanical Engineering
                Age: 29
                Regards, Jane Smith
                """,
                "expected_output": {
                    "Sender Name": "Jane Smith",
                    "Years of Work Experience": 5,
                    "House Address": "789 Pine Lane, Apt 3A, Seattle, WA, 98101",
                    "Highest Education Qualification": "Master’s in Mechanical Engineering",
                    "Age": 29,
                },
            },
            {
                "email": """
                    Dear HR Team,

                    I have 10 years of experience working in marketing and advertising. My current address is 123 Greenway Blvd, Apt 7C, Portland, OR, 97204. I hold a Bachelor’s degree in Business Administration, and I am 34 years old.

                    Best regards,  
                    Sophia Lee
                """,
                "expected_output": {
                    "Sender Name": "Sophia Lee",
                    "Years of Work Experience": 10,
                    "House Address": "123 Greenway Blvd, Apt 7C, Portland, OR, 97204",
                    "Highest Education Qualification": "Bachelor’s degree in Business Administration",
                    "Age": 34,
                },
            },
            {
                "email": """
                    Hi HR,

                    Thank you for your email, I hope you are doing fine and I am so excited to join the team.
                    I am currently 45 years old. I have been teaching high school Biology for 12 years. I live at 89 Sunset Drive, San Diego, CA, 92109. 
                    My highest qualification is a Master’s degree in Biology.
                    Hope to get your reply soon!!!!

                    Thanks,  
                    Laura Simmons
                """,
                "expected_output": {
                    "Sender Name": "Laura Simmons",
                    "Years of Work Experience": 12,
                    "House Address": "89 Sunset Drive, San Diego, CA, 92109",
                    "Highest Education Qualification": "Master’s degree in Biology",
                    "Age": 45,
                },
            },
        ]

        for index, case in enumerate(test_cases, start=1):
            email = case["email"]
            expected_output = case["expected_output"]

            parse_task = Task(
                description=f"Parse the following email: '{email}'",
                agent=self.parser,
                expected_output=f"JSON: {json.dumps(expected_output)}",
            )

            crew = Crew(
                agents=[self.parser],
                tasks=[parse_task],
                verbose=False,
            )

            result = crew.kickoff()

            actual_output = json.loads(result.raw)
            normalized_expected_output = json.loads(json.dumps(expected_output))

            print(f"\nParser Test {index} of {len(test_cases)}:")
            print(f"Email: {email}")
            print(f"Expected Output: {normalized_expected_output}")
            print(f"Actual Output: {actual_output}")

            self.assertEqual(
                actual_output,
                normalized_expected_output,
                f"Parser Test {index} failed: Parsed data mismatch.",
            )


if __name__ == "__main__":
    unittest.main()
