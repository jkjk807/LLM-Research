from crewai import Agent, Task, Crew, LLM

email = """
Dear [HR Manager's Name],

I am writing to submit the requested details regarding my work experience and personal information. Please find them below:

Years of Work Experience: 7 years
House Address: 123 Elm Street, Apartment 4B, Springfield, IL 62704
Highest Education Qualification: Master’s Degree in Business Administration (MBA)
Age: 32
Please let me know if you require any further details.

Best regards,
John
[Employee ID, if applicable]
"""

parser = Agent(
    role="email data parser",
    goal=(
        """Extract structured information from the unstructured email text. Specifically, extract the following fields:
        - Sender Name: The name of the person sending the email.
        - House Address: The full house address provided in the email.
        - Years of Work Experience: Total number of years of work experience mentioned.
        - Highest Education Qualification: The highest educational qualification provided.
        - Age: The age of the sender.
        Return the extracted data in the following JSON format:
        {
          "Sender Name": "<extracted name>",
          "House Address": "<extracted address>",
          "Years of Work Experience": <extracted years>,
          "Highest Education Qualification": "<extracted qualification>",
          "Age": <extracted age>
        }
        Ensure all fields are filled, and provide 'N/A' for any missing data."""
    ),
    backstory="You are an expert AI email parser, designed to extract and organize specific information accurately.",
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
)

parse_email = Task(
    description=f"Parse the following email: '{email}' ",
    agent=parser,
    expected_output="""
    JSON format:
    {
        "Sender Name": "<Full Name>",
        "House Address": "123 Elm Street, Apartment 4B, Springfield, IL 62704",
        "Years of Work Experience": 7,
        "Highest Education Qualification": "Master’s Degree in Business Administration (MBA)",
        "Age": 32
    }
    """,
)

crew = Crew(
    agents=[parser],
    tasks=[parse_email],
    verbose=True,
)

# Execute the crew
result = crew.kickoff()

print(result)
