from crewai import Agent, Task, Crew, LLM

email = """
nigrian price sending som gold

"""

classifier = Agent(
    role="email classifier",
    goal="accurately classify email based on their importance. give every email strictly one of these ratings: important, casual or spam. No further explanation needed",
    backstory="You are an AI assistant whose only job is to classify emails accurately and honestly. Do not afraid to give emails bad rating if they are not important. Your job is to help the user manage their inbox.",
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
)

responder = Agent(
    role="email responder",
    goal="Based on the importance of the email, write a concise and simple response. If the email is rated 'important' write a formal response, if the email is rated 'casual' write a casual response, and if the email is rated 'spam' ignore the email. no matter what, be very concise",
    backstory="You are an AI assistant whose only job is to write short responces to emails based on their importance. The importance will be provided to you by the 'classifier' agent.",
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
)

classify_email = Task(
    description=f"Classify the following email: '{email}'",
    agent=classifier,
    expected_output="One of these three options: 'important', 'casual', or 'spam'.",
)

respond_to_email = Task(
    description=f"Respond to the email: '{email}' based on the importance provided by the 'classifier' agent",
    agent=responder,
    expected_output="a very concise responce to the email based on the importance provided by the 'classifier' agent.",
)

crew = Crew(
    agents=[classifier, responder],
    tasks=[classify_email, respond_to_email],
    verbose=True,
)

# Execute the crew
result = crew.kickoff()

print(result)
