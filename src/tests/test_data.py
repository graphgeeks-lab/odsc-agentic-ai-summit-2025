"""
Test data for evaluating the Graph RAG workflow
"""

# For numerical answers, the expected values are strings that mention the number as an integer.
test_cases = [
    {
        "question": "How many patients with the last name 'Rosenbaum' received multiple immunizations?",
        "expected_values": ["1"],
    },
    {
        "question": "What are the full names of the patients treated by the practitioner named Josef Klein?",
        "expected_values": ["Lili Abbie Brekke", "Marinda Lindsay Veum", "Ana María Anita Barrios"],
    },
    {
        "question": "Did the practitioner 'Arla Fritsch' treat more than one patient? If so, return the patient's full names. ",
        "expected_values": ["Kerri Providencia Boyer", "Rogelio Clair Windler"],
    },
    {
        "question": "What are the unique categories of substances patients are allergic to?",
        "expected_values": ["food", "environment", "medication", "other"],
    },
    {
        "question": "How many patients were born in between the years 1990 and 2000?",
        "expected_values": ["12"],
    },
    {
        "question": "How many patients have been immunized after January 1, 2022?",
        "expected_values": ["6"],
    },
    {
        "question": "Which practitioner treated the most patients? Return their full name and how many patients they treated.",
        "expected_values": ["Vito Barton", "three"],
    },
    {
        "question": "Is the patient ID 45 allergic to the substance 'shellfish'? If so, what city and state do they live in, and what is the full name of the practitioner who treated them?",
        "expected_values": ["East Longmeadow", "Cletus Paucek", "Massachusetts"],
    },
    {
        "question": "How many patients are immunized for influenza?",
        "expected_values": ["14"],
    },
    {
        "question": "What substances cause food allergies in this database?",
        "expected_values": ["eggs", "shellfish", "wheat"],
    },
]
