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
        "expected_values": ["Lili Abbie Brekke", "Marinda Lindsay Veum", "Gary Everette Abshire", "Gabrielle Claudie Medhurst"],
    },
    {
        "question": "Did the practitioner 'Arla Fritsch' treat more than one patient?",
        "expected_values": ["yes"],
    },
    {
        "question": "What are the unique categories of substances patients are allergic to?",
        "expected_values": ["medication", "environment", "food", "other"],
    },
    {
        "question": "How many patients were born in between the years 1990 and 2000?",
        "expected_values": ["184"],
    },
    {
        "question": "How many patients have been immunized after January 1, 2022?",
        "expected_values": ["65"],
    },
    {
        "question": "Which practitioner treated the most patients? Return their full name and how many patients they treated.",
        "expected_values": ["Ted Reilly", "19"],
    },
    {
        "question": "Is the patient ID 45 allergic to the substance 'shellfish'? If so, what city and state do they live in, and what is the full name of the practitioner who treated them?",
        "expected_values": ["East Longmeadow", "Cletus Paucek", "Massachusetts"],
    },
    {
        "question": "How many patients are immunized for influenza?",
        "expected_values": ["204"],
    },
    {
        "question": "How many substances cause allergies in the category 'food'?",
        "expected_values": ["13"],
    },
]
