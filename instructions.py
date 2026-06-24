"""
Fixed pool of 50 prefix-sensitive instructions, spanning five task categories.

An instruction is prefix-sensitive when a persona or style prefix produces
noticeably different outputs for it. Instructions with a single correct short
answer are excluded because the judge (cosine similarity) would not reliably
distinguish prefix effects from answer correctness.
"""

INSTRUCTIONS: list[str] = [
    # Factual Q&A (10)
    "What causes the seasons to change on Earth?",
    "How does the immune system recognize and fight viruses?",
    "Why is the sky blue during the day but red at sunset?",
    "What is the difference between a democracy and a republic?",
    "How do vaccines train the immune system?",
    "Why do objects fall at the same rate in a vacuum regardless of mass?",
    "What is the significance of the Magna Carta?",
    "How does the brain store and retrieve memories?",
    "What causes inflation in an economy?",
    "Why do some people need glasses and others do not?",

    # Explanation (10)
    "Explain how the internet works.",
    "Explain the concept of entropy in thermodynamics.",
    "Explain why compound interest is powerful over long time horizons.",
    "Explain what machine learning is to someone who has never heard of it.",
    "Explain the difference between correlation and causation.",
    "Explain what DNA is and what it does.",
    "Explain how airplanes generate lift.",
    "Explain what a derivative is in calculus.",
    "Explain how search engines index and rank web pages.",
    "Explain what black holes are and how they form.",

    # Creative writing (10)
    "Write a short poem about the passage of time.",
    "Describe the feeling of seeing the ocean for the first time.",
    "Write a brief story about a robot who learns to paint.",
    "Write a metaphor for what it feels like to learn something difficult.",
    "Describe a thunderstorm from the perspective of a tree.",
    "Write a short piece about the last library on Earth.",
    "Describe what silence sounds like.",
    "Write a few sentences about a scientist who discovers she can talk to plants.",
    "Write a short ode to ordinary Tuesday mornings.",
    "Describe the smell of an old bookshop.",

    # Code (10)
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a function that returns the nth Fibonacci number.",
    "Write a Python function that flattens a nested list.",
    "Write a function that counts the frequency of each word in a string.",
    "Write a Python function that finds the longest common prefix in a list of strings.",
    "Write a function that converts a decimal number to binary.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write a function that rotates a list to the right by k positions.",
    "Write a Python function that checks whether parentheses in a string are balanced.",
    "Write a function that groups a list of words by their first letter.",

    # Advice / opinion (10)
    "What is the best way to learn a new skill as an adult?",
    "How should someone approach a disagreement with a close friend?",
    "What advice would you give to a first-time manager?",
    "How should a student decide which career to pursue?",
    "What are the most important habits for staying healthy over a lifetime?",
    "How do you evaluate whether a piece of writing is good?",
    "What is the best way to make a difficult decision under uncertainty?",
    "How should someone handle failure?",
    "What makes a team work well together?",
    "How do you stay motivated when progress feels slow?",
]
