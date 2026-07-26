evaluation_dataset = [

{
    "name": "duplicate_food",

    "conversation": [
        "My favorite food is sushi.",
        "My favorite food is sushi."
    ],

    "expected_baseline": [
        "My favorite food is sushi.",
        "My favorite food is sushi."
    ],

    "expected_conflict": [
        "My favorite food is sushi."
    ]
},

{
    "name": "duplicate_city",

    "conversation": [
        "I live in Boston.",
        "I live in Boston."
    ],

    "expected_baseline": [
        "I live in Boston.",
        "I live in Boston."
    ],

    "expected_conflict": [
        "I live in Boston."
    ]
},

{
    "name": "move_city",

    "conversation": [
        "I live in Boston.",
        "I moved to Seattle."
    ],

    "expected_baseline": [
        "I live in Boston.",
        "I moved to Seattle."
    ],

    "expected_conflict": [
        "I moved to Seattle."
    ]
},

{
    "name": "favorite_food_changed",

    "conversation": [
        "My favorite food is sushi.",
        "Actually my favorite food is ramen."
    ],

    "expected_baseline": [
        "My favorite food is sushi.",
        "My favorite food is ramen."
    ],

    "expected_conflict": [
        "My favorite food is ramen."
    ]
},

{
    "name": "job_changed",

    "conversation": [
        "I work as a teacher.",
        "I recently became a software engineer."
    ],

    "expected_baseline": [
        "I work as a teacher.",
        "I recently became a software engineer."
    ],

    "expected_conflict": [
        "I recently became a software engineer."
    ]
},

{
    "name": "pet_information",

    "conversation": [
        "I have a dog.",
        "His name is Max."
    ],

    "expected_baseline": [
        "I have a dog.",
        "His name is Max."
    ],

    "expected_conflict": [
        "I have a dog.",
        "His name is Max."
    ]
},

{
    "name": "family_information",

    "conversation": [
        "I have two sisters.",
        "One of them lives in New York."
    ],

    "expected_baseline": [
        "I have two sisters.",
        "One of them lives in New York."
    ],

    "expected_conflict": [
        "I have two sisters.",
        "One of them lives in New York."
    ]
},

{
    "name": "location_and_hobby",

    "conversation": [
        "I live in Boston.",
        "I enjoy hiking on weekends."
    ],

    "expected_baseline": [
        "I live in Boston.",
        "I enjoy hiking on weekends."
    ],

    "expected_conflict": [
        "I live in Boston.",
        "I enjoy hiking on weekends."
    ]
},

{
    "name": "random_preferences",

    "conversation": [
        "My favorite color is blue.",
        "I drink coffee every morning."
    ],

    "expected_baseline": [
        "My favorite color is blue.",
        "I drink coffee every morning."
    ],

    "expected_conflict": [
        "My favorite color is blue.",
        "I drink coffee every morning."
    ]
},

{
    "name": "birthday_and_music",

    "conversation": [
        "My birthday is July 3.",
        "I like jazz music."
    ],

    "expected_baseline": [
        "My birthday is July 3.",
        "I like jazz music."
    ],

    "expected_conflict": [
        "My birthday is July 3.",
        "I like jazz music."
    ]
},

{
    "name": "mixed_conversation",

    "conversation": [
        "I live in Boston.",
        "I have a dog.",
        "His name is Max.",
        "Actually I moved to Chicago.",
        "My favorite food is sushi.",
        "Actually my favorite food is ramen.",
        "I enjoy hiking."
    ],

    "expected_baseline": [
        "I live in Boston.",
        "I have a dog.",
        "His name is Max.",
        "I moved to Chicago.",
        "My favorite food is sushi.",
        "My favorite food is ramen.",
        "I enjoy hiking."
    ],

    "expected_conflict": [
        "I moved to Chicago.",
        "I have a dog.",
        "His name is Max.",
        "My favorite food is ramen.",
        "I enjoy hiking."
    ]
},

{
    "name": "temporary_context",

    "conversation": [
        "I'm traveling to Paris next week.",
        "I'll be back in Boston on Monday."
    ],

    "expected_baseline": [
        "I'm traveling to Paris next week.",
        "I'll be back in Boston on Monday."
    ],

    "expected_conflict": [
        "I'm traveling to Paris next week.",
        "I'll be back in Boston on Monday."
    ]
},

{
    "name": "goal_change",

    "conversation": [
        "I'm training for a marathon.",
        "Actually I've decided to train for a half marathon instead."
    ],

    "expected_baseline": [
        "I'm training for a marathon.",
        "I'm training for a half marathon."
    ],

    "expected_conflict": [
        "I'm training for a half marathon."
    ]
}

]

retrieval_dataset = [
    {
        "name": "move_city",
        "conversation": [
            "I live in Boston.",
            "I moved to Seattle."
        ],
        "question": "Where do I live?",

        "baseline_keywords": ["seattle"],
        "baseline_forbidden": ["boston"],

        "conflict_keywords": ["seattle"],
        "conflict_forbidden": ["boston"]
    },

    {
        "name": "favorite_food",
        "conversation": [
            "My favorite food is sushi."
        ],
        "question": "What's my favorite food?",

        "baseline_keywords": ["sushi"],
        "baseline_forbidden": [],

        "conflict_keywords": ["sushi"],
        "conflict_forbidden": []
    },

    {
        "name": "job_changed",
        "conversation": [
            "I work as a teacher.",
            "Now I work as software engineer."
        ],
        "question": "What do I do for work?",

        "baseline_keywords": ["software engineer"],
        "baseline_forbidden": ["teacher"],

        "conflict_keywords": ["software engineer"],
        "conflict_forbidden": ["teacher"]
    },

    {
        "name": "goal_change",
        "conversation": [
            "I'm training for a marathon.",
            "Actually, I'm training for a half marathon instead."
        ],
        "question": "What race am I training for?",

        "baseline_keywords": ["half marathon"],
        "baseline_forbidden": [],

        "conflict_keywords": ["half marathon"],
        "conflict_forbidden": []
    },

    {
        "name": "pet_information",
        "conversation": [
            "I have a golden retriever.",
            "His name is Max."
        ],
        "question": "Tell me about my dog.",

        "baseline_keywords": ["golden retriever", "max"],
        "baseline_forbidden": [],

        "conflict_keywords": ["golden retriever", "max"],
        "conflict_forbidden": []
    }
]