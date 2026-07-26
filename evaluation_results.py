#how you would run the above function

baseline_semantic = evaluate_retrieval(
    "baseline",
    "semantic"
)

baseline_svm = evaluate_retrieval(
    "baseline",
    "svm"
)

baseline_llm = evaluate_retrieval(
    "baseline",
    "llm"
)

conflict_semantic = evaluate_retrieval(
    "conflict",
    "semantic"
)

conflict_svm = evaluate_retrieval(
    "conflict",
    "svm"
)

conflict_llm = evaluate_retrieval(
    "conflict",
    "llm"
)