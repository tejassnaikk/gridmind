RETRIEVAL_CONFIG: dict = {
    "candidate_pool": 40,
    "final_k": 5,
    "rrf_k": 60,
    "w_dense": 1.0,
    "w_sparse": 0.6,
    "freshness_current": 1.00,
    "freshness_superseded": 0.50,
    "obligation": {
        "shall": 1.00,
        "should": 0.92,
        "may": 0.85,
        "informational": 0.80,
    },
    "crossref_penalty": 0.85,
}
