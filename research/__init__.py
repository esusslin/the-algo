"""Research plane — LOCAL ONLY.

Never import this package from server.py. It pulls heavy dependencies
(duckdb, numpyro, shap) that are deliberately absent from the Railway build,
and one stray import will break the deploy.
"""
