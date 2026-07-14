Analyse the entire project and update the test files to accurately cover the current state of the application. Follow these steps:

1. Read pyproject.toml to understand the project name, dependencies, and test configuration.
2. Read every file inside app/ recursively to understand:
   - All API endpoints (routes, methods, request body, response shape, status codes)
   - Authentication and authorization logic
   - Database models
   - Any business logic functions worth unit testing
3. Read all existing files inside tests/ to understand what is already tested — do not duplicate existing tests.
4. Based on everything you read, update tests/ with the following:
   - One test file per router (e.g. test_auth.py for app/routers/auth.py)
   - A test_main.py for the core app endpoints (e.g. /health)
   - For each endpoint, write tests for:
     - Happy path (valid input, expected response and status code)
     - Error cases (invalid input, duplicate data, wrong credentials, missing token)
   - For auth-protected endpoints, write one test with a valid token and one without
   - Use httpx AsyncClient for async tests
   - Use pytest fixtures for the test client and any shared setup

5. Rules:
   - Only test what actually exists in the code. Do not invent or assume features.
   - Use pytest and httpx — already in dev dependencies.
   - Each test function name must clearly describe what it tests (e.g. test_register_returns_201_on_valid_input).
   - Do not delete existing passing tests — only add or update.
   - After updating, print a short summary of which test files were created or updated and how many tests were added.
