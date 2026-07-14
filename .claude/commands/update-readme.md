Analyse the entire project and update README.md to accurately reflect the current state of the application. Follow these steps:

1. Read pyproject.toml to understand project name, version, description, dependencies, Python version, and scripts.
2. Read every file inside app/ recursively to understand:
   - All API endpoints (routes, methods, request/response shapes)
   - Project structure and modules
   - Key functions and what they do
   - Any environment variables or config being used
3. Read tests/ to understand what is being tested.
4. Based on everything you read, rewrite README.md with these sections:
   - Project name and a one-line description (from pyproject.toml)
   - ## Features — bullet points of what the app actually does (from the code, not assumptions)
   - ## Project Structure — the folder tree with a short note on each file's purpose
   - ## Requirements — Python version and key dependencies
   - ## Installation — exact commands to set up from scratch
   - ## Running the App — single command to start the server
   - ## API Endpoints — table with Method, Path, Description for every route found in app/
   - ## Running Tests — command to run tests
   - ## Environment Variables — any .env vars or config the app reads (if none found, omit this section)

Rules:
- Only document what actually exists in the code. Do not invent or assume features.
- Keep language clear and direct — this README is for a developer setting up the project for the first time.
- Do not add emojis.
- After updating, print a short summary of what changed.
