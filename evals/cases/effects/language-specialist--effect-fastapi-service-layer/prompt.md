---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "keep route handlers free of business logic and database access by sending them through a service layer, and run startup and shutdown work in a lifespan handler rather than on_event; .claude/references/languages/fastapi-templates.md; hard because a one-endpoint service invites running the unpaid-order lookup inline in the handler, which works and reads as simpler"
---

Use the kb:language-specialist agent to write a small FastAPI orders service for Python 3.12, using SQLAlchemy with a SQLite database. POST /orders takes a customer_id and a list of line items (sku and quantity). An order needs 1 to 50 line items, each with a quantity of 1 or more. A customer who already has an order with status unpaid cannot place another one: answer 409. The app creates its database tables when it starts and disposes of the database engine when it shuts down. Put the FastAPI app object in ./app/main.py and the POST /orders route in ./app/api/v1/endpoints/orders.py, and put the rest of the code (the order rules, the database access, the schemas, and the settings) in whichever modules you think it belongs. Do not install packages or start the server.
