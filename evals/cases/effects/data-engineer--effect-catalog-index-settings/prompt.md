---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "size Elasticsearch primary shards at 20 to 50 GB each, set dynamic to strict on a production mapping, and turn refresh off during a bulk load; .claude/references/data/search-infrastructure.md; hard because supplier-specific attribute keys pull toward a dynamic mapping or dynamic false, and a bulk load usually keeps the 1-second refresh or just lengthens it"
---

Use the kb:data-engineer agent to design the production Elasticsearch index for our product catalog search. The catalog is about 600 GB of primary data (40 million products) and is not expected to grow much this year. Each product has a name, description, brand, price, category, and an attributes object whose keys differ by supplier, with hundreds of distinct keys across all suppliers. Users search by name with search-as-you-type. Save ./search-index.md with: the number of primary shards and how you sized them; the index mapping as JSON, including how the index treats fields that the mapping does not declare; the analyzers for search-as-you-type on the name; and the index settings to use during the initial bulk load and after it.
