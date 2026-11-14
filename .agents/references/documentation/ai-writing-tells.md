# AI Writing Tells

Prose that a person reads outside the session should read as written by a careful author. Detectors and
attentive readers flag the same habits; this reference lists them and how to revise them. The goal is
better prose for the reader, not concealment: follow any venue or employer policy on disclosing AI
assistance.

## When To Apply

- Narrative deliverables: paper sections, README and guide text, design docs.
- Not chat replies, plans, or agent reports; those follow the session's communication rules.
- Leave intentionally uniform text alone: API reference, changelog entries, tables, numbered steps, and
  the instruction style in `technical-writing-for-devtools`.
- In deliverables, keep one idea per sentence but vary sentence length (see Rhythm by Rule).

## How Detectors Score Text

| Signal | What it measures | What trips it |
|---|---|---|
| Perplexity | How predictable each next word is to a language model | Always choosing the most likely word or phrase |
| Burstiness | How much sentence length and structure vary | Sentences of near-equal length; paragraphs of equal size |
| Stock phrasing | Words and frames overrepresented in model output | "delve", "tapestry", "moreover", "it's important to note" |
| Template shape | Document-level structure | An intro that restates the prompt, a list-shaped body, a conclusion opening "Overall" that repeats the thesis |
| Vagueness | How specific the claims are | No numbers, names, or examples; no stated position |

Detectors are unreliable: formal and technical human prose also scores as predictable, and no score
proves anything either way. Treat the table as habits to revise, not a score to beat.

## Tell Catalog

### Staging Instead of Stating
- "It's not X, it's Y" when nobody claimed X. State Y.
- One-line closers and dramatic fragments. Cut them.
- Run-ups before the point ("Here's the thing:", "The answer? Simple."). Start with the point.
- Sayings that sound deep ("At its core, ..."). Say the specific thing.
- Rebutting objections nobody raised. Delete them.

### Rhythm by Rule
- Lists of three by habit. Use as many items as there are.
- Several sentences in a row that open the same way. Vary the openings.
- Uniform sentence and paragraph length. Mix short sentences with longer ones that carry a qualification
  or an example.
- Dashes as the all-purpose connector. Use a comma, colon, semicolon, or a new sentence.
- Stacked hedges ("could potentially", "may possibly"). Keep one, or state the evidence.

### Inflation and Borrowed Authority
- Stock words: delve, tapestry, testament, landscape, realm, pivotal, showcase, underscore, foster,
  seamless, robust. Use the plain word. The tell is density, not a single use.
- Inflated significance ("a pivotal moment", "reshapes the field"). State the measured effect.
- "Serves as", "stands as", "boasts" where "is" or "has" fits. Use "is" or "has".
- Trailing "-ing" clauses that add vague meaning (", highlighting its importance"). Cut them or make the
  claim.
- Vague attribution ("experts say", "studies show"). Cite the source or drop the claim.
- Vague links ("is associated with", "plays a role in") where the relation is known. Name it.

### Formatting by Rule
- Bold on phrases that are not defined terms or warnings.
- Emoji or arrows in headings, or a heading style that differs from the rest of the document.
- Curly quotes in plain-text or code-adjacent output.
- A heading repeated as its section's first sentence.

### Chat and Draft Leftovers
- "Great question", "I hope this helps", "Let me know if you need anything else".
- Knowledge-limit disclaimers in a finished document.
- Writing about the document ("This section will explore") instead of its subject.

### Wrong Reader
- Re-explaining what the stated audience already knows.
- Putting the result or decision last. Lead with it.

## Revision Pass

1. Draft for the reader: lead with the result; use concrete numbers, names, and examples; take the
   position the evidence supports.
2. Scan for the strong tells first: staging, stock words, template shape. Rewrite the sentence, not the
   word; a swapped synonym keeps the shape.
3. Read each paragraph for rhythm. If every sentence runs about the same length, split one and combine two.
4. Check that no fact, number, citation, or claim changed or disappeared.
5. Leave code, LaTeX commands (`\cite`, `\ref`, `\label`), paths, and link targets byte-identical.

## When Not To Act

- One weak tell (a single dash, triad, or bolded term) is not a defect; act when several cluster in one
  passage.
- Quotations, titles, proper names, and text that discusses a phrase keep their wording.
- Never invent detail, anecdote, or opinion to sound human. With nothing specific to say, write a
  shorter, plainer sentence.

## Papers

- State contributions as specific, checkable claims, not "we delve into" or "a novel framework".
- Keep the field's normal register: hedged claims and connectives such as "moreover" are fine at the
  density the venue's published papers use.
