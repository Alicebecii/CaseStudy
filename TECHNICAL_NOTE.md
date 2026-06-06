# Technical note: two key design decisions

## 1. Contracts and factories, with a from-scratch agent loop

The single most consequential choice was to put **every swappable part behind a tiny
interface in `core/`** (`LLMProvider`, `Embedder`, `Retriever`, `Tool`, `Validator`) and
wire concrete implementations in one place per package (a factory). The agent depends on
`Retriever` and `LLMProvider`, never on the hybrid retriever or the Ollama client, so a
backend can be added, removed, or rewritten by touching only its factory - the mock
backend that powers the offline tests, the local Ollama model, and the OpenAI API are one
environment variable apart.

The same principle drove writing the **ReAct loop from scratch** rather than adopting
LangChain. The loop is ~70 lines of visible control flow: ask the model, run the tool it
requests, feed the result back, repeat until it answers or the step budget runs out. The
cost is more code to own; the benefit is that the behaviour is fully inspectable and
testable - the loop is unit-tested deterministically with a scripted mock model, and tool
or argument errors become observations the agent recovers from rather than crashes. For a
system explicitly expected to change a lot, keeping control flow visible and seams small
mattered more than saving a few lines.

## 2. Grounding over fluency: two-layer validation

The second decision was to treat **an uncited or unsupported claim as a failure, not a
feature**. A deterministic, LLM-free grounding check always runs as the floor: it measures
how much of an answer's content vocabulary actually appears in the chunks it cites, so it
works offline, is not the author model grading itself, and makes the tests meaningful. When
a real backend is configured, an independent LLM critic adds a second opinion, combined by
a conservative worst-verdict-wins rule so the critic can only lower the verdict, never
inflate it. On a failing verdict the agent gets one bounded retry with the critique fed
back; if it still cannot ground the answer, the system returns it flagged rather than
dressed up as fact. In the demo this layer correctly certifies soundly-cited answers and
flags plausible-but-mis-cited ones - refusing to over-claim is itself the reliability
feature.
