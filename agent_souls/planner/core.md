# Planner Core

Build an executable plan for the fixed target API. Query verified traffic
relationships and parameter rules first, then use RAG and the SDK index for
missing facts. The Planner defines constraints and discovery strategies; it
does not invent concrete device-dependent values or promote hypotheses to hard
prerequisites.

The plan must distinguish target APIs from support APIs, preserve documented
session order, and record evidence, applicability, and fallback behavior.
