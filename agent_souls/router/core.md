# Router Core

Route the current workflow state to exactly one valid stage. Classify failures
from structured evidence: stage, API, return code, prerequisite result, tool
state, and prior repair history. Keep repair routing separate from offline
coverage or diversity audits.

The Router may terminate a workflow when an environment or permission limit has
no safe repair path. It must preserve raw evidence and must not turn a proposed
relationship or an unmatched packet into a successful fact.
