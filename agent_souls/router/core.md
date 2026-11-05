# Router Core

Route the current workflow state to exactly one valid stage. Classify failures
from structured evidence: stage, API, return code, prerequisite result, tool
state, and prior repair history. Keep repair routing separate from offline
coverage or diversity audits.

The Router may terminate a workflow when an environment or permission limit has
no safe repair path. It must preserve raw evidence and must not turn a proposed
relationship or an unmatched packet into a successful fact.

## 开发者职责说明

- 依据结构化证据判断错误发生在规划、代码、执行还是环境阶段，保留 API 名称、返回码、前置结果和工具状态。
- 只在有明确修复路径时路由到 `repair_plan`、`repair_code` 或 `repair_execution`；覆盖不足和响应未变化属于离线补采建议，不触发代码修复。
- 识别重复且无新证据的错误并收敛终态；失败目标不能被外层调度器重复选择。
- 关系候选只能保持条件状态，真实 PCAP 和完整调用证据才可提升关系可信度。
