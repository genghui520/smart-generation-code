from __future__ import annotations

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent.traffic_graph import (
    ApiNode,
    InMemoryTrafficGraphStore,
    TrafficGraphUnavailable,
    TrafficRelation,
    TrafficRelationshipService,
)
from smart_traffic_agent.traffic_relationship_mining import build_runtime_relation_payload, seed_documented_relations
from smart_traffic_agent.parameter_relationship_rules import resolve_rules, rules_for_target
from smart_traffic_agent.agents.executor import ExecutionAgent
from smart_traffic_agent.agent_tools.traffic_graph import RecordTrafficRelationsInput, RecordTrafficRelationsTool
from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, ExecutionResult, NcProgramSpec, PlanStep, TaskRequest, WorkflowState


class TrafficGraphTests(unittest.TestCase):
    def test_structured_runtime_relation_payload_is_parsed_by_tool(self) -> None:
        store = InMemoryTrafficGraphStore()
        service = TrafficRelationshipService(store)
        tool = RecordTrafficRelationsTool(service)
        result = tool.invoke(RecordTrafficRelationsInput({
            "schema_version": "trcm.runtime_relation_observations.v1",
            "relations": [{
                "relation": {
                    "source_api": "cnc_statinfo",
                    "target_api": "cnc_rdmacro",
                    "relation_type": "PRECEDES",
                    "rule_id": "runtime:test",
                    "confidence_status": "proposed",
                },
                "evidence": [{"evidence_id": "run-1"}],
            }],
        }))
        self.assertTrue(result["ok"])
        self.assertEqual(result["relation_count"], 1)
        self.assertEqual(
            next(iter(store.relations.values())).rule_id,
            "runtime:test",
        )

    def test_structured_payload_cannot_claim_runtime_verified_without_evidence(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        tool = RecordTrafficRelationsTool(service)
        result = tool.invoke(RecordTrafficRelationsInput({
            "relations": [{
                "source_api": "cnc_statinfo", "target_api": "cnc_rdmacro",
                "relation_type": "PRECEDES", "confidence_status": "runtime_verified",
            }],
        }))
        self.assertFalse(result["ok"])
        self.assertEqual(result["relation_count"], 0)

    def test_successful_knowledge_plan_promotes_candidate_relation(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-kb-1", task_id="task-kb-1", scenario_type="read",
            scenario_goal="read target", target_environment="simulator",
            nc_program_type="none", nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name=""), steps=[
                PlanStep("S1", "before", "source", "source", protocol_function="cnc_source"),
                PlanStep("S2", "during", "target", "target", protocol_function="cnc_target", role="target"),
            ], expected_outputs=[], target_api="cnc_target", dependencies=[{
                "source_api": "cnc_source", "target_api": "cnc_target", "relation_type": "filters",
                "rule_id": "kb:source-target", "dependency_path": "number",
                "source_field": "data.number", "filter_mode": "include",
            }],
        )
        logs = [
            ApiCallLog("t", "task-kb-1", "S1", "before", "source", {}, 0, {"data": {"number": 1}}, protocol_function="cnc_source"),
            ApiCallLog("t", "task-kb-1", "S2", "during", "target", {"number": 1}, 0, {}, protocol_function="cnc_target"),
        ]
        payload = build_runtime_relation_payload(plan, ExecutionResult("task-kb-1", True, logs, [], Path(".")))
        assert payload["perfect_execution"] is True
        assert payload["relations"][0]["relation"]["confidence_status"] == "proposed"
        assert payload["discovered_relations"][0]["status"] == "awaiting_user_approval"

    def test_f0id_parameter_rules_have_evidence_and_runtime_constraints(self) -> None:
        report = resolve_rules(
            ("cnc_rdmacro", "cnc_rdparam", "cnc_download3", "cnc_absolute", "cnc_skip"),
            allowed_api_names=("cnc_rdmacro", "cnc_rdmacroinfo", "cnc_rdparam", "cnc_rdparanum", "cnc_rdparainfo", "cnc_download3", "cnc_dwnstart3", "cnc_dwnend3", "cnc_rdprogdir3", "cnc_absolute", "cnc_sysinfo", "cnc_skip", "cnc_rdaxisname", "cnc_setpath", "cnc_wrparam", "cnc_wrmacro"),
            device_scope="fanuc:0i-D",
        )
        self.assertGreaterEqual(len(report["relations"]), 5)
        self.assertFalse(report["candidates"])
        for item in report["relations"]:
            self.assertEqual(item["confidence_status"], "sdk_validated")
            self.assertTrue(item["source_fields"])
            self.assertTrue(item["target_parameters"])
            self.assertTrue(item["transform"])
            self.assertTrue(item["evidence_ids"])

    def test_parameter_discovery_rules_keep_values_runtime_selected(self) -> None:
        report = rules_for_target("cnc_rdmacro")
        self.assertTrue(report["relations"])
        self.assertTrue(all(item["target_parameters"] for item in report["relations"]))
        self.assertTrue(all("require_runtime_discovery" not in item or item["require_runtime_discovery"] for item in report["relations"]))

    def test_parameter_discovery_seed_is_idempotent_and_scoped(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_sysinfo", "cnc_rdaxisname", "cnc_rdposition", "cnc_absolute", "cnc_rdmacro", "cnc_rdmacroinfo")
        seed_documented_relations(service, ("cnc_absolute", "cnc_rdmacro"), allowed_api_names=allowed, device_scope="fanuc:0i-D")
        first = len(service.store.relations)
        seed_documented_relations(service, ("cnc_absolute", "cnc_rdmacro"), allowed_api_names=allowed, device_scope="fanuc:0i-D")
        self.assertEqual(first, len(service.store.relations))
        self.assertTrue(all(
            item.applicability.get("device_scope") == "fanuc:0i-D"
            for item in service.store.relations.values()
            if item.attributes.get("discovery_family")
        ))

    def test_sdk_validated_status_is_chain_valid(self) -> None:
        store = InMemoryTrafficGraphStore()
        store.upsert_api(ApiNode("focas:source", "source"))
        store.upsert_api(ApiNode("focas:target", "target"))
        store.upsert_relation(TrafficRelation("rel:sdk", "source", "target", "PROVIDES_INPUT", confidence_status="sdk_validated"))
        self.assertTrue(store.validate_traffic_chain("target")["valid"])

    def test_relation_failure_prunes_candidates_but_preserves_verified_edges(self) -> None:
        store = InMemoryTrafficGraphStore()
        store.upsert_api(ApiNode("focas:source", "source"))
        store.upsert_api(ApiNode("focas:target", "target"))
        store.upsert_relation(TrafficRelation("rel:proposed", "source", "target", "REQUIRES", confidence_status="proposed"))
        store.upsert_relation(TrafficRelation("rel:sdk", "source", "target", "PROVIDES_INPUT", confidence_status="sdk_validated"))
        rejected = store.record_relation_failure("rel:proposed", [{"evidence_id": "err-1"}], reason="EW_ATTRIB")
        preserved = store.record_relation_failure("rel:sdk", [{"evidence_id": "err-2"}], reason="device_option")
        self.assertEqual(rejected["relation"]["confidence_status"], "rejected")
        self.assertEqual(preserved["relation"]["confidence_status"], "sdk_validated")
        self.assertTrue(preserved["relation"]["attributes"]["prune_history"])
    def test_proposed_relation_never_validates_as_mandatory(self) -> None:
        store = InMemoryTrafficGraphStore()
        store.upsert_api(ApiNode("focas:source", "source"))
        store.upsert_api(ApiNode("focas:target", "target"))
        relation = TrafficRelation(
            "rel:test", "source", "target", "REQUIRES", confidence_status="proposed"
        )
        store.upsert_relation(relation)

        self.assertFalse(store.validate_traffic_chain("target")["valid"])
        verified = store.verify_relation("rel:test", [{"evidence_id": "run-1"}])
        self.assertEqual(verified["relation"]["confidence_status"], "runtime_verified")
        self.assertTrue(store.validate_traffic_chain("target")["valid"])

    def test_download_target_seeds_documented_start_and_end_chain(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download3"])
        chain = service.call("get_traffic_chain", "cnc_download3", 3)

        names = {row["name"] for row in chain["nodes"]}
        pairs = {(row["source_api"], row["target_api"]) for row in chain["relations"]}
        self.assertTrue({"cnc_download3", "cnc_dwnstart3", "cnc_dwnend3"} <= names)
        self.assertIn(("cnc_dwnstart3", "cnc_download3"), pairs)
        self.assertIn(("cnc_download3", "cnc_dwnend3"), pairs)

    def test_sram_read_target_seeds_discovery_filter_and_session_chain(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_getsraminfo", "cnc_sramgetstart", "cnc_sramget", "cnc_sramgetend")
        seed_documented_relations(
            service, ["cnc_sramget"], allowed_api_names=allowed, device_scope="fanuc:0i-D"
        )
        rows = service.call("get_api_dependencies", "cnc_sramget", device_scope="fanuc:0i-D", allowed_api_names=allowed)["dependencies"]
        relations = [item["relation"] for item in rows]
        pairs = {(item["source_api"], item["target_api"]) for item in relations}
        self.assertIn(("cnc_sramgetstart", "cnc_sramget"), pairs)
        self.assertIn(("cnc_sramget", "cnc_sramgetend"), pairs)

        start_rows = service.call("get_api_dependencies", "cnc_sramgetstart", device_scope="fanuc:0i-D", allowed_api_names=allowed)["dependencies"]
        discovery = [
            item["relation"] for item in start_rows
            if item["relation"]["source_api"] == "cnc_getsraminfo"
            and item["relation"]["relation_type"] == "filters"
        ]
        self.assertEqual(len(discovery), 1)
        attrs = discovery[0]["attributes"]
        self.assertEqual(attrs["source_field"], "info[].sramname")
        self.assertEqual(attrs["dependency_path"], "name")
        self.assertEqual(attrs["filter_mode"], "include")
        self.assertEqual(attrs["transform"], "identity")

        session = [item for item in relations if item["target_api"] == "cnc_sramgetend"]
        self.assertEqual(session[0]["attributes"]["end_condition"], "cnc_sramget.stat == 0")

    def test_sram_read_relation_seed_is_idempotent(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_getsraminfo", "cnc_sramgetstart", "cnc_sramget", "cnc_sramgetend")
        seed_documented_relations(service, ["cnc_sramget"], allowed_api_names=allowed, device_scope="fanuc:0i-D")
        first = len(service.store.relations)
        seed_documented_relations(service, ["cnc_sramget"], allowed_api_names=allowed, device_scope="fanuc:0i-D")
        self.assertEqual(first, len(service.store.relations))

    def test_cexesram_relations_seed_size_before_read_and_write(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_cexesramsize", "cnc_rdcexesram", "cnc_wrcexesram")
        seed_documented_relations(
            service,
            ["cnc_rdcexesram", "cnc_wrcexesram", "cnc_cexesramsize"],
            allowed_api_names=allowed,
            device_scope="fanuc:0i-D",
        )
        overview = service.call(
            "get_graph_overview",
            device_scope="fanuc:0i-D",
            allowed_api_names=allowed,
        )
        pairs = {
            (row["source_api"], row["target_api"], row["relation_type"])
            for row in overview["relations"]
        }
        self.assertIn(("cnc_cexesramsize", "cnc_rdcexesram", "prepares"), pairs)
        self.assertIn(("cnc_cexesramsize", "cnc_wrcexesram", "prepares"), pairs)
        self.assertIn(("cnc_rdcexesram", "cnc_wrcexesram", "prepares"), pairs)

    def test_alarm_history_seeds_stop_count_read_and_restart_chain(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_stopophis", "cnc_rdalmhisno", "cnc_rdalmhistry5", "cnc_startophis")
        seed_documented_relations(
            service,
            ["cnc_rdalmhistry5"],
            allowed_api_names=allowed,
            device_scope="fanuc:0i-D",
        )
        overview = service.call(
            "get_graph_overview",
            device_scope="fanuc:0i-D",
            allowed_api_names=allowed,
        )
        pairs = {
            (row["source_api"], row["target_api"], row["relation_type"])
            for row in overview["relations"]
        }
        self.assertIn(("cnc_stopophis", "cnc_rdalmhisno", "prepares"), pairs)
        self.assertIn(("cnc_rdalmhisno", "cnc_rdalmhistry5", "prepares"), pairs)
        self.assertIn(("cnc_stopophis", "cnc_rdalmhistry5", "prepares"), pairs)
        self.assertIn(("cnc_rdalmhistry5", "cnc_startophis", "prepares"), pairs)

    def test_axis_and_spindle_name_apis_are_seeded_as_one_family_without_order_edges(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        names = ("cnc_rdaxisname", "cnc_exaxisname", "cnc_exaxisname2", "cnc_rdspdlname")
        seed_documented_relations(service, names, allowed_api_names=names, device_scope="fanuc:0i-D")
        for name in names:
            node = service.store.nodes[f"focas:{name}"]
            self.assertEqual(node.metadata["knowledge_family"], "controlled_axis_spindle_names")
        overview = service.call("get_graph_overview", device_scope="fanuc:0i-D", allowed_api_names=names)
        self.assertEqual(overview["relation_count"], 0)

    def test_upload_target_seeds_its_matching_session_only(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_upload3"])
        chain = service.call("get_traffic_chain", "cnc_upload3", 3)
        names = {row["name"] for row in chain["nodes"]}
        pairs = {(row["source_api"], row["target_api"]) for row in chain["relations"]}
        self.assertTrue({"cnc_upload3", "cnc_upstart3", "cnc_upend3"} <= names)
        self.assertIn(("cnc_upstart3", "cnc_upload3"), pairs)
        self.assertIn(("cnc_upload3", "cnc_upend3"), pairs)

    def test_program_chain_keeps_directory_discovery_optional_and_exposes_rules(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_upload3", "cnc_download3"])

        upload_rules = service.call("get_parameter_rules", "cnc_upstart3")
        self.assertTrue(upload_rules["available"])
        self.assertTrue({"s_number", "e_number", "type"} <= {
            item["parameter"] for item in upload_rules["rules"]
        })

        download_rules = service.call("get_parameter_rules", "cnc_download3")
        self.assertTrue(download_rules["available"])
        program_rule = next(item for item in download_rules["rules"] if item["parameter"] == "program_number")
        self.assertEqual(program_rule["filter_strategy"], "unused_unprotected_program")

        dependencies = service.call("get_api_dependencies", "cnc_upstart3")["dependencies"]
        discovery = [item for item in dependencies if item["relation"]["relation_type"] == "PROVIDES_INPUT"]
        self.assertEqual(len(discovery), 1)
        self.assertFalse(discovery[0]["relation"]["applicability"]["required"])
        self.assertEqual(
            discovery[0]["relation"]["applicability"]["required_if_missing_inputs"],
            ["s_number", "e_number"],
        )
        self.assertEqual(
            discovery[0]["relation"]["attributes"]["selection_rule"],
            "registered_readable_program",
        )
        self.assertEqual(
            discovery[0]["relation"]["attributes"]["forbid_default_values"],
            {"s_number": [0], "e_number": [0]},
        )

    def test_program_chain_relations_carry_repeatable_data_and_evidence(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download3"])
        rows = service.call("get_api_dependencies", "cnc_download3")["dependencies"]
        session = [item for item in rows if item["relation"]["relation_type"] == "SESSION_SEQUENCE"]
        self.assertTrue(session)
        self.assertTrue(session[0]["relation"]["attributes"]["data_repeatable"])
        self.assertTrue(session[0]["relation"]["evidence_ids"])

    def test_download4_seeds_optional_program_directory_provider(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download4"])

        dependencies = service.call("get_api_dependencies", "cnc_download4")["dependencies"]
        discovery = [
            item for item in dependencies
            if item["relation"]["relation_type"] == "PROVIDES_INPUT"
        ]
        self.assertEqual(len(discovery), 1)
        relation = discovery[0]["relation"]
        self.assertEqual(relation["source_api"], "cnc_rdprogdir3")
        self.assertEqual(relation["attributes"]["source_api"], "cnc_rdprogdir3")
        self.assertEqual(relation["attributes"]["chain_id"], "focas_program_download4")
        self.assertFalse(relation["applicability"]["required"])
        self.assertTrue(relation["evidence_ids"])

        rules = service.call("get_parameter_rules", "cnc_download4")["rules"]
        program_rule = next(item for item in rules if item["parameter"] == "program_number")
        self.assertEqual(program_rule["source_api"], "cnc_rdprogdir3")
        self.assertEqual(program_rule["filter_strategy"], "unused_unprotected_program")
        self.assertIn("type", {item["parameter"] for item in rules})
        self.assertIn("length", {item["parameter"] for item in rules})

    def test_upload4_seeds_optional_program_directory_provider(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_upload4"])
        dependencies = service.call("get_api_dependencies", "cnc_upload4")["dependencies"]
        discovery = [
            item for item in dependencies
            if item["relation"]["relation_type"] == "PROVIDES_INPUT"
        ]
        self.assertEqual(len(discovery), 1)
        self.assertEqual(discovery[0]["relation"]["source_api"], "cnc_rdprogdir3")
        rules = service.call("get_parameter_rules", "cnc_upload4")["rules"]
        self.assertEqual(
            next(item for item in rules if item["parameter"] == "s_number")["source_api"],
            "cnc_rdprogdir3",
        )

    def test_program_chain_does_not_skip_the_repeatable_data_step(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_upload3", "cnc_download3"])
        session_edges = {
            (item.source_api, item.target_api)
            for item in service.store.relations.values()
            if item.relation_type == "SESSION_SEQUENCE"
        }
        self.assertNotIn(("cnc_upstart3", "cnc_upend3"), session_edges)
        self.assertNotIn(("cnc_dwnstart3", "cnc_dwnend3"), session_edges)
        self.assertIn(("cnc_upstart3", "cnc_upload3"), session_edges)
        self.assertIn(("cnc_upload3", "cnc_upend3"), session_edges)
        self.assertIn(("cnc_dwnstart3", "cnc_download3"), session_edges)
        self.assertIn(("cnc_download3", "cnc_dwnend3"), session_edges)

    def test_unconfigured_service_is_explicitly_unknown(self) -> None:
        result = TrafficRelationshipService(None).call("get_api_dependencies", "cnc_statinfo")
        self.assertFalse(result["available"])
        self.assertEqual(result["knowledge_status"], "unknown")

    def test_offline_configured_graph_does_not_block_documented_seed(self) -> None:
        class OfflineStore:
            def upsert_api(self, *_args, **_kwargs):
                return {"available": False, "backend": "neo4j"}

            def upsert_relation(self, *_args, **_kwargs):
                return {"available": False, "backend": "neo4j"}

            def migrate_legacy_relations(self):
                raise TrafficGraphUnavailable("offline")

            def _run(self, *_args, **_kwargs):
                raise TrafficGraphUnavailable("offline")

        service = TrafficRelationshipService(OfflineStore())
        rows = seed_documented_relations(
            service,
            ["cnc_download3"],
            allowed_api_names=("cnc_download3", "cnc_dwnstart3", "cnc_dwnend3", "cnc_rdprogdir3"),
            device_scope="fanuc:0i-D",
        )
        self.assertTrue(rows)

    def test_environment_service_loads_local_credentials_without_exposing_password(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            credentials = Path(tmp) / "neo4j.credentials"
            credentials.write_text(
                "uri=bolt://127.0.0.1:7687\nuser=neo4j\npassword=secret-value\ndatabase=neo4j\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "SMP_TRG_NEO4J_CREDENTIALS_FILE": str(credentials),
                "SMP_TRG_NEO4J_URI": "",
                "SMP_TRG_NEO4J_USER": "",
                "SMP_TRG_NEO4J_PASSWORD": "",
                "SMP_TRG_NEO4J_DATABASE": "",
            }, clear=False):
                service = TrafficRelationshipService.from_environment()
            self.assertTrue(service.available)
            self.assertEqual(service.store.uri, "bolt://127.0.0.1:7687")
            self.assertEqual(service.store.user, "neo4j")
            self.assertEqual(service.store.password, "secret-value")
            diagnostic = service.unavailable("test")
            self.assertNotIn("secret-value", str(diagnostic))

    def test_device_scope_filters_graph_endpoints_without_deleting_global_relations(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download3", "cnc_statinfo"])
        allowed = ("cnc_download3", "cnc_dwnstart3", "cnc_dwnend3")
        chain = service.call(
            "get_traffic_chain", "cnc_download3", 3,
            device_scope="fanuc:0i-D", allowed_api_names=allowed,
        )
        names = {row["name"] for row in chain["nodes"]}
        self.assertTrue(names <= set(allowed))
        self.assertIn("cnc_download3", names)
        self.assertIn("cnc_statinfo", service.store.nodes)
        self.assertGreater(len(service.store.relations), 0)

    def test_graph_overview_supports_full_view_and_name_search(self) -> None:
        store = InMemoryTrafficGraphStore()
        for name in ("cnc_statinfo", "cnc_rdprogdir3", "cnc_download3", "cnc_other"):
            store.upsert_api(ApiNode(f"focas:{name}", name))
        store.upsert_relation(TrafficRelation("rel:one", "cnc_rdprogdir3", "cnc_download3", "PROVIDES_INPUT", confidence_status="documented"))
        store.upsert_relation(TrafficRelation("rel:two", "cnc_download3", "cnc_statinfo", "PRECEDES", confidence_status="runtime_verified"))
        service = TrafficRelationshipService(store)
        full = service.call("get_graph_overview", device_scope="fanuc:0i-D", allowed_api_names=("cnc_statinfo", "cnc_rdprogdir3", "cnc_download3"))
        self.assertEqual(full["node_count"], 3)
        self.assertEqual(full["relation_count"], 2)
        searched = service.call("get_graph_overview", query="download", device_scope="fanuc:0i-D", allowed_api_names=("cnc_statinfo", "cnc_rdprogdir3", "cnc_download3"))
        self.assertEqual({node["name"] for node in searched["nodes"]}, {"cnc_statinfo", "cnc_rdprogdir3", "cnc_download3"})
        self.assertEqual(searched["relation_count"], 2)

    def test_scoped_seed_uses_distinct_relation_ids(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download3"])
        global_ids = set(service.store.relations)
        seed_documented_relations(
            service, ["cnc_download3"], allowed_api_names=("cnc_download3", "cnc_dwnstart3", "cnc_dwnend3"),
            device_scope="fanuc:0i-D",
        )
        self.assertTrue(global_ids)
        self.assertGreater(len(service.store.relations), len(global_ids))
        self.assertTrue(any("scope_fanuc:0i-D" in item for item in service.store.relations if item not in global_ids))
        dependencies = service.call(
            "get_api_dependencies", "cnc_download3", device_scope="fanuc:0i-D",
            allowed_api_names=("cnc_download3", "cnc_dwnstart3", "cnc_dwnend3"),
        )["dependencies"]
        keys = {(row["relation"]["source_api"], row["relation"]["relation_type"], row["relation"]["rule_id"]) for row in dependencies}
        self.assertEqual(len(keys), len(dependencies))

    def test_executor_persists_relationship_feedback_without_changing_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "trg-feedback"
            state = WorkflowState(request=TaskRequest("read two APIs", task_id))
            state.plan = ExecutionPlan(
                "plan-trg", task_id, "parameter_read", "read", "simulator", "", [],
                NcProgramSpec(""),
                [
                    PlanStep("S1", "before", "read", "ReadStatus", {}, protocol_function="cnc_statinfo"),
                    PlanStep("S2", "during", "read", "ReadMacro", {}, protocol_function="cnc_rdmacro"),
                ], [],
            )
            state.result = ExecutionResult(
                task_id, True,
                [
                    ApiCallLog("t", task_id, "S1", "before", "ReadStatus", {}, 0, {}, protocol_function="cnc_statinfo"),
                    ApiCallLog("t", task_id, "S2", "during", "ReadMacro", {}, 0, {}, protocol_function="cnc_rdmacro"),
                ], [], Path(tmp),
            )
            store = InMemoryTrafficGraphStore()
            store.upsert_api(ApiNode("focas:cnc_statinfo", "cnc_statinfo"))
            store.upsert_api(ApiNode("focas:cnc_rdmacro", "cnc_rdmacro"))
            agent = ExecutionAgent()
            agent.traffic_relationships = TrafficRelationshipService(store)
            agent._record_relationship_feedback(state)

            self.assertTrue(state.result.success)
            self.assertEqual(state.plan.rag_context["traffic_relationship_feedback"]["status"], "recorded")
            self.assertTrue((Path(tmp) / "traffic_relationship_feedback.json").is_file())


if __name__ == "__main__":
    unittest.main()
