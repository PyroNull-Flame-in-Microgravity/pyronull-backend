import importlib


def test_reference_schemas_import_and_resolve_investigation_measurements():
    schemas = importlib.import_module("reference.schemas")
    findings = schemas.InvestigationFindings(
        investigation_id="fixture-id",
        title="Example",
        conditions="Unknown",
        parameters=[schemas.MeasurementParameterSchema(name="burn_time")],
        supporting_source="Unverified fixture",
    )
    assert findings.parameters[0].name == "burn_time"