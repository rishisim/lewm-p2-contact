import json
from pathlib import Path
R=Path(__file__).parent
def j(n): return json.loads((R/n).read_text())
def test_decision_mapping():
 assert j('evidence_ledger.json')['binding_evidence_count']==0
 assert j('decision.json')['decision']=='upload_metadata_contains_no_generator_binding'
def test_caps():
 q=j('query_inventory.json'); assert q['total_http_requests_conservative']<=48 and q['archive_payload_bytes']<=16777216
def test_archive_metadata_only():
 a=j('archive_header_metadata.json'); assert a['checksum_valid'] and a['name'].endswith('.h5') and not a['embedded_comment_digest_source_fields']
def test_zero_execution():
 m=j('metrics.json'); assert m['hdf5_rows_opened']==m['cube_environment_executions']==m['v5_episodes']==0
