#!/usr/bin/env python3
import hashlib,json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def load(n): return json.loads((R/n).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
for n in ('PREREGISTRATION.md','protocol.json'):
    side=(R/('preregistration.sha256' if n.startswith('P') else 'protocol.sha256')).read_text().split()[0]
    checks[n+'_seal']=sha(R/n)==side
p=load('protocol.json'); q=load('query_inventory.json'); d=load('decision.json'); e=load('evidence_ledger.json'); a=load('archive_header_metadata.json'); m=load('metrics.json')
checks['exact_revisions']=p['immutable_revisions']==[x['commit'] for x in load('revision_inventory.json')['revisions']]
checks['linear_history']=load('revision_inventory.json')['linear_history']
checks['raw_response_hashes']=all(sha(R/'raw/metadata'/f"{x['name']}.body")==x['body_sha256'] and sha(R/'raw/headers'/f"{x['name']}.headers")==x['headers_sha256'] for x in load('raw_response_inventory.json')['responses'] if x['name']!='archive_prefix')
checks['archive_hash']=sha(R/'raw/archive/prefix.bin')==next(x for x in load('raw_response_inventory.json')['responses'] if x['name']=='archive_prefix')['body_sha256']
checks['request_cap']=q['total_http_requests_conservative']<=q['maximum_http_requests']
checks['payload_cap']=q['archive_payload_bytes']<=q['maximum_archive_payload_bytes']==16777216
checks['range_behavior']=next(x for x in load('raw_response_inventory.json')['responses'] if x['name']=='archive_prefix')['status_chain']==[302,206] and next(x for x in load('raw_response_inventory.json')['responses'] if x['name']=='archive_prefix')['content_range']=='bytes 0-1048575/46184624478'
pointer=(R/'raw/archive/lfs_pointer.txt').read_text()
checks['pointer_namespaces']=('oid sha256:3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600' in pointer and p['known_archive_namespaces']['git_oid']!='3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600' and p['known_archive_namespaces']['xet_hash']!='3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600')
block=(R/'raw/archive/first_tar_block.bin').read_bytes(); checks['tar_header']=len(block)==512 and a['checksum_valid'] and a['name']=='cube_single_expert.h5' and a['size']==101942558720 and not a['embedded_comment_digest_source_fields']
st=load('local_extracted_stat.json'); checks['packaging_match']=st['bytes']==a['size'] and st['mtime_unix']==a['mtime_unix']
checks['evidence_mapping']=e['binding_evidence_count']==0 and e['narrowing_evidence_count']==0 and d['decision']=='upload_metadata_contains_no_generator_binding'
checks['zero_prohibited']=m['hdf5_rows_opened']==m['cube_environment_executions']==m['policy_trajectories']==m['model_or_gate_rows']==m['v5_episodes']==0 and not m['v3_test_targets_opened'] and not m['combined_v3_cache_numpy_loaded']
prior=R.parent/'lewm_adaptive_compute_source_lock_adjudication'/'artifact_manifest.json'; expected='8b91cd6dafebda232b26c82d4d9cc6cb42603e2ba273d31aaa8512aad5224896'; checks['prior_manifest_preserved']=sha(prior)==expected
out={'independent_implementation_imported_collection':False,'checks':checks,'passed':all(checks.values())}
print(json.dumps(out,indent=2,sort_keys=True)); sys.exit(0 if out['passed'] else 1)
