#!/usr/bin/env python3
import hashlib, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REVS=json.loads((ROOT/'protocol.json').read_text())['immutable_revisions']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

commits={}
for line in (ROOT/'raw/git_commits.tsv').read_text().splitlines():
    f=line.split('\t')
    commits[f[0]]={'commit':f[0],'parents':f[1].split() if f[1] else [],'author':{'name':f[2],'email':f[3],'timestamp':f[4]},'committer':{'name':f[5],'email':f[6],'timestamp':f[7]},'signature_status':f[8],'signature_key':f[9] or None,'message':f[10]}

revision_rows=[]
prev=None
for rev in REVS:
    tree=json.loads((ROOT/f'raw/metadata/tree_{rev}.body').read_text())
    bypath={x['path']:x for x in tree}
    readme=ROOT/f'raw/metadata/readme_{rev}.body'
    row=commits[rev]|{'tree':tree,'siblings':sorted(bypath),'readme':{'bytes':readme.stat().st_size,'sha256':sha(readme),'git_blob':bypath['README.md']['oid']},'readme_diff_from_parent':None}
    if prev:
        a=(ROOT/f'raw/metadata/readme_{prev}.body').read_text(errors='replace').splitlines()
        b=readme.read_text(errors='replace').splitlines()
        import difflib
        row['readme_diff_from_parent']='\n'.join(difflib.unified_diff(a,b,fromfile=prev,tofile=rev,lineterm=''))+'\n'
    archive=bypath.get('cube_single_expert.tar.zst')
    row['archive']=None if not archive else {'file_action':'added' if rev==REVS[1] else 'unchanged','git_oid':archive['oid'],'lfs_sha256':archive['lfs']['oid'],'lfs_pointer_size':archive['lfs']['pointerSize'],'size':archive['size'],'xet_hash':archive['xetHash'],'last_commit':archive['lastCommit']['id']}
    row['gitattributes']={'git_blob':bypath['.gitattributes']['oid'],'bytes':bypath['.gitattributes']['size'],'sha256':sha(ROOT/'raw/metadata/gitattributes_de75ff3cef0851c208b849c58570d765a2ae0637.body')}
    revision_rows.append(row); prev=rev
dump('revision_inventory.json',{'revision_count':6,'linear_history':all((not x['parents'] if i==0 else x['parents']==[REVS[i-1]]) for i,x in enumerate(revision_rows)),'tags':[],'releases':[],'workflows':[],'revisions':revision_rows})

responses=[]
for body in sorted((ROOT/'raw/metadata').glob('*.body')):
    hdr=ROOT/'raw/headers'/(body.stem+'.headers')
    text=hdr.read_text(errors='replace') if hdr.exists() else ''
    statuses=[int(x) for x in re.findall(r'^HTTP/\S+ (\d+)',text,re.M)]
    def last(field):
        vals=re.findall(rf'^{re.escape(field)}:\s*(.+)$',text,re.I|re.M); return vals[-1].strip() if vals else None
    responses.append({'name':body.stem,'status_chain':statuses,'etag':last('etag'),'content_range':last('content-range'),'content_length':last('content-length'),'body_bytes':body.stat().st_size,'body_sha256':sha(body),'headers_sha256':sha(hdr) if hdr.exists() else None})
ah=ROOT/'raw/archive/prefix.headers'; ab=ROOT/'raw/archive/prefix.bin'
responses.append({'name':'archive_prefix','url':'https://huggingface.co/datasets/quentinll/lewm-cube/resolve/f6fe469578297a910bd4c88b9857f572908d7a34/cube_single_expert.tar.zst','request_range':'bytes=0-1048575','status_chain':[302,206],'etag_namespace':'xet_hash','etag':'ffe07ef65b18ea16455f304e1d22fc9e617381cd874576103419dea1064503bf','content_range':'bytes 0-1048575/46184624478','content_length':'1048576','body_bytes':ab.stat().st_size,'body_sha256':sha(ab),'headers_sha256':sha(ah),'redirect_count':1})
dump('raw_response_inventory.json',{'responses':responses})
dump('query_inventory.json',{'curl_logical_requests':23,'git_http_requests':3,'archive_redirect_http_requests':2,'total_http_requests_conservative':28,'maximum_http_requests':48,'archive_payload_bytes':ab.stat().st_size,'maximum_archive_payload_bytes':16777216,'all_urls_immutable_for_evidence':True,'failed_endpoint_requests':1})

header=json.loads((ROOT/'archive_header_metadata.json').read_text())
stat=json.loads((ROOT/'local_extracted_stat.json').read_text())
evidence=[
 {'id':'dataset_linear_history','class':'descriptive_only','finding':'Six-commit linear history; upload commit adds only the LFS pointer and card edits add only generic LeWorldModel links.'},
 {'id':'archive_hash_namespaces','class':'descriptive_only','finding':'Git blob 01e3f0…, LFS SHA-256 3725d6…, Xet hash ffe07e…, and size are consistent and distinct.'},
 {'id':'upload_signature','class':'descriptive_only','finding':'Upload commit contains a Hugging Face system RSA signature key 6A528E38E0733467; verification is unavailable without its public key and the signed object names no generator.'},
 {'id':'tar_header','class':'descriptive_only','finding':f"Valid tar header names {header['name']}, size {header['size']}, mtime {header['mtime_utc']}, uname/gname {header['uname']}; no embedded source/digest/comment/PAX field."},
 {'id':'local_packaging_match','class':'descriptive_only','finding':f"Local stat size and mtime match tar metadata ({stat['bytes']}, {stat['mtime_unix']}); packaging evidence only."},
 {'id':'generator_binding','class':'none','finding':'No exact source commit/coherent same-commit lock or container digest/build record is explicitly tied to the exact archive hash or upload revision.'},
 {'id':'generator_narrowing','class':'none','finding':'No collector command/config, dependency versions, environment variables, seed, renderer/backend, or source tag is explicitly tied to the exact archive.'}
]
dump('evidence_ledger.json',{'binding_evidence_count':0,'narrowing_evidence_count':0,'descriptive_evidence_count':5,'entries':evidence})
decision={'decision':'upload_metadata_contains_no_generator_binding','binding_evidence_count':0,'narrowing_evidence_count':0,'immutable_revisions_reconstructed':6,'archive_header_available':True,'archive_payload_bytes':1048576,'request_cap_respected':True,'payload_cap_respected':True,'mechanics_replayed':False,'close_local_generator_archaeology':True}
dump('decision.json',decision)
dump('metrics.json',{'http_requests_conservative':28,'http_request_cap':48,'archive_payload_bytes':1048576,'archive_payload_cap':16777216,'revision_count':6,'signed_commits_unverified_public_key_absent':5,'unsigned_commits':1,'archive_members_inspected':1,'hdf5_rows_opened':0,'cube_environment_executions':0,'v3_test_targets_opened':False,'combined_v3_cache_numpy_loaded':False,'policy_trajectories':0,'model_or_gate_rows':0,'v5_episodes':0})
