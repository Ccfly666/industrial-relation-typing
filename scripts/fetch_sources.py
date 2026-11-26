"""RECONSTRUCTED_REPRODUCTION_TOOL. Does not replace the historical acquisition script."""
from pathlib import Path,PurePosixPath
import hashlib,json,urllib.request,zipfile,stat,argparse,shutil
DOI='10.5281/zenodo.17091043'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe_extract(archive,dest):
 dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   name=PurePosixPath(info.filename.replace('\\','/'))
   target=(dest/str(name)).resolve()
   if name.is_absolute() or '..' in name.parts or ':' in str(name) or dest not in target.parents or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe ZIP member: '+info.filename)
  z.extractall(dest)
def obtain(dest,expected,local=None,url=None):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if sha(dest)!=expected:raise ValueError('Existing source checksum mismatch')
  return 'EXISTING_VERIFIED'
 if local:
  if sha(local)!=expected:raise ValueError('Local source checksum mismatch')
  shutil.copyfile(local,dest);return 'LOCAL_SNAPSHOT_VERIFIED'
 if not url:raise ValueError('A licensed local snapshot or official URL is required')
 tmp=dest.with_suffix(dest.suffix+'.partial')
 with urllib.request.urlopen(url,timeout=60) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
 if sha(tmp)!=expected:raise ValueError('Downloaded source checksum mismatch; partial retained')
 tmp.replace(dest);return 'ONLINE_VERIFIED'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sha256',required=True);p.add_argument('--local');p.add_argument('--url');p.add_argument('--extract',action='store_true');a=p.parse_args()
 state=obtain(a.out,a.sha256,a.local,a.url)
 manifest=dict(state=state,sha256=sha(a.out),doi=DOI,source_url=a.url,source_mode='local snapshot' if a.local else 'explicit URL or verified existing artifact',license='Consult DOI metadata and upstream taxonomy terms; see DATA_ACCESS.md')
 if a.extract:
  directory=Path(a.out).with_suffix('');safe_extract(a.out,directory);names={'etim-eclass','eclass-gpc','eclass-unspsc','gpc-unspsc','gpc-unspscplus'};manifest['datasets']=[]
  for reference in sorted(directory.rglob('reference.rdf')):
   if reference.parent.name not in names:continue
   manifest['datasets'].append(dict(task=reference.parent.name,role='conflict-screening variant' if reference.parent.name=='gpc-unspscplus' else 'formal task source',files=[dict(member=str((reference.parent/n).relative_to(directory)),sha256=sha(reference.parent/n)) for n in ['source.rdf','target.rdf','reference.rdf']]))
 Path(a.out).with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))
if __name__=='__main__':main()
