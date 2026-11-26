"""RECONSTRUCTED_REPRODUCTION_TOOL: RDF/XML and Alignment API audit, no fixed counts."""
from pathlib import Path
from collections import defaultdict,Counter
import xml.etree.ElementTree as ET,hashlib,json,csv,argparse
from urllib.parse import urljoin
RDF='http://www.w3.org/1999/02/22-rdf-syntax-ns#';RDFS='http://www.w3.org/2000/01/rdf-schema#';OWL='http://www.w3.org/2002/07/owl#';XML='http://www.w3.org/XML/1998/namespace'
LABELS=['=','>','<','~'];REVERSE={'=':'=','>':'<','<':'>','~':'~'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def parse(path):
 data=Path(path).read_bytes()
 if b'<!DOCTYPE' in data or b'<!ENTITY' in data:raise ValueError('External/DTD entities are not allowed')
 return ET.fromstring(data)
def concepts(path):
 root=parse(path);base=root.get('{'+XML+'}base','');out={}
 for e in root.iter():
  uri=e.get('{'+RDF+'}about')
  declared=e.tag=='{'+OWL+'}Class' or any(x.tag=='{'+RDF+'}type' and x.get('{'+RDF+'}resource')=='http://www.w3.org/2002/07/owl#Class' for x in e)
  if not declared or uri is None:continue
  uri=urljoin(base,uri);q=out.setdefault(uri,dict(uri=uri,labels=[],parents=[]))
  for c in e:
   if c.tag=='{'+RDFS+'}label':q['labels'].append(dict(text=(c.text or '').strip(),language=c.get('{'+XML+'}lang','unspecified')))
   if c.tag=='{'+RDFS+'}subClassOf' and c.get('{'+RDF+'}resource') is not None:q['parents'].append(urljoin(base,c.get('{'+RDF+'}resource')))
 for uri,q in out.items():q['name']=next((x['text'] for x in q['labels'] if x['language'].lower().startswith('en')),q['labels'][0]['text'] if q['labels'] else '');q['parents']=list(dict.fromkeys(q['parents']))
 # Longest declared-parent DAG path; reject cycles rather than invent a historical cycle convention.
 done={};active=set();cycles=[]
 def depth(u):
  if u in done:return done[u]
  if u in active:raise ValueError('Hierarchy cycle at '+u)
  active.add(u);parents=[p for p in out[u]['parents'] if p in out];v=max((depth(p)+1 for p in parents),default=0);active.remove(u);done[u]=v;return v
 for u in out:out[u]['depth']=depth(u);out[u]['missing_named_parent']=[p for p in out[u]['parents'] if p not in out]
 return out
def assertions(path,task):
 root=parse(path);base=root.get('{'+XML+'}base','');fingerprint=sha(path);out=[]
 for e in root.iter():
  if e.tag.rsplit('}',1)[-1]!='Cell':continue
  q={c.tag.rsplit('}',1)[-1]:c for c in e}
  if not all(k in q for k in ['entity1','entity2','relation']):raise ValueError('Malformed Alignment Cell')
  i=len(out);s=urljoin(base,q['entity1'].get('{'+RDF+'}resource',''));t=urljoin(base,q['entity2'].get('{'+RDF+'}resource',''));r=(q['relation'].text or '').strip()
  out.append(dict(task=task,cell_index=i,source_uri=s,target_uri=t,relation=r,reference_sha256=fingerprint))
 return out
def clean(records,source,target):
 pairs=defaultdict(list)
 for r in records:pairs[r['source_uri'],r['target_uri']].append(r)
 good=[];issues=[];same=0
 for (s,t),rr in pairs.items():
  labels=set(r['relation'] for r in rr);same+=len(rr)-len(labels)
  reason='multi_label_pair' if len(labels)>1 else 'invalid_relation' if not labels<=set(LABELS) else 'undeclared_endpoint' if s not in source or t not in target else None
  if reason:issues.append(dict(source=s,target=t,reason=reason,raw_labels='|'.join(r['relation'] for r in rr)));continue
  good.append(dict(source=s,target=t,relation=rr[0]['relation']))
 return good,issues,dict(raw=len(records),unique_pairs=len(pairs),conflict_pairs=sum(x['reason']=='multi_label_pair' for x in issues),other_invalid=sum(x['reason']!='multi_label_pair' for x in issues),same_label_duplicates=same,clean=len(good))
def writecsv(p,rows,fields=None):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]) if rows else fields or ['status']);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--task',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.input);o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
 s,t=concepts(d/'source.rdf'),concepts(d/'target.rdf');rr=assertions(d/'reference.rdf',a.task);good,issues,summary=clean(rr,s,t)
 writecsv(o/'raw_assertions.csv',rr);writecsv(o/'clean_pairs.csv',good);writecsv(o/'issues.csv',issues)
 (o/'concepts.json').write_text(json.dumps(dict(source=s,target=t),ensure_ascii=False),encoding='utf-8');(o/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary)
if __name__=='__main__':main()
