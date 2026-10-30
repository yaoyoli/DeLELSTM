"""Run frozen Electricity entry with exactly one approved path constant substituted."""
import ast,copy,hashlib,json,difflib
from pathlib import Path
A=Path('/Users/yao/DeLELSTM-track-a')
ORIGINAL=A/'Electricity.py'
DATA=A/'DATA/Electricity'
WINDOWS=r'C:\clean\Explain_time_series\Code\DATA\Processed\elec'
def make_code():
 raw=ORIGINAL.read_bytes();tree=ast.parse(raw.decode('utf-8'),filename=str(ORIGINAL))
 nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='data_path']
 assert len(nodes)==1 and nodes[0].lineno==36 and isinstance(nodes[0].value,ast.Constant) and nodes[0].value.value==WINDOWS,'Unapproved entry path/source'
 transformed=copy.deepcopy(tree)
 node=next(n for n in ast.walk(transformed) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='data_path' for t in n.targets))
 old=node.value.value;node.value=ast.copy_location(ast.Constant(value=str(DATA)),node.value)
 restored=copy.deepcopy(transformed)
 restore=next(n for n in ast.walk(restored) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='data_path' for t in n.targets));restore.value.value=old
 assert ast.dump(restored,include_attributes=False)==ast.dump(tree,include_attributes=False),'Extra AST change'
 ast.fix_missing_locations(transformed)
 lines=raw.decode('utf-8').splitlines(keepends=True);new=lines.copy()
 new[35]="    data_path = "+repr(str(DATA))+"\r\n"
 proof={'original':str(ORIGINAL),'original_source_sha256':hashlib.sha256(raw).hexdigest(),'only_assignment_line':36,'old_literal':WINDOWS,'new_literal':str(DATA),'mapped_exact_inputs':[str(DATA/n) for n in ['newX_train.csv','second_y.csv']],'all_other_AST_nodes_identical':True,'original_source_file_unmodified':True,'global_file_read_patch':False,'pandas_read_csv_replaced':False,'compile_filename_preserves_original':str(ORIGINAL),'data_and_RNG_statements_unchanged':True,'unified_diff':''.join(difflib.unified_diff(lines,new,fromfile='Frozen/Electricity.py',tofile='ExternalPathAdaptedEntry'))}
 return compile(transformed,str(ORIGINAL),'exec',dont_inherit=True,optimize=0),proof
