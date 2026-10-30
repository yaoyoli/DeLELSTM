"""Approved external PM entry adaptation: one path and only the second model_list assignment."""
import ast,copy,hashlib,difflib
from pathlib import Path
A=Path('/Users/yao/DeLELSTM-track-a');ORIGINAL=A/'PM.py';DATA=A/'DATA/PM2.5'
WINDOWS=r'C:\clean\Explain_time_series\Code\DATA\Processed\pm'
NAMES=['Delelstm','IMV_full','IMV_tensor','Retain','LSTM']
DEVIATION='Controlled Execution Deviation — Restored Five-model Comparison'
def assignments(tree,name):
 return [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
def make_code():
 raw=ORIGINAL.read_bytes();tree=ast.parse(raw.decode('utf-8'),filename=str(ORIGINAL))
 paths=assignments(tree,'data_path');models=sorted(assignments(tree,'model_list'),key=lambda n:n.lineno)
 assert len(paths)==1 and paths[0].lineno==39 and ast.literal_eval(paths[0].value)==WINDOWS
 assert len(models)==2 and [n.lineno for n in models]==[91,92]
 assert ast.literal_eval(models[0].value)==NAMES and ast.literal_eval(models[1].value)==['Retain']
 patched=copy.deepcopy(tree);p=assignments(patched,'data_path')[0];m=sorted(assignments(patched,'model_list'),key=lambda n:n.lineno)[1]
 p.value=ast.copy_location(ast.Constant(str(DATA)),p.value)
 m.value=ast.copy_location(ast.List(elts=[ast.Constant(x) for x in NAMES],ctx=ast.Load()),m.value)
 ast.fix_missing_locations(patched)
 restored=copy.deepcopy(patched);assignments(restored,'data_path')[0].value=copy.deepcopy(paths[0].value)
 sorted(assignments(restored,'model_list'),key=lambda n:n.lineno)[1].value=copy.deepcopy(models[1].value)
 assert ast.dump(restored,include_attributes=True)==ast.dump(tree,include_attributes=True),'Unapproved AST change including location'
 lines=raw.decode('utf-8').splitlines(True);new=lines.copy()
 new[38]='    data_path = '+repr(str(DATA))+'\r\n'
 new[91]='    model_list = '+repr(NAMES)+'\r\n'
 proof={'original':str(ORIGINAL),'original_source_sha256':hashlib.sha256(raw).hexdigest(),'deviation_label':DEVIATION,'approved_assignments':[{'line':39,'target':'data_path','old':WINDOWS,'new':str(DATA)},{'line':92,'target':'model_list','old':['Retain'],'new':NAMES}],'first_model_list_line91_unchanged':True,'all_other_AST_nodes_and_locations_identical':True,'original_AST':ast.dump(tree,include_attributes=True,indent=2),'patched_AST':ast.dump(patched,include_attributes=True,indent=2),'original_source_file_unmodified':True,'global_file_read_patch':False,'mapped_exact_inputs':[str(DATA/n) for n in ['newX_train.csv','second_y.csv']],'compile_filename_preserves_original':str(ORIGINAL),'Seed_and_RNG_statements_unchanged':True,'original_PM_training_unchanged':True,'unified_diff':''.join(difflib.unified_diff(lines,new,fromfile='Frozen/PM.py',tofile='ExternalApprovedPMEntry'))}
 return compile(patched,str(ORIGINAL),'exec',dont_inherit=True,optimize=0),proof
