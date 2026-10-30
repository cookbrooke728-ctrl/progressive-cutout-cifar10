import json
import shutil
import numpy as np
import pytest
import torch
from cutout_lab.weights import resolve_weight, restore_runs, suite_action
from cutout_lab.utils import sha256, metrics
from cutout_lab.model import ResNet20

@pytest.fixture
def published(tmp_path):
    run = tmp_path / 'runs/baseline_seed42'; run.mkdir(parents=True)
    weights = tmp_path / 'weights'; weights.mkdir()
    config = {'method':'baseline','seed':42,'split_sha256':'test-only','normalization':{}}
    path = weights / 'baseline_seed42_best.pt'
    torch.save({'state_dict':ResNet20().state_dict(),'config':config,'best_epoch':1,'validation_accuracy':10.},path)
    status = {'method':'baseline','seed':42,'complete':True,'best_epoch':1,'best_weight_sha256':sha256(path)}
    for name, value in [('config',config),('status',status),('provenance',{}),('initialization',{})]:
        (run / (name+'.json')).write_text(json.dumps(value))
    (run/'history.csv').write_text('epoch\n1\n')
    (run/'evaluation.json').write_text(json.dumps({'weight_sha256':sha256(path),'sentinel':'original'}))
    (weights/'manifest.json').write_text(json.dumps({'weights':[{'run':run.name,'filename':path.name,'sha256':sha256(path)}]}))
    return run, weights, path

def test_only_published_weights_restore_and_preserve_evidence(published, tmp_path):
    run,w,path=published
    assert resolve_weight(run)==path
    out=tmp_path/'new/runs'
    restore_runs(run.parent,w,out)
    assert (out/run.name/'best.pt').read_bytes()==path.read_bytes()
    assert not (out/run.name/'evaluation.json').exists()
    assert json.loads((run/'evaluation.json').read_text())['sentinel']=='original'
    with pytest.raises(FileExistsError): restore_runs(run.parent,w,out)

def test_missing_paths_fail(published):
    run,w,path=published
    path.unlink()
    with pytest.raises(FileNotFoundError,match='Missing weight'): resolve_weight(run)
    (w/'manifest.json').unlink()
    with pytest.raises(FileNotFoundError,match='manifest'): resolve_weight(run)

def test_bad_hash_fails_even_with_cached_evaluation(published):
    from cutout_lab.engine import evaluate_run
    run,w,path=published
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA-256'): evaluate_run(None,run)

def test_manifest_and_checkpoint_identity(published):
    run,w,path=published
    manifest=json.loads((w/'manifest.json').read_text())
    manifest['weights'][0]['filename']='cutout_seed42_best.pt'
    (w/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'): resolve_weight(run)
    shutil.copyfile(path,run/'best.pt')
    cp=torch.load(path,weights_only=True);cp['config']['seed']=43;torch.save(cp,run/'best.pt')
    status=json.loads((run/'status.json').read_text());status['best_weight_sha256']=sha256(run/'best.pt');(run/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='Checkpoint'): resolve_weight(run)

def test_completed_suite_checks_weights_and_resume_requires_last(published):
    run,w,path=published
    assert suite_action(run)=='skip_verified_complete'
    path.unlink()
    with pytest.raises(FileNotFoundError): suite_action(run)
    status=json.loads((run/'status.json').read_text());status['complete']=False;(run/'status.json').write_text(json.dumps(status))
    with pytest.raises(FileExistsError): suite_action(run)
    with pytest.raises(FileNotFoundError,match='last.pt'): suite_action(run,True)
    (run/'last.pt').write_bytes(b'only existence checked by planner')
    assert suite_action(run,True)=='resume'
    assert suite_action(run.parent/'cutout_seed44')=='train'

def test_force_runs_all_ten_conditions(published,monkeypatch):
    import cutout_lab.engine as engine
    run,w,path=published
    calls=[]
    def predict(*args,**kwargs):
        calls.append(kwargs.get('length',0))
        return {**metrics(np.arange(10),np.arange(10)),'loss':0.},np.arange(10),np.ones(10)
    monkeypatch.setattr(engine,'predict',predict)
    class Data:
        device=torch.device('cpu');split_hash='test-only';stats={}
        test=torch.zeros(10,3,32,32);test_labels=torch.arange(10)
        def load_test(self): pass
    assert engine.evaluate_run(Data(),run)['sentinel']=='original'
    assert calls==[]
    engine.evaluate_run(Data(),run,force=True)
    assert calls==[0,8,8,8,16,16,16,24,24,24]

def test_cli_missing_runs_fails_before_dataset(tmp_path,monkeypatch):
    import sys
    from cutout_lab.cli import main
    monkeypatch.setattr(sys,'argv',['cutout_lab','evaluate','--runs',str(tmp_path/'absent')])
    with pytest.raises(FileNotFoundError,match='No run metadata'): main()

def test_restore_rejects_missing_run_metadata(published,tmp_path):
    run,w,path=published
    m=json.loads((w/'manifest.json').read_text());m['weights'].append({'run':'cutout_seed43'})
    (w/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='run set mismatch'): restore_runs(run.parent,w,tmp_path/'new')
