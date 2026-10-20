# Full pipeline. Run from the project root after creating the environment (see README).
# On Windows keep the venv on a short path (e.g. C:\venvs\recfe): torch and pip exceed 260 chars otherwise.
param([string]$Py = "python")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force reports\logs | Out-Null

function Step($mod, $log) {
    Write-Host ">> $mod"
    & $Py -u -I -X utf8 -m $mod.Split(" ")[0] $mod.Split(" ")[1..9] *> "reports\logs\$log.log"
    if ($LASTEXITCODE -ne 0) { throw "failed: $mod (see reports\logs\$log.log)" }
}

Step "recfe.data.download" "download"
Step "recfe.data.prepare_yambda" "prepare_yambda"
Step "recfe.data.prepare_kion" "prepare_kion"
Step "recfe.tasks.yambda" "rows_yambda"
Step "recfe.tasks.kion" "rows_kion"
& $Py -I -X utf8 tests\test_engine.py

foreach ($ds in "yambda", "kion") {
    Step "recfe.experiments.baseline $ds" "baseline_$ds"
    Step "recfe.experiments.autofe_run $ds autofe" "autofe_$ds"
    Step "recfe.experiments.autofe_run $ds manual_autofe" "manual_autofe_$ds"
    Step "recfe.experiments.controls $ds random" "random_$ds"
    Step "recfe.experiments.controls $ds grid" "grid_$ds"
    Step "recfe.experiments.window_sweep $ds" "window_$ds"
    Step "recfe.graph.train $ds gnn" "gnn_$ds"
    Step "recfe.graph.train $ds gnn_emb" "gnn_emb_$ds"
    Step "recfe.graph.train $ds lgcn_bpr" "lgcn_$ds"
    Step "recfe.experiments.hybrid $ds" "hybrid_$ds"
    Step "recfe.experiments.significance $ds" "significance_$ds"
    Step "recfe.experiments.seeds $ds" "seeds_$ds"
}
Step "recfe.experiments.report" "report"
