# Full pipeline. Run from the project root after creating the environment (see README).
# On Windows keep the venv on a short path (e.g. C:\venvs\recfe): torch and pip exceed 260 chars otherwise.
param([string]$Py = "python")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force reports\logs | Out-Null

# Each step runs through cmd so that warnings on stderr go to the log instead of becoming
# PowerShell errors; success is decided by the exit code only.
function Run([string]$cmdline, [string]$log) {
    Write-Host ">> $cmdline"
    cmd /c "`"$Py`" -u -I -X utf8 $cmdline > reports\logs\$log.log 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "failed: $cmdline (see reports\logs\$log.log)" }
}

function Step([string]$module, [string]$log) { Run "-m $module" $log }

Step "recfe.data.download" "download"
Step "recfe.data.prepare_yambda" "prepare_yambda"
Step "recfe.data.prepare_kion" "prepare_kion"
Step "recfe.tasks.yambda" "rows_yambda"
Step "recfe.tasks.kion" "rows_kion"
Run "tests\test_engine.py" "tests"

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
