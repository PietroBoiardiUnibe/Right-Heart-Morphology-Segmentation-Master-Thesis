#!/bin/bash
# Where do I stand on UBELIX?   bash ~/code/ImageSegmentation/hpc/rh100_status.sh
RH=~/data/inselspital/rh100
LIST=$RH/qc/included_cases.txt

echo "== my jobs in the queue (R running, PD pending)"
squeue --me -o "%.14i %.10P %.14j %.4t %.10M %R" | head -15

echo
echo "== my jobs of the last 3 days (COMPLETED / FAILED / TIMEOUT / OUT_OF_MEMORY)"
sacct -S now-3days --format=JobID%16,JobName%14,State%14,Elapsed%10,MaxRSS%10 -X | head -25

total=$(tr -d '\r' < "$LIST" | grep -c .)
finished=0; passed=0; rims_ok=0
for id in $(tr -d '\r' < "$LIST"); do
  c=$RH/cases/$id
  [ -f "$c/rims.json" ] && finished=$((finished + 1))
  grep -q '"included": true' "$c/surface_chambers_qc.json" 2>/dev/null && passed=$((passed + 1))
  [ "$(grep -c '"status": "ok"' "$c/rims.json" 2>/dev/null)" = "4" ] && rims_ok=$((rims_ok + 1))
done
echo
echo "== per-case progress (included cases: $total)"
echo "   05b + 06 + 06b finished (rims.json present): $finished"
echo "   06 surface QC passed:                        $passed"
echo "   all 4 rims ok (06b):                         $rims_ok"

echo
echo "== job logs containing a Python error"
grep -l Traceback "$RH"/logs/surf_*.out 2>/dev/null | head -10
echo "(nothing above = no errors)"

echo
echo "== disk"
du -sh "$RH/cases" "$RH/logs" 2>/dev/null
