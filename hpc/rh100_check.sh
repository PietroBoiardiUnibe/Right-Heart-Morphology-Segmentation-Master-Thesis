#!/bin/bash
# Is the cluster state clean and final?   bash ~/code/ImageSegmentation/hpc/rh100_check.sh
RH=~/data/inselspital/rh100
L=$RH/logs
LIST=$RH/qc/included_cases.txt

echo "== queue (empty = nothing running)"
squeue --me

echo
echo "== log files per job, and how many contain a Python traceback"
for job in $(ls $L | sed 's/^\(surf\|rims\)_\([0-9]*\)_.*/\1_\2/' | sort -u); do
  n=$(ls $L/${job}_*.out | wc -l)
  t=$(grep -l Traceback $L/${job}_*.out 2>/dev/null | wc -l)
  e=$(find $L -name "${job}_*.out" -size 0 | wc -l)
  echo "$job: $n logs, $t with traceback, $e empty"
done

echo
f=$(grep -l Traceback $L/*.out 2>/dev/null | head -1)
if [ -n "$f" ]; then
  echo "== last 15 lines of the first log with a traceback: $f"
  tail -15 "$f"
else
  echo "== no traceback in any log"
fi

echo
echo "== outputs per case (included cases: $(tr -d '\r' < $LIST | grep -c .))"
echo "rims.json with anchor:   $(grep -l '"anchor"' $RH/cases/*/rims.json 2>/dev/null | wc -l)"
echo "rims.json status ok x4:  $(for i in $(tr -d '\r' < $LIST); do [ "$(grep -c '"status": "ok"' $RH/cases/$i/rims.json)" = "4" ] && echo $i; done | wc -l)"
echo "surface QC included:     $(grep -l '"included": true' $RH/cases/*/surface_chambers_qc.json 2>/dev/null | wc -l)"
echo "repaired surface QC key: $(grep -l repaired_vertices $RH/cases/*/surface_chambers_qc.json 2>/dev/null | wc -l)"

echo
echo "== code fingerprints (line-ending free, first 8 chars)"
for f in 05b_interface_orifices.py 06_labelled_surface.py 06b_rims.py; do
  echo "$(tr -d '\r' < ~/code/ImageSegmentation/scripts/$f | md5sum | cut -c1-8) $f"
done

echo
echo "== data fingerprints of per-case outputs"
ids=$(tr -d '\r' < $LIST | LC_ALL=C sort)
for f in orifices.json surface_chambers.vtp surface_chambers_qc.json rims.json rims_qc.png; do
  printf "%-26s" $f
  for i in $ids; do md5sum $RH/cases/$i/$f | cut -d' ' -f1; done | md5sum | cut -c1-12
done
