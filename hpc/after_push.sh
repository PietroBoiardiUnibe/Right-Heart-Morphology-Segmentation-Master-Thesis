#!/bin/bash
# Runs on UBELIX right after push_code.ps1 unpacked the code: fix permissions and line endings, write the fingerprint list.
cd ~/code || exit 1
chmod -R u+rwX ~/code
find ImageSegmentation/scripts ImageSegmentation/hpc ShapeAnalysis/scripts ShapeAnalysis/hpc \
     -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) ! -path '*/_archive/*' ! -path '*/__pycache__/*' \
     -exec sed -i 's/\r$//' {} +
find ImageSegmentation/scripts ImageSegmentation/hpc ShapeAnalysis/scripts ShapeAnalysis/hpc \
     -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) ! -path '*/_archive/*' ! -path '*/__pycache__/*' \
     | sed 's#^#./#' | LC_ALL=C sort | while read -r f; do printf '%s  %s\n' "$(md5sum < "$f" | cut -c1-32)" "$f"; done > ~/code/CODE_MANIFEST.md5
echo "files: $(wc -l < ~/code/CODE_MANIFEST.md5)"
echo "manifest: $(md5sum ~/code/CODE_MANIFEST.md5 | cut -c1-8)"
