#!/bin/bash
# Can ShapeWorks run on UBELIX?   bash ~/code/ImageSegmentation/hpc/rh100_sw_check.sh
SW=~/software/ShapeWorks-v6.7.0-linux

echo "== ShapeWorks folder"
ls $SW | head -20
echo
echo "== executables"
ls $SW/bin | head -20
echo
echo "== shapeworks --version"
$SW/bin/shapeworks --version 2>&1 | head -5
echo
echo "== python module (system python of the venv)"
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rh/bin/activate
python -c "import shapeworks as sw; print('import ok:', sw.__file__)" 2>&1 | tail -3
echo
echo "== python module with ShapeWorks' own Python path"
export PYTHONPATH=$SW/bin:$PYTHONPATH
export LD_LIBRARY_PATH=$SW/lib:$LD_LIBRARY_PATH
python -c "import shapeworks as sw; print('import ok:', sw.__file__)" 2>&1 | tail -3
echo
echo "== python module with the package folder (Python/) AND the compiled bin/ on the path"
export PYTHONPATH=$SW/Python:$SW/bin
python -c "
import shapeworks as sw
need = ['Project', 'Subject', 'Parameters', 'Variant', 'utils']
print('import ok')
print('names 01 needs:', {n: hasattr(sw, n) for n in need})
p = sw.Project()
print('empty Project() created:', type(p).__name__)
import shapeworks_py
print('shapeworks_py also importable:', shapeworks_py.__file__)
" 2>&1 | tail -8
echo
echo "== what the Python/ folder holds"
ls $SW/Python | head -20
echo
echo "== bundled python"
ls $SW/bin | grep -i python
ls $SW | grep -i -E "python|conda"
echo
echo "== cores and memory on a compute node (per epyc2 node, for sizing the job)"
sinfo -p epyc2 -o "%P %c %m %D" | head -3
