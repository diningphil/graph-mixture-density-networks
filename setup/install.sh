# default pytorch version is 1.6.0
PYTORCH_VERSION=1.7.0
PYTORCH_GEOMETRIC_VERSION=1.6.0

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.7.0 are cpu, cu92, cu101, cu102, cu110
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
conda create --name gmdn python=3.7 -y
conda activate gmdn

# install requirements
pip install -r setup/requirements.txt

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  conda install pytorch==${PYTORCH_VERSION} cpuonly -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu92' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=9.2 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu101' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.1 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu102' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.2 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu110' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=11.0 -c pytorch -y
fi

conda install jupyter

# install torch-geometric dependencies
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==${PYTORCH_GEOMETRIC_VERSION}

echo "Done. Remember to "
echo " 1) append the anaconda/miniconda lib path to the LD_LIBRARY_PATH variable using the export command"
echo " 2) export the variable DGLBACKEND as pytorch"
echo "Modify the .bashrc file to make permanent changes."
