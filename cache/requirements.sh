ml purge

ml conda

# Check if conda environment exists, create only if missing
if conda env list | grep -q "^mmgraphrag "; then
    echo "Environment 'mmgraphrag' already exists, activating..."
else
    echo "Creating new environment 'mmgraphrag'..."
    conda create -n mmgraphrag -y python=3.10
    pip install -r requirements.txt
fi

conda activate mmgraphrag
# mmgraphrag环境

# pip install openai
# pip install sentence-transformers
# pip install nano-vectordb
# pip install python-docx
# pip install PyMuPDF
# pip install ultralytics
# pip install tiktoken

pip install -U "mineru[core]"


