

conda remove -n cali --all
conda create -n cali python=3.10
conda activate cali

pip install torch torchvision pillow

python -m pip install fipy




pip install git+https://github.com/openai/CLIP.git



pip install "transformers>=4.40" accelerate torch
