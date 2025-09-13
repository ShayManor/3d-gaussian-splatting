#python -m venv .venv
#source .venv/bin/activate
#
#pip install --upgrade pip
#pip install nerfstudio gsplat
sudo apt-get update && sudo apt-get install -y ffmpeg colmap
sudo apt-get update && sudo apt-get install -y xvfb

# Processes video into images, scene
xvfb-run -a ns-process-data video \
  --data src/content/input/video.mp4 \
  --output-dir src/content/scene \
  --num-frames-target 120

rm -rf src/content/outputs/scene/splatfacto/*

# Trains gaussian splatting model
ns-train splatfacto --data src/content/scene \
--output-dir src/content/outputs

# Convert format to viewable form
ns-export gaussian-splat \
--load-config src/content/outputs/scene/splatfacto/2025-09-13_201337/config.yml \
--output-dir src/content/exports