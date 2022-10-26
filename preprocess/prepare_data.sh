python prepare_data.py --data '../../datasets/mgn/' --views 360 --cam 'orth'

cd ../../datasets/mgn/125611487366942/
cp norm_render_360/0.png norm_render_360/90.png norm_render_360/180.png norm_render_360/270.png ../../../ARCH/logs/
cd ../../../ARCH/
git add .
git commit -m "render complete"
git push
