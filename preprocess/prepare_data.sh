python prepare_data.py --data '../../MGNData/' --views 360 --cam 'orth'

cd ../../MGNData/125611487366942/
cp norm_render_360/0.png norm_render_360/90.png norm_render_360/180.png norm_render_360/270.png ../../CMPT985/logs/
cd ../../CMPT985/
git add .
git commit -m "render complete"
git push
