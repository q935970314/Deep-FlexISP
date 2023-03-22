#!/bin/bash
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
_=/usr/bin/env

echo " "
echo "############################"
echo "#### stage0 pre_process ####"
echo "############################"
cd /workdir/stage0
python pre_process.py


echo " "
echo "############################"
echo "###### stage1 denoise ######"
echo "############################"
cd /workdir/stage1
python denoise.py


echo " "
echo "############################"
echo "### stage2 white_balance ###"
echo "############################"
cd /workdir/stage2
python white_balance.py


echo " "
echo "############################"
echo "###### stage3 raw2rgb ######"
echo "############################"
cd /workdir/stage3
python test_simple_isp.py


echo " "
echo "############################"
echo "###### stage4 refine #######"
echo "############################"
cd /workdir/stage4
python test_night_render_2023.py


echo " "
echo "############################"
echo "# stage5 super-resolution  #"
echo "############################"
cd /workdir/stage5
bash test.sh


echo " "
echo "############################"
echo "### processing completed ###"
echo "############################"
cd /workdir


