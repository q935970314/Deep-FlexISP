

echo " "
echo "############################"
echo "#### stage0 pre_process ####"
echo "############################"
cd ./stage0
# python pre_process.py
cd ..


echo " "
echo "############################"
echo "###### stage1 denoise ######"
echo "############################"
cd ./stage1
# python denoise.py
cd ..


echo " "
echo "############################"
echo "### stage2 white_balance ###"
echo "############################"
cd ./stage2
# python white_balance.py
cd ..


echo " "
echo "############################"
echo "###### stage3 raw2rgb ######"
echo "############################"
cd ./stage3
python test_night_render.py
cd ..


echo " "
echo "############################"
echo "### processing completed ###"
echo "############################"



