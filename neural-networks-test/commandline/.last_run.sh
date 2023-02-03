python train.py -a squeezenet_1_1 -c cat_to_name.json data
python train.py -a squeezenet_1_1 -e 7 -m 0.7 -l 0.002 -c cat_to_name.json data -s data-test
python train.py -a squeezenet_1_1 -e 5 -m 0.7 -l 0.002 -c cat_to_name.json data -s checkpoints 
python test.py data checkpoints/model_1675355356-0251653.pth
python sanity.py data checkpoints/model_1675355356-0251653.pth -c cat_to_name.json -n 5
ls * */* | entr -rc python sanity.py data checkpoints/model_1675355356-0251653.pth -c cat_to_name.json -n 5
ls * */* | entr -rc python predict.py data/th-955986389.jpg checkpoints/model_1675355356-0251653.pth -c cat_to_name.json
