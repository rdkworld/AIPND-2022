python train.py flowers/ --arch alexnet --arch_type custom --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch densenet121 --arch_type custom --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch vgg13 --arch_type custom --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch alexnet --arch_type existing --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch densenet121 --arch_type existing --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch vgg13 --arch_type existing --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch resnet18 --arch_type existing --epochs 10 --gpu >> log.txt
python train.py flowers/ --arch resnet18 --arch_type custom --epochs 10 --gpu >> log.text
