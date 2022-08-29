#Prior to Training
#Pip installs
pip install transformers huggingface_hub tensorboard matplotlib sklearn
pip install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113
pip install -q torchinfo

#Clone Model Repository
# rm -rf runs
git clone https://rdkulkarni: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@huggingface.co/rdkulkarni/flower-models-repo
cd flower-models-repo
git init
cd ..

#Clone Model Inference Repository
git clone https://rdkulkarni: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@huggingface.co/spaces/rdkulkarni/which-flower
cd which-flower
git init
cd ..

#Clone Custom Modules from Github
#mkdir flowers
git clone https://github.com/rdkworld/AIPND-2022 
mv AIPND-2022/Generalized/*.py flowers/
rm -rf AIPND-2022

#Training
python generic_train.py

#Post Training
## 1. Push both models and training metrics to Huggingface Models (Model Repository)
cd flower-models-repo
git lfs install
cp ../flowers/models/*.pth .
cp -r ../runs .
git config user.email "rdkulkarni@gmail.com"
git config user.name "Raghavendra Kulkarni"
git add . 
git commit -m "Updated model and training metrics"
git push
cd ..
rm -rf flower-models-repo
#rm -rf runs

## 2. Push only models to to Huggingface Space (Model Hosting)
cd which-flower
git lfs install
cp ../flowers/models/*.pth .
git config user.email "rdkulkarni@gmail.com"
git config user.name "Raghavendra Kulkarni"
git add . 
git commit -m "Updated model changes"
git push
cd ..
rm -rf which-flower